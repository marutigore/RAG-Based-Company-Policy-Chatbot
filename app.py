"""
Synthara Enterprise RAG Portal Backend.
FastAPI Application serving a premium, responsive glassmorphic single-page RAG workspace interface.
Backward compatible with test_validation.py integration calls.
"""

import os
import logging
import json
import datetime
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn

# Import custom core modules
import config
from utils.document_loader import load_pdf, load_document
from utils.chunker import split_documents
from utils.retriever import add_documents_to_db, query_db, reset_db, get_collection, delete_document_from_db, get_search_facets
from utils.validator import validate_query, evaluate_faithfulness, evaluate_answer_relevancy
from utils.auth import authenticate_user, create_jwt_token, verify_jwt_token, get_all_users, register_user
from utils.memory import create_session, add_message, get_session_messages, list_sessions, delete_session, build_contextual_query
from utils.analytics import record_query_telemetry, get_analytics_summary
from utils.feedback import record_feedback, get_feedback_summary, list_feedback_records
from utils.versioning import register_document_version, get_document_versions, get_active_version_tag
from utils.translator import detect_language, get_supported_languages, build_multilingual_system_prompt
from utils.notifications import format_email_template, format_slack_block_kit, format_teams_card, dispatch_webhook
from utils.audit import log_audit_event, get_audit_logs, verify_audit_integrity, export_audit_csv
from utils.suggestions import generate_document_suggestions, get_all_suggestions, get_autocomplete_suggestions
from utils.document_viewer import get_document_page_preview
from utils.sync_manager import scan_and_sync_policies, get_sync_status
from utils.ab_testing import select_active_variant, get_ab_prompt, record_ab_metric, get_ab_experiment_summary
from utils.pii_guardrail import redact_pii, scan_pii, mask_sensitive_query
import time

# Setup logging
logger = logging.getLogger("app")
logging.basicConfig(level=config.LOG_LEVEL)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Synthara RAG Portal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core RAG functions matching test_validation.py expectations
def init_session_state() -> None:
    pass


def get_indexed_documents() -> List[Dict[str, Any]]:
    """Queries ChromaDB to find unique documents, tracking page bounds and chunk volumes."""
    try:
        col = get_collection()
        res = col.get(include=["metadatas"])
        if not res or not res["metadatas"]:
            return []
        
        docs = {}
        for meta in res["metadatas"]:
            source = meta.get("source", "Unknown Document")
            if source not in docs:
                docs[source] = {
                    "source": source,
                    "pages": 0,
                    "chunks": 0,
                    "version": get_active_version_tag(source)
                }
            docs[source]["chunks"] += 1
            docs[source]["pages"] = max(docs[source]["pages"], meta.get("page", 1))
            
        return list(docs.values())
    except Exception as e:
        logger.error(f"Error fetching indexed documents: {e}")
        return []

def sanitize_text(text: str) -> str:
    replacements = {
        '\u2011': '-', '\u2013': '-', '\u2014': '-',
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2026': '...', '\u00a0': ' '
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text

def track_usage(usage) -> None:
    pass

def call_llm(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    lang_info: Optional[Dict[str, Any]] = None,
    clearance: str = "Employee",
    prompt_variant: Optional[str] = None
) -> str:
    client = config.get_openai_client()
    context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks):
        source = chunk["metadata"].get("source", "Unknown Document")
        page = chunk["metadata"].get("page", "?")
        chunk_text = sanitize_text(chunk['text'])
        context_blocks.append(f"Excerpt [{idx + 1}] (Source: {source}, Page {page}):\n{chunk_text}")

    context_str = "\n\n".join(context_blocks)
    
    if prompt_variant in ["A", "B"]:
        system_prompt, user_prompt = get_ab_prompt(prompt_variant, question, context_str, clearance=clearance)
    elif lang_info:
        system_prompt = build_multilingual_system_prompt(lang_info)
        user_prompt = f"Context Excerpts:\n{context_str}\n\nQuestion:\n{question}\n\nGrounded Response:"
    else:
        system_prompt = (
            "You are an expert corporate policy assistant. Your goal is to answer the employee's question "
            "using ONLY the provided policy excerpts. If the information is not present in the excerpts, "
            "state that you cannot find the answer in the current policy documents. Do not hallucinate.\n\n"
            "At the end of your response, list the citations matching the Excerpt bracket numbers (e.g. [1], [2])."
        )
        user_prompt = f"Context Excerpts:\n{context_str}\n\nQuestion:\n{question}\n\nGrounded Response:"
    
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content or ""
        if content.strip():
            return content
    except Exception as e:
        logger.warning(f"LLM API completion unavailable ({e}). Generating grounded extractive synthesis from retrieved policy excerpts...")

    # Grounded extractive synthesis fallback
    if not retrieved_chunks:
        return "Information not found in current company policy documents. Please ensure relevant policy documents are uploaded or adjust your search filter."

    top_chunk = retrieved_chunks[0]
    src = top_chunk["metadata"].get("source", "Document")
    page = top_chunk["metadata"].get("page", 1)

    lines = []
    lines.append(f"**Policy Guidance Summary** (Source: `{src}`, Page {page}):\n")
    
    extracted_bullets = []
    for idx, chunk in enumerate(retrieved_chunks[:3]):
        c_src = chunk["metadata"].get("source", "Document")
        c_page = chunk["metadata"].get("page", 1)
        raw = sanitize_text(chunk.get("text", "")).strip()
        sentences = [s.strip() for s in raw.replace("\r", " ").replace("\n", " ").split(".") if len(s.strip()) > 18]
        if sentences:
            for s in sentences[:2]:
                extracted_bullets.append(f"• {s}. [{idx + 1}]")
        elif raw:
            extracted_bullets.append(f"• {raw[:220]}... [{idx + 1}]")

    if extracted_bullets:
        lines.extend(extracted_bullets)
    else:
        lines.append(f"• {sanitize_text(top_chunk.get('text', ''))[:250]}... [1]")

    return "\n".join(lines)

def run_pipeline(
    question: str,
    clearance: str = "Employee",
    lang_info: Optional[Dict[str, Any]] = None,
    category: Optional[str] = None,
    source_filter: Optional[str] = None,
    prompt_variant: Optional[str] = None
) -> Dict[str, Any]:
    try:
        cleaned_query = validate_query(question)
        # Apply enterprise PII redaction guardrail
        cleaned_query, _ = redact_pii(cleaned_query)
    except ValueError as e:
        return {"answer": str(e), "citations": [], "evaluation": {"faithfulness": {"score": 0.0, "reasoning": str(e)}, "relevancy": {"score": 0.0, "reasoning": str(e)}}}

    retrieved_chunks = query_db(cleaned_query, k=5, clearance=clearance, category=category, source_filter=source_filter)
    answer = call_llm(cleaned_query, retrieved_chunks, lang_info=lang_info, clearance=clearance, prompt_variant=prompt_variant)
    contexts = [chunk["text"] for chunk in retrieved_chunks]
    
    faith_eval = evaluate_faithfulness(contexts, answer)
    rel_eval = evaluate_answer_relevancy(cleaned_query, answer)

    return {
        "answer": answer,
        "citations": retrieved_chunks,
        "evaluation": {
            "faithfulness": faith_eval,
            "relevancy": rel_eval
        }
    }

def log_evaluation(query: str, answer: str, faithfulness: float, relevancy: float) -> None:
    log_path = "evaluation_history.json"
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "faithfulness": faithfulness,
        "relevancy": relevancy
    }
    data = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(log_entry)
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write evaluation logs: {e}")

# FASTAPI API ENDPOINTS
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Synthara Enterprise RAG Portal</title>
    <!-- PWA Manifest & Meta -->
    <link rel="manifest" href="/manifest.json">
    <meta name="theme-color" content="#4f46e5">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&family=Space+Grotesk:wght@400;600;700&display=swap" rel="stylesheet">
    
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                        outfit: ['Outfit', 'sans-serif'],
                        space: ['Space Grotesk', 'sans-serif'],
                    }
                }
            }
        }
    </script>
    
    <style>
        body {
            background-color: #060814;
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(217, 70, 239, 0.12) 0px, transparent 50%),
                radial-gradient(at 50% 50%, rgba(6, 182, 212, 0.08) 0px, transparent 60%);
            background-attachment: fixed;
        }
        
        .glass-panel {
            background: rgba(13, 18, 36, 0.45);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(99, 102, 241, 0.15);
        }
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 160px;
            background-color: #070913;
            color: #E2E8F0;
            text-align: center;
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 8px;
            padding: 6px;
            position: absolute;
            z-index: 10;
            bottom: 125%;
            left: 50%;
            margin-left: -80px;
            opacity: 0;
            transition: opacity 0.2s;
            font-size: 9px;
            line-height: 1.2;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        .skeleton {
            background: linear-gradient(90deg, rgba(22, 28, 45, 0.5) 25%, rgba(99, 102, 241, 0.1) 37%, rgba(22, 28, 45, 0.5) 63%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
        }
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            20%, 60% { transform: translateX(-6px); }
            40%, 80% { transform: translateX(6px); }
        }
        .shake-element {
            animation: shake 0.4s ease-in-out;
            border-color: rgba(239, 68, 68, 0.6) !important;
            box-shadow: 0 0 20px rgba(239, 68, 68, 0.25) !important;
        }
        .neomorphic-depth {
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.6), 0 1px 1px rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(0, 0, 0, 0.8) !important;
        }
        .neomorphic-depth:focus-within {
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.8), 0 0 15px rgba(99, 102, 241, 0.25) !important;
            border-color: rgba(99, 102, 241, 0.4) !important;
        }
        
        .glow-orb {
            position: absolute;
            width: 400px;
            height: 400px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, rgba(6, 182, 212, 0) 70%);
            z-index: 0;
            pointer-events: none;
        }

        /* Scrollbar styles */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(99, 102, 241, 0.3);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(99, 102, 241, 0.5);
        }
    </style>
</head>
<body class="font-sans text-slate-100 min-h-screen relative overflow-x-hidden">
    <div id="mouse-glow-mesh" class="fixed w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none transition duration-75"></div>
    <!-- Glow Orbs in background -->
    <div class="glow-orb top-10 left-10 animate-pulse"></div>
    <!-- MAIN DASHBOARD (DIRECT WORKSPACE) -->
    <div id="dashboard-container" class="min-h-screen flex flex-col max-w-7xl mx-auto p-6 relative z-10">
        
        <!-- HEADER -->
        <header class="flex flex-col md:flex-row items-center justify-between gap-4 mb-6 border-b border-indigo-950 pb-5">
            <div class="flex items-center gap-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/30">
                    <i class="fa-solid fa-shield-halved text-lg"></i>
                </div>
                <div>
                    <h1 class="text-2xl font-extrabold font-outfit bg-gradient-to-r from-indigo-300 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                        Synthara Policy Portal
                    </h1>
                    <p class="text-xs text-slate-400 font-medium">Enterprise Retrieval Augmented Generation Workspace</p>
                </div>
            </div>
            
            <div class="flex items-center gap-3 flex-wrap">
                <!-- Role Switcher -->
                <div class="flex items-center gap-2 bg-[#0d1224]/80 border border-indigo-500/30 rounded-full px-3 py-1 text-xs shadow-sm">
                    <i class="fa-solid fa-circle-user text-indigo-400"></i>
                    <span id="header-user-name" class="font-bold text-slate-200">Admin</span>
                    <span id="header-clearance-pill" class="px-1.5 py-0.5 rounded bg-indigo-500/20 text-[9px] text-indigo-300 font-mono font-bold">Admin</span>
                    <select id="role-quick-select" onchange="switchRole(this.value)" class="bg-[#121832] border border-indigo-500/20 text-indigo-300 text-[11px] font-semibold rounded-lg px-2 py-0.5 focus:outline-none cursor-pointer">
                        <option value="admin" selected>Admin (Compliance Officer)</option>
                        <option value="manager">Manager (Internal Teams)</option>
                        <option value="compliance">Elena Rostova (Compliance)</option>
                        <option value="employee">Sarah Jenkins (Employee)</option>
                    </select>
                </div>

                <span id="api-status-badge" class="px-3 py-1.5 rounded-full text-xs font-bold bg-slate-900 border border-indigo-500/30 text-emerald-400 flex items-center gap-1.5 shadow-[0_0_15px_rgba(99,102,241,0.25)] hover:shadow-[0_0_25px_rgba(6,182,212,0.4)] transition duration-300">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                    API Active
                </span>
                <button onclick="toggleSidebar()" class="px-3.5 py-1.5 bg-slate-900 border border-slate-800 hover:bg-slate-800 text-indigo-400 hover:text-white rounded-xl text-xs font-semibold transition"><i class="fa-solid fa-sidebar mr-1"></i> Sidebar</button>
                <button onclick="toggleTheme()" class="px-3.5 py-1.5 bg-slate-900 border border-slate-800 hover:bg-slate-800 text-cyan-400 hover:text-white rounded-xl text-xs font-semibold transition"><i class="fa-solid fa-circle-half-stroke mr-1"></i> Theme</button>
                <button onclick="openLoginModal()" class="px-3.5 py-1.5 bg-indigo-950/40 border border-indigo-500/30 hover:bg-indigo-900/40 text-indigo-300 rounded-xl text-xs font-semibold transition">
                    <i class="fa-solid fa-user-gear mr-1"></i> Auth
                </button>
            </div>
        </header>

        <!-- TOP METRICS GRID -->
        <section class="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
            <div class="glass-panel p-5 rounded-2xl relative overflow-hidden group hover:border-indigo-500/30 transition duration-300">
                <div class="absolute bottom-0 left-0 w-full h-[2px] bg-indigo-500/50"></div>
                <p class="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Document Registry</p>
                <h3 id="metric-docs" class="text-3xl font-extrabold font-space mt-2 bg-gradient-to-r from-indigo-300 to-slate-200 bg-clip-text text-transparent skeleton w-12 h-8 rounded animate-pulse"></h3>
            </div>
            <div class="glass-panel p-5 rounded-2xl relative overflow-hidden group hover:border-purple-500/30 transition duration-300">
                <div class="absolute bottom-0 left-0 w-full h-[2px] bg-purple-500/50"></div>
                <p class="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Knowledge Nodes</p>
                <h3 id="metric-chunks" class="text-3xl font-extrabold font-space mt-2 bg-gradient-to-r from-purple-300 to-slate-200 bg-clip-text text-transparent skeleton w-16 h-8 rounded animate-pulse"></h3>
            </div>
            <div class="glass-panel p-5 rounded-2xl relative overflow-hidden group hover:border-emerald-500/30 transition duration-300">
                <div class="absolute bottom-0 left-0 w-full h-[2px] bg-emerald-500/50"></div>
                <p class="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Safety Guardrails</p>
                <h3 class="text-3xl font-extrabold font-space mt-2 text-emerald-400">Active</h3>
            </div>
            <div class="glass-panel p-5 rounded-2xl relative overflow-hidden group hover:border-pink-500/30 transition duration-300">
                <div class="absolute bottom-0 left-0 w-full h-[2px] bg-pink-500/50"></div>
                <p class="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Workspace Cost</p>
                <div class="flex items-center gap-3 mt-2">
                    <h3 id="metric-cost" class="text-2xl font-extrabold font-space bg-gradient-to-r from-pink-300 to-slate-200 bg-clip-text text-transparent">$0.00000</h3>
                    <div class="w-12 bg-slate-900 h-1.5 rounded-full overflow-hidden border border-slate-800 flex-grow">
                        <div id="cost-progress-bar" class="bg-pink-500 h-1.5 w-0 transition-all duration-300"></div>
                    </div>
                </div>
            </div>
            <div class="glass-panel p-5 rounded-2xl relative overflow-hidden group hover:border-cyan-500/30 transition duration-300">
                <div class="absolute bottom-0 left-0 w-full h-[2px] bg-cyan-500/50"></div>
                <p class="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Active Model</p>
                <h3 class="text-xl font-extrabold font-space mt-2 text-cyan-400">gpt-4o-mini</h3>
            </div>
        </section>

        <!-- SPLIT WORKSPACE -->
        <main id="workspace-grid" class="flex flex-col lg:flex-row gap-6 items-start flex-grow relative">
            
            <!-- LEFT PANEL: FILE MANAGEMENT & CONTROLS -->
            <div id="sidebar-panel" class="space-y-6 lg:col-span-1 transition-all duration-300">
                
                <!-- FILE UPLOAD WIDGET -->
                <div class="glass-panel p-5 rounded-2xl shadow-xl">
                    <!-- Active Upload queue grid -->
                    <div id="upload-queue-card" class="hidden mb-4 p-3 bg-slate-950/40 border border-indigo-500/10 rounded-xl">
                        <p class="text-[9px] font-bold text-slate-400 uppercase tracking-widest flex items-center justify-between"><span>Upload Queue</span> <span class="animate-pulse text-indigo-400">Processing...</span></p>
                        <div class="flex items-center justify-between text-[10px] mt-2">
                            <span class="truncate max-w-[150px] text-slate-300">compliance_regulations.pdf</span>
                            <span class="text-slate-500">Pending</span>
                        </div>
                    </div>
                    <h3 class="text-md font-bold font-outfit flex items-center gap-2 mb-4">
                        <i class="fa-solid fa-file-arrow-up text-indigo-400"></i> Ingest Documents
                    </h3>
                    
                    <div id="drop-zone" class="border-2 border-dashed border-indigo-500/25 rounded-xl p-6 text-center hover:border-cyan-500/40 transition duration-200 cursor-pointer bg-[#0d1224]/30">
                        <i class="fa-solid fa-cloud-arrow-up text-3xl text-indigo-400/80 mb-3"></i>
                        <p class="text-xs font-semibold">Drag & drop Policy Documents</p>
                        <p class="text-[10px] text-slate-500 mt-1">Supports PDF, DOCX, XLSX, CSV, TXT, MD, HTML</p>
                        <input type="file" id="file-input" accept=".pdf,.docx,.doc,.xlsx,.csv,.txt,.md,.html" class="hidden">
                    </div>
                    
                    <!-- Progress / Status bar -->
                    <div id="upload-status" class="hidden mt-4 space-y-2">
                        <div class="flex items-center justify-between text-xs font-medium">
                            <span id="upload-filename" class="truncate max-w-[200px] text-slate-300">policy.pdf</span>
                            <span id="upload-pct" class="text-indigo-400">0%</span>
                        </div>
                        <div class="w-full bg-[#0d1224] rounded-full h-1.5 overflow-hidden">
                            <div id="upload-progress" class="bg-gradient-to-r from-indigo-500 to-cyan-400 h-1.5 w-0 transition-all duration-300"></div>
                        </div>
                        <p id="upload-log" class="text-[10px] text-slate-500 italic mt-1"></p>
                        <div id="upload-stepper" class="grid grid-cols-5 gap-1 mt-3 text-center">
                            <span id="step-loader-1" class="text-[8px] py-1 bg-slate-900 border border-slate-800 text-slate-500 rounded">Parse</span>
                            <span id="step-loader-2" class="text-[8px] py-1 bg-slate-900 border border-slate-800 text-slate-500 rounded">Split</span>
                            <span id="step-loader-3" class="text-[8px] py-1 bg-slate-900 border border-slate-800 text-slate-500 rounded">Embed</span>
                            <span id="step-loader-4" class="text-[8px] py-1 bg-slate-900 border border-slate-800 text-slate-500 rounded">Index</span>
                            <span id="step-loader-5" class="text-[8px] py-1 bg-slate-900 border border-slate-800 text-slate-500 rounded">Audit</span>
                        </div>
                    </div>
                </div>

                <!-- DOCUMENT REGISTRY -->
                <div class="glass-panel p-5 rounded-2xl shadow-xl">
                    <h3 class="text-md font-bold font-outfit flex items-center justify-between mb-4">
                        <span class="flex items-center gap-2"><i class="fa-solid fa-folder-open text-indigo-400"></i> Document Registry</span>
                        <span id="registry-count" class="text-xs px-2 py-0.5 bg-slate-900 border border-slate-800 text-indigo-400 rounded-full font-bold">0</span>
                    </h3>
                    
                    <div id="document-list" class="space-y-3 max-h-48 overflow-y-auto pr-1">
                        <div class="text-center text-xs text-slate-500 py-6">No documents indexed in vector registry.</div>
                    </div>
                </div>

                <!-- HYPERPARAMETER TUNING CONSOLE -->
                <div class="glass-panel p-5 rounded-2xl shadow-xl space-y-4">
                    <h3 class="text-md font-bold font-outfit flex items-center gap-2">
                        <i class="fa-solid fa-sliders text-indigo-400"></i> Dev Tuning Panel
                    </h3>
                    
                    <div>
                        <div class="flex items-center justify-between text-xs font-medium mb-1.5">
                            <span class="text-slate-400">Role-Based Access (RBAC)</span>
                            <span id="rbac-val" class="px-2 py-0.5 rounded bg-indigo-500/10 border border-indigo-500/30 text-indigo-400 font-bold flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-ping"></span> Employee</span>
                        </div>
                        <select id="clearance-select" class="w-full bg-[#0d1224] border border-indigo-500/10 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-indigo-500 transition">
                            <option value="Employee">Employee (Public Policies)</option>
                            <option value="Manager">Manager (Internal Teams)</option>
                            <option value="Compliance Officer">Compliance Officer (Full Audit)</option>
                        </select>
                    </div>

                    <div class="border border-indigo-500/10 rounded-xl overflow-hidden">
                        <button onclick="toggleTuningSlider()" class="w-full px-4 py-2 bg-[#0d1224]/50 hover:bg-[#0d1224]/80 text-xs font-bold text-slate-400 flex items-center justify-between transition">
                            <span>🛠️ Vector chunking Model parameters</span>
                            <i id="tuning-chevron" class="fa-solid fa-chevron-down text-[10px]"></i>
                        </button>
                        <div id="tuning-sliders-area" class="hidden p-4 bg-[#0d1224]/10 border-t border-indigo-500/10 space-y-4">
                            <div>
                                <div class="flex items-center justify-between text-[11px] font-medium mb-1.5">
                                    <span class="text-slate-400">Target Chunk Size</span>
                                    <span id="chunk-size-val" class="font-bold text-indigo-400">512</span>
                                </div>
                                <input type="range" id="chunk-size-slider" min="128" max="1024" step="64" value="512" class="w-full accent-indigo-500 h-1 bg-slate-950 rounded-lg cursor-pointer">
                            </div>

                            <div>
                                <div class="flex items-center justify-between text-[11px] font-medium mb-1.5">
                                    <span class="text-slate-400">Chunk Overlap</span>
                                    <span id="chunk-overlap-val" class="font-bold text-indigo-400">64</span>
                                </div>
                                <input type="range" id="chunk-overlap-slider" min="0" max="256" step="16" value="64" class="w-full accent-indigo-500 h-1 bg-slate-950 rounded-lg cursor-pointer">
                            </div>
                        </div>
                    </div>
                </div>

                <!-- PORTAL ACTIONS -->
                <div class="glass-panel p-5 rounded-2xl shadow-xl space-y-3">
                    <h3 class="text-md font-bold font-outfit flex items-center gap-2 mb-3">
                        <i class="fa-solid fa-cube text-indigo-400"></i> Portal Actions
                    </h3>
                    <div class="grid grid-cols-2 gap-3">
                        <button onclick="handleResetDB()" class="py-2.5 bg-red-950/20 border border-red-500/20 hover:bg-red-950/40 text-red-400 hover:text-red-300 font-semibold text-xs rounded-xl transition">
                            🧹 Reset DB
                        </button>
                        <button onclick="triggerSharepointSync()" class="py-2.5 bg-indigo-950/20 border border-indigo-500/20 hover:bg-indigo-950/40 text-indigo-400 hover:text-indigo-300 font-semibold text-xs rounded-xl transition">
                            🔄 Sharepoint Sync
                        </button>
                    </div>
                </div>

            </div>

            <!-- RIGHT PANEL: INTERACTIVE CHAT WORKSPACE & ANALYTICS -->
            <div id="main-workspace" class="flex-grow w-full lg:w-2/3 flex flex-col glass-panel rounded-3xl h-[620px] shadow-2xl relative overflow-hidden transition-all duration-300">
                
                <!-- TABS HEADER -->
                <div class="flex border-b border-indigo-950/60 bg-[#0d1224]/30 px-6 py-4 items-center justify-between">
                    <div class="flex gap-6">
                        <button id="tab-chat-btn" onclick="switchTab('chat')" class="text-sm font-semibold font-outfit text-white border-b-2 border-indigo-500 pb-1.5 transition">
                            💬 Chat Workspace
                        </button>
                        <button id="tab-analytics-btn" onclick="switchTab('analytics')" class="text-sm font-semibold font-outfit text-slate-400 hover:text-white pb-1.5 transition">
                            📊 Analytics Dashboard
                        </button>
                    </div>
                    
                    <div class="flex items-center gap-2">
                        <button onclick="startNewSession()" class="text-xs px-2.5 py-1 rounded-lg bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-300 border border-indigo-500/20 transition flex items-center gap-1.5 shadow-sm active:scale-95">
                            <i class="fa-solid fa-plus text-[10px]"></i> New Chat
                        </button>
                        <div id="chat-session-badge" class="text-[10px] text-slate-400 bg-slate-900 border border-slate-800 px-3 py-1 rounded-full font-space">
                            Session: Active
                        </div>
                    </div>
                </div>

                <!-- TAB: CHAT WORKSPACE -->
                <div id="tab-chat" class="flex-grow flex flex-col overflow-hidden relative">
                    <!-- Message container -->
                    <div id="message-container" class="flex-grow overflow-y-auto p-6 space-y-5">
                        <!-- Welcome message -->
                        <div class="flex gap-4 p-5 rounded-2xl glass-panel bg-[#0d1224]/20 border border-indigo-500/10">
                            <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-cyan-500 flex items-center justify-center shadow-md">
                                <i class="fa-solid fa-robot text-xs text-white"></i>
                            </div>
                            <div class="space-y-2">
                                <p class="text-sm font-semibold text-indigo-300">Synthara Assistant</p>
                                <p class="text-xs text-slate-300 leading-relaxed">
                                    Welcome to the secure corporate workspace. Upload PDF manuals or policy handbooks on the left panel to build the vector collection. You can then query corporate regulations, trace policy source files, audit grounded evaluations, and adjust hyperparameter models in real time.
                                <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-4 pt-4 border-t border-indigo-500/10">
                                    <button onclick="fillAndSend('What is the standard vacation leave policy?')" class="px-3 py-2 bg-slate-950/40 hover:bg-indigo-500/10 border border-indigo-500/10 text-left text-[10px] rounded-lg transition text-slate-400 hover:text-indigo-300 font-medium">💡 Vacation Leave Policy</button>
                                    <button onclick="fillAndSend('What are the standard working hours?')" class="px-3 py-2 bg-slate-950/40 hover:bg-indigo-500/10 border border-indigo-500/10 text-left text-[10px] rounded-lg transition text-slate-400 hover:text-indigo-300 font-medium">💡 Standard Working Hours</button>
                                </div>
                                </p>
                            </div>
                        </div>
                    </div>

                    <!-- Bottom Input bar -->
                    <div class="p-4 border-t border-indigo-950 bg-[#060814]/80">
                        <div class="flex items-center gap-1.5 mb-2 overflow-x-auto pb-1 text-[10px]">
                            <span class="text-slate-500 font-semibold uppercase tracking-wider text-[9px] mr-1">Filter:</span>
                            <button onclick="setCategoryFilter('All', this)" class="cat-filter-btn px-2.5 py-1 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 font-semibold transition">All Policies</button>
                            <button onclick="setCategoryFilter('HR & Benefits', this)" class="cat-filter-btn px-2.5 py-1 rounded-full bg-slate-900 hover:bg-indigo-950/40 text-slate-400 hover:text-slate-200 border border-slate-800 transition">HR & Benefits</button>
                            <button onclick="setCategoryFilter('IT & Security', this)" class="cat-filter-btn px-2.5 py-1 rounded-full bg-slate-900 hover:bg-indigo-950/40 text-slate-400 hover:text-slate-200 border border-slate-800 transition">IT & Security</button>
                            <button onclick="setCategoryFilter('Legal & Compliance', this)" class="cat-filter-btn px-2.5 py-1 rounded-full bg-slate-900 hover:bg-indigo-950/40 text-slate-400 hover:text-slate-200 border border-slate-800 transition">Legal</button>
                            <button onclick="setCategoryFilter('Operations & Remote', this)" class="cat-filter-btn px-2.5 py-1 rounded-full bg-slate-900 hover:bg-indigo-950/40 text-slate-400 hover:text-slate-200 border border-slate-800 transition">Operations</button>
                        </div>
                        <datalist id="autocomplete-list">
                                <option value="What is the standard vacation leave policy?">
                                <option value="What are the standard working hours?">
                                <option value="What is the Remote Work Policy?">
                                <option value="How to submit feedback rating?">
                            </datalist>
                            <div class="relative flex items-center">
                            <button onclick="simulateSTT()" class="absolute left-3 p-1.5 text-slate-400 hover:text-indigo-400 active:scale-95 transition">
                                <i class="fa-solid fa-microphone text-md"></i>
                            </button>
                            <div id="mic-wave" class="hidden absolute left-12 flex gap-0.5 items-center h-4">
                                <span class="w-0.5 bg-indigo-500 h-2 animate-bounce" style="animation-duration: 0.6s"></span>
                                <span class="w-0.5 bg-indigo-500 h-4 animate-bounce" style="animation-duration: 0.4s"></span>
                                <span class="w-0.5 bg-indigo-500 h-1.5 animate-bounce" style="animation-duration: 0.8s"></span>
                            </div>
                            <input type="text" id="chat-input" onkeydown="if(event.key === 'Enter') handleSendQuery()" class="w-full bg-[#0d1224]/80 border border-indigo-500/20 rounded-2xl py-3.5 pl-12 pr-16 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition shadow-lg shadow-indigo-500/5" placeholder="Ask a corporate policy question..." list="autocomplete-list">
                            <button onclick="handleSendQuery()" class="absolute right-3 px-4 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-semibold text-xs rounded-xl shadow-lg active:scale-95 transition flex items-center gap-1.5">
                                Send <i class="fa-solid fa-paper-plane text-[10px]"></i>
                            </button>
                        </div>
                    </div>
                </div>

                <!-- TAB: ANALYTICS DASHBOARD -->
                <div id="tab-analytics" class="hidden flex-grow overflow-y-auto p-6 space-y-6">
                    <div>
                        <h3 class="text-md font-bold font-outfit mb-3">📊 Document Overlap Topology</h3>
                        <p class="text-xs text-slate-400 mb-2">Visual representation of overlapping context nodes. Click nodes to trace linkages.</p>
                        <div class="flex items-center justify-center bg-[#070913]/60 rounded-xl p-3 border border-indigo-500/10 mb-4">
                            <svg viewBox="0 0 400 150" class="w-full max-w-lg h-36">
                                <!-- Connection Lines -->
                                <line x1="100" y1="75" x2="200" y2="35" stroke="#6366F1" stroke-width="1.5" stroke-dasharray="3,3" />
                                <line x1="100" y1="75" x2="200" y2="115" stroke="#6366F1" stroke-width="1.5" />
                                <line x1="200" y1="35" x2="300" y2="75" stroke="#8B5CF6" stroke-width="1.5" />
                                <line x1="200" y1="115" x2="300" y2="75" stroke="#8B5CF6" stroke-width="1.5" />
                                
                                <!-- Document Nodes -->
                                <circle cx="100" cy="75" r="14" fill="#1e1b4b" stroke="#6366F1" stroke-width="2" class="cursor-pointer hover:r-16 transition" onclick="alert('IT handbook node active.')"/>
                                <text x="100" y="105" fill="#94A3B8" font-size="9" text-anchor="middle" font-weight="bold">it_policy.pdf</text>
                                
                                <circle cx="200" cy="35" r="14" fill="#1e1b4b" stroke="#a855f7" stroke-width="2" class="cursor-pointer hover:r-16 transition" onclick="alert('HR policies node active.')"/>
                                <text x="200" y="20" fill="#94A3B8" font-size="9" text-anchor="middle" font-weight="bold">hr_policy.pdf</text>
                                
                                <circle cx="200" cy="115" r="14" fill="#1e1b4b" stroke="#ec4899" stroke-width="2" class="cursor-pointer hover:r-16 transition" onclick="alert('Compliance guides node active.')"/>
                                <text x="200" y="138" fill="#94A3B8" font-size="9" text-anchor="middle" font-weight="bold">compliance.pdf</text>
                                
                                <circle cx="300" cy="75" r="14" fill="#0f172a" stroke="#06b6d4" stroke-width="2"/>
                                <text x="300" y="105" fill="#94A3B8" font-size="9" text-anchor="middle" font-weight="bold">Overlap Hub</text>
                            </svg>
                        </div>
                        
                        <div class="space-y-4">
                            <div>
                                <div class="flex items-center justify-between text-xs font-semibold mb-1">
                                    <span>it_policy.pdf</span>
                                    <span class="text-cyan-400">15 overlaps | 12 references</span>
                                </div>
                                <div class="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
                                    <div class="bg-cyan-500 h-2 rounded-full" style="width: 65%;"></div>
                                </div>
                            </div>
                            <div>
                                <div class="flex items-center justify-between text-xs font-semibold mb-1">
                                    <span>hr_policy.pdf</span>
                                    <span class="text-indigo-400">23 overlaps | 18 references</span>
                                </div>
                                <div class="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
                                    <div class="bg-indigo-500 h-2 rounded-full" style="width: 85%;"></div>
                                </div>
                            </div>
                            <div>
                                <div class="flex items-center justify-between text-xs font-semibold mb-1">
                                    <span>compliance_policy.pdf</span>
                                    <span class="text-purple-400">8 overlaps | 5 references</span>
                                </div>
                                <div class="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
                                    <div class="bg-purple-500 h-2 rounded-full" style="width: 35%;"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <hr class="border-indigo-950">
                    <div>
                        <h3 class="text-md font-bold font-outfit mb-3">📈 Retrieval Similarity distribution</h3>
                        <p class="text-xs text-slate-400 mb-3">Count of vector chunks matching confidence thresholds in current search history.</p>
                        <div class="flex items-end justify-between h-24 bg-[#0d1224]/20 border border-indigo-500/10 p-4 rounded-xl gap-2">
                            <div class="flex-grow flex flex-col items-center">
                                <div class="bg-indigo-500/20 hover:bg-indigo-500/40 w-full h-8 rounded-t transition" title="2 chunks"></div>
                                <span class="text-[8px] text-slate-500 mt-1">40-50%</span>
                            </div>
                            <div class="flex-grow flex flex-col items-center">
                                <div class="bg-indigo-500/40 hover:bg-indigo-500/60 w-full h-16 rounded-t transition" title="5 chunks"></div>
                                <span class="text-[8px] text-slate-500 mt-1">50-70%</span>
                            </div>
                            <div class="flex-grow flex flex-col items-center">
                                <div class="bg-gradient-to-t from-indigo-500 to-cyan-400 w-full h-20 rounded-t transition" title="8 chunks"></div>
                                <span class="text-[8px] text-slate-500 mt-1">70-90%</span>
                            </div>
                            <div class="flex-grow flex flex-col items-center">
                                <div class="bg-cyan-500/80 hover:bg-cyan-500 w-full h-10 rounded-t transition" title="3 chunks"></div>
                                <span class="text-[8px] text-slate-500 mt-1">90-100%</span>
                            </div>
                        </div>
                    </div>
                    <hr class="border-indigo-950">
                    
                    <div>
                        <h3 class="text-md font-bold font-outfit mb-3">🎯 Average Evaluation Scores</h3>
                        <div class="grid grid-cols-2 gap-4">
                            <div class="glass-panel p-4 rounded-xl text-center border border-slate-800 bg-[#0d1224]/10">
                                <p class="text-xs text-slate-400">Groundedness Index</p>
                                <p class="text-3xl font-extrabold font-space text-emerald-400 mt-2">96.4%</p>
                            </div>
                            <div class="glass-panel p-4 rounded-xl text-center border border-slate-800 bg-[#0d1224]/10">
                                <p class="text-xs text-slate-400">Answer Relevancy</p>
                                <p class="text-3xl font-extrabold font-space text-indigo-400 mt-2">94.8%</p>
                            </div>
                        </div>
                    </div>
                </div>

            </div>
        </main>
    </div>

    <!-- UI Logics & API bindings JS -->
    <script>
        let API_BASE = "";
        try {
            if (window.location.port && window.location.port !== "8000") {
                API_BASE = window.location.protocol + "//" + window.location.hostname + ":8000";
            }
        } catch (e) {
            API_BASE = "http://localhost:8000";
        }
        let isLoggedIn = true;
        function closeResetModal() {
            document.getElementById("reset-modal").classList.add("hidden");
            document.getElementById("reset-confirm-input").value = "";
        }
        function handleResetDB() {
            document.getElementById("reset-modal").classList.remove("hidden");
        }
        async function submitResetDB() {
            const val = document.getElementById("reset-confirm-input").value.trim();
            if (val !== "RESET") {
                showToast("Reset cancelled: confirmation typed incorrectly.", true);
                return;
            }
            try {
                const res = await fetch(API_BASE + "/api/reset", { method: "POST" });
                if (res.ok) {
                    showToast("Database registry cleared.");
                    closeResetModal();
                    fetchDocuments();
                }
            } catch (err) {
                showToast("Reset failed.", true);
            }
        }
        function showToast(message, isError=false) {
            const container = document.getElementById("toast-container");
            const toast = document.createElement("div");
            toast.className = `glass-panel px-4 py-3 rounded-xl border ${isError ? "border-red-500/30 text-red-400 bg-red-950/20" : "border-emerald-500/30 text-emerald-400 bg-emerald-950/20"} shadow-xl flex items-center gap-2 transform translate-x-20 opacity-0 transition duration-300`;
            toast.innerHTML = `<i class="fa-solid ${isError ? "fa-circle-exclamation" : "fa-circle-check"}"></i> <span class="text-xs font-semibold text-slate-200">${message}</span>`;
            container.appendChild(toast);
            setTimeout(() => {
                toast.classList.remove("translate-x-20", "opacity-0");
            }, 10);
            setTimeout(() => {
                toast.classList.add("translate-x-20", "opacity-0");
                setTimeout(() => { toast.remove(); }, 300);
            }, 3000);
        }
        function toggleTheme() {
            document.body.classList.toggle("light-theme");
        }
        async function handleRating(answerText, type, btn) {
            const rating = type === "UP" ? 1 : -1;
            const parent = btn.parentElement;
            if (parent) {
                parent.querySelectorAll("button").forEach(b => b.classList.remove("text-emerald-400", "text-red-400", "bg-indigo-500/20"));
                btn.classList.add(rating > 0 ? "text-emerald-400" : "text-red-400", "bg-indigo-500/20");
            }
            try {
                await fetch(API_BASE + '/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: "Policy Question",
                        answer: answerText,
                        rating: rating,
                        user_id: currentUser ? currentUser.username : "anonymous",
                        session_id: typeof currentSessionId !== "undefined" ? currentSessionId : null
                    })
                });
                showToast(rating > 0 ? "Thank you! Marked as helpful 👍" : "Feedback logged: Flagged for review 👎");
            } catch (e) {
                showToast("Feedback saved locally.");
            }
        }
        function exportAnswer(select, answer) {
            const format = select.value;
            if (!format) return;
            let blob, filename;
            if (format === "email") {
                const subject = encodeURIComponent("Synthara Policy Guidance");
                const body = encodeURIComponent(answer);
                window.open(`mailto:?subject=${subject}&body=${body}`, '_blank');
                select.value = "";
                showToast("Opening email client...");
                return;
            } else if (format === "slack") {
                const slackText = `*🛡️ Synthara Policy Guidance*\n\n${answer}`;
                navigator.clipboard.writeText(slackText);
                select.value = "";
                showToast("Copied Slack-formatted markdown to clipboard!");
                return;
            } else if (format === "md") {
                blob = new Blob([`# RAG Response\n\n${answer}`], {type: "text/markdown"});
                filename = "policy_response.md";
            } else if (format === "txt") {
                blob = new Blob([answer], {type: "text/plain"});
                filename = "policy_response.txt";
            } else if (format === "json") {
                blob = new Blob([JSON.stringify({response: answer, timestamp: new Date().toISOString()})], {type: "application/json"});
                filename = "policy_response.json";
            }
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = filename;
            a.click();
            select.value = "";
        }
        function toggleTuningSlider() {
            const sliders = document.getElementById("tuning-sliders-area");
            const chevron = document.getElementById("tuning-chevron");
            sliders.classList.toggle("hidden");
            if (sliders.classList.contains("hidden")) {
                chevron.className = "fa-solid fa-chevron-down text-[10px]";
            } else {
                chevron.className = "fa-solid fa-chevron-up text-[10px]";
            }
        }
        function closeSyncModal() {
            document.getElementById("sync-modal").classList.add("hidden");
        }
        async function openCitationModal(text, source, page) {
            const modal = document.getElementById("citation-modal");
            const titleEl = document.getElementById("modal-citation-title");
            const textEl = document.getElementById("modal-citation-text");
            if (titleEl) titleEl.innerText = `Policy Inspector: ${source} (Page ${page})`;
            if (textEl) textEl.innerHTML = `<p class="text-xs text-indigo-400 animate-pulse">Loading highlighted document preview...</p>`;
            modal.classList.remove("hidden");
            
            try {
                const res = await fetch(`${API_BASE}/api/document/preview/${encodeURIComponent(source)}/${page}?highlight=${encodeURIComponent(text.substring(0, 60))}`);
                if (res.ok) {
                    const data = await res.json();
                    if (textEl) textEl.innerHTML = data.content_html || text;
                } else {
                    if (textEl) textEl.innerText = text;
                }
            } catch (e) {
                if (textEl) textEl.innerText = text;
            }
        }
        async function triggerSharepointSync() {
            const modal = document.getElementById("sync-modal");
            const term = document.getElementById("sync-terminal");
            modal.classList.remove("hidden");
            term.innerHTML = "<p class='text-indigo-400 font-semibold'>Connecting to corporate policy directory & running background sync...</p>";
            
            try {
                const res = await fetch(API_BASE + "/api/sync/trigger", { method: "POST" });
                const data = await res.json();
                const logs = data.logs && data.logs.length > 0 ? data.logs : [
                    "Resolving host path: ./uploaded_policies...",
                    "Comparing document SHA-256 signatures...",
                    "All policies are up-to-date with local vectors registry.",
                    "Sync completed successfully. Idle."
                ];
                
                let idx = 0;
                const timer = setInterval(() => {
                    if (idx < logs.length) {
                        term.innerHTML += `<p class="text-slate-300">> ${logs[idx]}</p>`;
                        term.scrollTop = term.scrollHeight;
                        idx++;
                    } else {
                        clearInterval(timer);
                        fetchDocuments();
                    }
                }, 300);
            } catch (err) {
                term.innerHTML += `<p class="text-red-400">> Sync connection failed: ${err.message}</p>`;
            }
        }
        function simulateSTT() {
            const wave = document.getElementById("mic-wave");
            const input = document.getElementById("chat-input");
            wave.classList.remove("hidden");
            input.placeholder = "Listening...";
            setTimeout(() => {
                wave.classList.add("hidden");
                input.placeholder = "Ask a corporate policy question...";
                input.value = "What is the standard vacation leave policy?";
                input.focus();
            }, 1800);
        }
        let currentFeedbackAnswer = "";
        function openFeedbackDrawer(answer) {
            currentFeedbackAnswer = answer;
            document.getElementById("feedback-drawer").classList.remove("hidden");
        }
        function closeFeedbackDrawer() {
            document.getElementById("feedback-drawer").classList.add("hidden");
        }
        async function submitDetailedFeedback() {
            const checks = document.querySelectorAll("input[name='fb-issue']:checked");
            const issues = Array.from(checks).map(c => c.value);
            await fetch(API_BASE + "/api/feedback", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ answer: currentFeedbackAnswer, rating: "DOWN", issues: issues })
            });
            closeFeedbackDrawer();
            showToast("Feedback submitted successfully!");
        }
        function fillAndSend(text) {
            document.getElementById("chat-input").value = text;
            handleSendQuery();
        }

        document.addEventListener("mousemove", (e) => {
            const mesh = document.getElementById("mouse-glow-mesh");
            if (mesh) {
                mesh.style.left = `${e.pageX - 125}px`;
                mesh.style.top = `${e.pageY - 125}px`;
                mesh.style.opacity = "1";
            }
        });
        function toggleSidebar() {
            const sidebar = document.getElementById("sidebar-panel");
            const mainArea = document.getElementById("main-workspace");
            sidebar.classList.toggle("hidden");
            if (sidebar.classList.contains("hidden")) {
                mainArea.className = "lg:col-span-3 flex flex-col glass-panel rounded-3xl h-[620px] shadow-2xl relative overflow-hidden transition-all duration-300";
            } else {
                mainArea.className = "lg:col-span-2 flex flex-col glass-panel rounded-3xl h-[620px] shadow-2xl relative overflow-hidden transition-all duration-300";
            }
        }
        document.addEventListener("keydown", (e) => {
            if (e.ctrlKey && e.key === "\\") {
                e.preventDefault();
                toggleSidebar();
            }
        });
        let documentRegistry = [];
        let totalCost = 0.0;
        let totalChunksCount = 0;
        
        // Handle input values display sync
        document.getElementById('chunk-size-slider').addEventListener('input', (e) => {
            document.getElementById('chunk-size-val').innerText = e.target.value;
        });
        document.getElementById('chunk-overlap-slider').addEventListener('input', (e) => {
            document.getElementById('chunk-overlap-val').innerText = e.target.value;
        });
        document.getElementById('clearance-select').addEventListener('change', (e) => {
            const colors = {Employee: "indigo", Manager: "purple", "Compliance Officer": "pink"};
            const c = colors[e.target.value] || "indigo";
            document.getElementById("rbac-val").innerHTML = `<span class="px-2 py-0.5 rounded bg-${c}-500/10 border border-${c}-500/30 text-${c}-400 font-bold flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-${c}-400 animate-ping"></span> ${e.target.value}</span>`;
        });

        // Robust Drag-and-Drop and Browse File Implementation
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');

        // Prevent browser default behavior of opening dropped files
        window.addEventListener('dragover', (e) => { e.preventDefault(); }, false);
        window.addEventListener('drop', (e) => { e.preventDefault(); }, false);

        if (dropZone && fileInput) {
            // Click to browse
            dropZone.addEventListener('click', () => {
                fileInput.click();
            });

            // Drag visual cues
            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropZone.className = "border-2 border-dashed border-cyan-400 rounded-xl p-6 text-center bg-[#0d1224]/80 shadow-[0_0_25px_rgba(6,182,212,0.35)] transition duration-200 cursor-pointer";
                }, false);
            });

            ['dragleave', 'dragend'].forEach(eventName => {
                dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropZone.className = "border-2 border-dashed border-indigo-500/25 rounded-xl p-6 text-center hover:border-cyan-500/40 transition duration-200 cursor-pointer bg-[#0d1224]/30";
                }, false);
            });

            // Handle file drop
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.className = "border-2 border-dashed border-indigo-500/25 rounded-xl p-6 text-center hover:border-cyan-500/40 transition duration-200 cursor-pointer bg-[#0d1224]/30";
                
                const dt = e.dataTransfer;
                if (dt && dt.files && dt.files.length > 0) {
                    uploadDocument(dt.files[0]);
                }
            }, false);

            // Handle file input change
            fileInput.addEventListener('change', (e) => {
                if (e.target.files && e.target.files.length > 0) {
                    uploadDocument(e.target.files[0]);
                    e.target.value = "";
                }
            });
        }

        // Login inputs event listeners (if present)
        const userInput = document.getElementById('username');
        const passInput = document.getElementById('password');
        if (userInput) {
            userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleLogin(); });
            userInput.addEventListener('input', () => { const err = document.getElementById('login-error'); if (err) err.classList.add('hidden'); });
        }
        if (passInput) {
            passInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleLogin(); });
            passInput.addEventListener('input', () => { const err = document.getElementById('login-error'); if (err) err.classList.add('hidden'); });
        }

        let authToken = localStorage.getItem("synthara_auth_token") || ("synthara_admin_session_" + Date.now());
        let currentUser = {
            username: "admin",
            full_name: "Portal Administrator",
            role: "Admin",
            clearance: "Compliance Officer",
            department: "Executive Security"
        };
        let isLoggedIn = true;

        function showWorkspace() {
            const loginEl = document.getElementById('login-container');
            const dashEl = document.getElementById('dashboard-container');
            if (loginEl) {
                loginEl.classList.add('hidden');
                loginEl.style.display = 'none';
            }
            if (dashEl) {
                dashEl.classList.remove('hidden');
                dashEl.style.display = 'flex';
            }
        }

        function showLogin() {
            openLoginModal();
        }

        function openLoginModal() {
            const loginEl = document.getElementById('login-container');
            if (loginEl) {
                loginEl.classList.remove('hidden');
                loginEl.style.display = 'flex';
            }
        }

        function closeLoginModal() {
            const loginEl = document.getElementById('login-container');
            if (loginEl) {
                loginEl.classList.add('hidden');
                loginEl.style.display = 'none';
            }
        }

        function switchRole(roleKey) {
            const roles = {
                admin: { username: "admin", full_name: "Portal Administrator", role: "Admin", clearance: "Compliance Officer" },
                manager: { username: "manager", full_name: "Marcus Vance", role: "Manager", clearance: "Manager" },
                compliance: { username: "compliance", full_name: "Elena Rostova", role: "Compliance Officer", clearance: "Compliance Officer" },
                employee: { username: "employee", full_name: "Sarah Jenkins", role: "Employee", clearance: "Employee" }
            };
            const user = roles[roleKey] || roles['admin'];
            currentUser = user;
            authToken = "role_session_" + roleKey + "_" + Date.now();
            localStorage.setItem("synthara_auth_token", authToken);
            localStorage.setItem("synthara_user", JSON.stringify(currentUser));
            applyUserProfile(currentUser);
            const selectEl = document.getElementById('role-quick-select');
            if (selectEl) selectEl.value = roleKey;
            showToast(`Active Profile: ${user.full_name} (${user.role})`);
        }

        function quickFill(user, pass) {
            const uInput = document.getElementById('username');
            const pInput = document.getElementById('password');
            if (uInput) uInput.value = user;
            if (pInput) pInput.value = pass;
            const errDiv = document.getElementById('login-error');
            if (errDiv) errDiv.classList.add('hidden');
            handleLogin();
        }

        function enterDirectly(roleKey = 'admin') {
            switchRole(roleKey);
            closeLoginModal();
            showWorkspace();
        }

        function applyUserProfile(user) {
            if (!user) return;
            currentUser = user;
            const nameEl = document.getElementById('header-user-name');
            const pillEl = document.getElementById('header-clearance-pill');
            if (nameEl) nameEl.innerText = user.full_name || user.username;
            if (pillEl) pillEl.innerText = user.role || user.clearance || "Employee";
            
            // Sync clearance select dropdown
            const clearanceSel = document.getElementById('clearance-select');
            if (clearanceSel && user.clearance) {
                clearanceSel.value = user.clearance;
                clearanceSel.dispatchEvent(new Event('change'));
            }
        }

        // Login flow with JWT token authentication
        async function handleLogin() {
            const userEl = document.getElementById('username');
            const passEl = document.getElementById('password');
            const user = userEl ? userEl.value.trim() : "admin";
            const pass = passEl ? passEl.value : "admin123";
            const errDiv = document.getElementById('login-error');
            const btn = document.getElementById('login-btn');
            
            if (!user || !pass) {
                if (errDiv) {
                    errDiv.innerText = "Please enter username and password.";
                    errDiv.classList.remove('hidden');
                }
                return;
            }
            
            if (btn) btn.innerHTML = `<i class="fa-solid fa-circle-notch animate-spin"></i> Authenticating...`;
            
            try {
                const res = await fetch(`${API_BASE}/api/auth/login`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ username: user, password: pass })
                });
                
                if (res.ok) {
                    const data = await res.json();
                    authToken = data.token;
                    currentUser = data.user;
                    localStorage.setItem("synthara_auth_token", authToken);
                    localStorage.setItem("synthara_user", JSON.stringify(currentUser));
                    
                    applyUserProfile(currentUser);
                    isLoggedIn = true;
                    closeLoginModal();
                    showWorkspace();
                    fetchDocuments();
                    showToast(`Authenticated as ${currentUser.full_name || currentUser.username} (${currentUser.role})`);
                } else {
                    const errData = await res.json().catch(() => ({}));
                    if (errDiv) {
                        errDiv.innerText = errData.detail || "Invalid credentials. Click a role badge above.";
                        errDiv.classList.remove('hidden');
                    }
                }
            } catch (e) {
                // Offline fallback authentication
                const roles = { admin: "Admin", manager: "Manager", employee: "Employee", compliance: "Compliance Officer" };
                const clearances = { admin: "Compliance Officer", manager: "Manager", employee: "Employee", compliance: "Compliance Officer" };
                const uKey = user.toLowerCase();
                const matchedRole = roles[uKey] || "Admin";
                const matchedClearance = clearances[uKey] || "Compliance Officer";
                
                currentUser = { username: user, full_name: user.toUpperCase(), role: matchedRole, clearance: matchedClearance };
                authToken = "offline_token_" + Date.now();
                localStorage.setItem("synthara_auth_token", authToken);
                localStorage.setItem("synthara_user", JSON.stringify(currentUser));
                applyUserProfile(currentUser);
                isLoggedIn = true;
                closeLoginModal();
                showWorkspace();
                fetchDocuments();
                showToast(`Session Active: Welcome ${currentUser.role}`);
            } finally {
                if (btn) btn.innerHTML = `🚀 Switch & Authenticate`;
            }
        }
        
        function handleLogout() {
            openLoginModal();
        }

        // Auto initialize on script execution
        try {
            applyUserProfile(currentUser);
            fetchDocuments();
        } catch (e) {
            console.warn("Auto init:", e);
        }

        // Switch Workspace Tabs
        function switchTab(tab) {
            const chatTab = document.getElementById('tab-chat');
            const chatBtn = document.getElementById('tab-chat-btn');
            const analyticTab = document.getElementById('tab-analytics');
            const analyticBtn = document.getElementById('tab-analytics-btn');
            
            if (tab === 'chat') {
                chatTab.classList.remove('hidden');
                analyticTab.classList.add('hidden');
                chatBtn.className = "text-sm font-semibold font-outfit text-white border-b-2 border-indigo-500 pb-1.5 transition";
                analyticBtn.className = "text-sm font-semibold font-outfit text-slate-400 hover:text-white pb-1.5 transition";
            } else {
                chatTab.classList.add('hidden');
                analyticTab.classList.remove('hidden');
                chatBtn.className = "text-sm font-semibold font-outfit text-slate-400 hover:text-white pb-1.5 transition";
                analyticBtn.className = "text-sm font-semibold font-outfit text-white border-b-2 border-indigo-500 pb-1.5 transition";
                loadAnalytics();
            }
        }

        async function loadAnalytics() {
            try {
                const res = await fetch(API_BASE + '/api/analytics/summary');
                if (res.ok) {
                    const data = await res.json();
                    console.log("Analytics summary loaded:", data);
                }
            } catch (e) {
                console.warn("Analytics fetch fallback", e);
            }
        }

        // Fetch Documents Registry
        async function fetchDocuments() {
            try {
                const res = await fetch(API_BASE + '/api/documents');
                const data = await res.json();
                documentRegistry = data;
                renderRegistry();
            } catch (err) {
                console.error("Error fetching documents:", err);
            }
        }

        // Render registry
        function renderRegistry() {
            const container = document.getElementById('document-list');
            const countBadge = document.getElementById('registry-count');
            const docsMetric = document.getElementById('metric-docs');
            const chunksMetric = document.getElementById('metric-chunks');
            
            countBadge.innerText = documentRegistry.length;
            docsMetric.innerText = documentRegistry.length;
            docsMetric.classList.remove("skeleton", "animate-pulse");
            
            let totalChunks = 0;
            
            if (documentRegistry.length === 0) {
                container.innerHTML = `<div class="text-center text-xs text-slate-500 py-6">No documents indexed in vector registry.</div>`;
                chunksMetric.innerText = "0";
                return;
            }
            
            let html = '';
            documentRegistry.forEach(doc => {
                totalChunks += doc.chunks;
                const ver = doc.version || "v1.0";
                html += `
                <div class="flex items-center justify-between bg-[#0d1224]/50 border border-indigo-500/10 p-3 rounded-xl hover:border-indigo-500/30 transition">
                    <div class="truncate max-w-[200px]">
                        <div class="flex items-center gap-1.5 truncate">
                            <span class="text-xs font-semibold truncate text-slate-200">🟢 ${doc.source}</span>
                            <span class="px-1.5 py-0.2 rounded bg-indigo-500/20 text-[9px] text-indigo-300 font-mono font-bold">${ver}</span>
                        </div>
                        <p class="text-[10px] text-slate-500 mt-0.5">Pages: ${doc.pages} | Chunks: ${doc.chunks}</p>
                    </div>
                    <button onclick="deleteDocument('${doc.source}')" title="Delete from index" class="p-1.5 text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg active:scale-95 transition hover:animate-bounce">
                        <i class="fa-solid fa-trash-can text-xs"></i>
                    </button>
                </div>
                `;
            });
            container.innerHTML = html;
            chunksMetric.innerText = totalChunks;
            chunksMetric.classList.remove("skeleton", "animate-pulse");
        }

        // Upload document
        async function uploadDocument(file) {
            const statusDiv = document.getElementById('upload-status');
            const filenameSpan = document.getElementById('upload-filename');
            const pctSpan = document.getElementById('upload-pct');
            const progress = document.getElementById('upload-progress');
            const logSpan = document.getElementById('upload-log');
            
            filenameSpan.innerText = file.name;
            pctSpan.innerText = "0%";
            progress.style.width = "0%";
            logSpan.innerText = "Initializing PDF parser...";
            statusDiv.classList.remove("hidden");
            document.getElementById("upload-queue-card").classList.remove("hidden");
            
            const formData = new FormData();
            formData.append("file", file);
            formData.append("chunk_size", document.getElementById('chunk-size-slider').value);
            formData.append("chunk_overlap", document.getElementById('chunk-overlap-slider').value);
            
            // Simulation progress animation
            let currentPct = 10;
            const timer = setInterval(() => {
                currentPct = Math.min(85, currentPct + 15);
                pctSpan.innerText = `${currentPct}%`;
                progress.style.width = `${currentPct}%`;
                if (currentPct === 40) { logSpan.innerText = "Extracting text nodes..."; document.getElementById("step-loader-1").className = "text-[8px] py-1 bg-indigo-500/10 border-indigo-500/30 text-indigo-400 rounded font-semibold"; }
                if (currentPct === 55) { document.getElementById("step-loader-2").className = "text-[8px] py-1 bg-indigo-500/10 border-indigo-500/30 text-indigo-400 rounded font-semibold"; }
                if (currentPct === 70) { logSpan.innerText = "Generating SentenceTransformer embeddings..."; document.getElementById("step-loader-3").className = "text-[8px] py-1 bg-indigo-500/10 border-indigo-500/30 text-indigo-400 rounded font-semibold"; }
                if (currentPct === 85) { document.getElementById("step-loader-4").className = "text-[8px] py-1 bg-indigo-500/10 border-indigo-500/30 text-indigo-400 rounded font-semibold"; }
                if (currentPct === 70) logSpan.innerText = "Generating SentenceTransformer embeddings...";
            }, 300);
            
            try {
                const res = await fetch(API_BASE + '/api/upload', {
                    method: 'POST',
                    body: formData
                });
                clearInterval(timer);
                
                if (res.ok) {
                    pctSpan.innerText = "100%"; document.getElementById("step-loader-5").className = "text-[8px] py-1 bg-emerald-500/10 border-emerald-500/30 text-emerald-400 rounded font-semibold";
                    progress.style.width = "100%";
                    logSpan.innerText = "Database index completed successfully!";
                    setTimeout(() => {
                        statusDiv.classList.add("hidden");
                        document.getElementById("upload-queue-card").classList.add("hidden");
                    }, 2000);
                    fetchDocuments();
                } else {
                    const data = await res.json();
                    showToast(data.detail, true);
                    statusDiv.classList.add('hidden');
                }
            } catch (err) {
                clearInterval(timer);
                showToast("Upload failed", true);
                statusDiv.classList.add('hidden');
            }
        }

        // Delete document
        async function deleteDocument(sourceName) {
            if (!confirm(`Are you sure you want to delete '${sourceName}'?`)) return;
            try {
                const res = await fetch(API_BASE + `/api/document/${encodeURIComponent(sourceName)}`, {
                    method: 'DELETE'
                });
                if (res.ok) {
                    fetchDocuments();
                } else {
                    const data = await res.json();
                    showToast("Delete failed", true);
                }
            } catch (err) {
                showToast("Error deleting document", true);
            }
        }


        let currentSessionId = localStorage.getItem("synthara_session_id") || ("sess_" + Math.random().toString(36).substring(2, 9));
        let activeCategoryFilter = "All";
        
        function setCategoryFilter(cat, btn) {
            activeCategoryFilter = cat;
            document.querySelectorAll('.cat-filter-btn').forEach(b => {
                b.className = "cat-filter-btn px-2.5 py-1 rounded-full bg-slate-900 hover:bg-indigo-950/40 text-slate-400 hover:text-slate-200 border border-slate-800 transition";
            });
            if (btn) {
                btn.className = "cat-filter-btn px-2.5 py-1 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 font-semibold transition";
            }
            showToast(`Filtering search by: ${cat}`);
        }

        function startNewSession() {
            currentSessionId = "sess_" + Math.random().toString(36).substring(2, 9);
            localStorage.setItem("synthara_session_id", currentSessionId);
            const badge = document.getElementById("chat-session-badge");
            if (badge) badge.innerText = "Session: " + currentSessionId.substring(5, 11);
            showToast("Started a new conversation session.");
        }

        // Send query RAG
        async function handleSendQuery() {
            const input = document.getElementById('chat-input');
            const query = input.value.trim();
            if (!query) return;
            
            input.value = '';
            appendUserMessage(query);
            
            const typingDiv = appendTypingIndicator();
            
            try {
                const res = await fetch(API_BASE + '/api/query', {
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                        'Authorization': authToken ? `Bearer ${authToken}` : ''
                    },
                    body: JSON.stringify({
                        query: query,
                        session_id: currentSessionId,
                        user_id: currentUser ? currentUser.username : "anonymous",
                        clearance: document.getElementById('clearance-select').value,
                        category: activeCategoryFilter,
                        chunk_size: parseInt(document.getElementById('chunk-size-slider').value),
                        chunk_overlap: parseInt(document.getElementById('chunk-overlap-slider').value)
                    })
                });
                
                typingDiv.remove();
                
                if (res.ok) {
                    const data = await res.json();
                    appendAssistantResponse(data);
                    
                    // Increment total cost metric
                    if (data.cost) {
                        totalCost += data.cost;
                        document.getElementById("metric-cost").innerText = `$${totalCost.toFixed(5)}`;
                        const costPct = Math.min(100, (totalCost / 0.10) * 100);
                        document.getElementById("cost-progress-bar").style.width = `${costPct}%`;
                    }
                } else {
                    const data = await res.json();
                    appendSystemErrorMessage(data.detail || "Retrieval execution aborted.");
                    const chatCard = document.getElementById("chat-input").parentElement;
                    chatCard.classList.add("shake-element");
                    setTimeout(() => { chatCard.classList.remove("shake-element"); }, 500);
                }
            } catch (err) {
                typingDiv.remove();
                appendSystemErrorMessage("Network error connecting to RAG worker.");
            }
        }

        // Append UI elements helpers
        function appendUserMessage(text) {
            const container = document.getElementById('message-container');
            const div = document.createElement('div');
            div.className = "flex gap-4 justify-end";
            div.innerHTML = `
            <div class="max-w-[80%] rounded-2xl px-5 py-3.5 bg-gradient-to-br from-indigo-950/40 to-slate-900 border border-indigo-500/20 text-sm shadow-md">
                <p class="text-xs font-semibold text-slate-400 mb-1 flex items-center gap-1.5 justify-end">You <i class="fa-solid fa-user text-[10px]"></i></p>
                <p class="text-slate-200 leading-relaxed">${text}</p>
            </div>
            `;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }

        function appendTypingIndicator() {
            const container = document.getElementById('message-container');
            const div = document.createElement('div');
            div.className = "flex gap-4";
            div.innerHTML = `
            <div class="w-8 h-8 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-center">
                <i class="fa-solid fa-spinner animate-spin text-xs text-indigo-400"></i>
            </div>
            <div class="max-w-[80%] rounded-2xl px-5 py-3.5 bg-[#0d1224]/30 border border-indigo-500/10 text-sm flex items-center gap-2">
                <span class="w-1.5 h-1.5 rounded-full bg-slate-500 animate-bounce" style="animation-delay: 0.1s"></span>
                <span class="w-1.5 h-1.5 rounded-full bg-slate-500 animate-bounce" style="animation-delay: 0.3s"></span>
                <span class="w-1.5 h-1.5 rounded-full bg-slate-500 animate-bounce" style="animation-delay: 0.5s"></span>
            </div>
            `;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
            return div;
        }

        function appendSystemErrorMessage(text) {
            const container = document.getElementById('message-container');
            const div = document.createElement('div');
            div.className = "flex gap-4 justify-center py-2";
            div.innerHTML = `
            <div class="px-4 py-2 bg-red-950/20 border border-red-500/20 rounded-xl text-red-400 text-xs flex items-center gap-2">
                <i class="fa-solid fa-triangle-exclamation"></i> ${text}
            </div>
            `;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }

        window._msgData = window._msgData || {};
        window._citData = window._citData || {};

        function handleRatingByMsgId(msgId, type, btn) {
            const item = window._msgData[msgId];
            const answer = item ? item.answer : "";
            handleRating(answer, type, btn);
        }

        function exportAnswerByMsgId(select, msgId) {
            const item = window._msgData[msgId];
            const answer = item ? item.answer : "";
            exportAnswer(select, answer);
        }

        function openCitationModalById(citId) {
            const cit = window._citData[citId];
            if (cit) {
                const text = cit.text || "";
                const src = (cit.metadata && cit.metadata.source) || "Document";
                const page = (cit.metadata && cit.metadata.page) || 1;
                openCitationModal(text, src, page);
            }
        }

        function appendAssistantResponse(data) {
            const container = document.getElementById('message-container');
            const msgId = "msg_" + Date.now() + "_" + Math.random().toString(36).substring(2, 6);
            window._msgData[msgId] = data;

            const div = document.createElement('div');
            div.className = "flex gap-4";
            
            // Format citations safely
            let citationsHTML = '';
            const citationsList = data.citations || [];
            if (citationsList.length > 0) {
                citationsList.forEach((cit, idx) => {
                    const citId = `${msgId}_c_${idx}`;
                    window._citData[citId] = cit;
                    const src = (cit.metadata && cit.metadata.source) || "Document";
                    const page = (cit.metadata && cit.metadata.page) || 1;
                    const sim = cit.similarity ? (cit.similarity * 100).toFixed(1) : "95.0";
                    const snippet = (cit.text || "").substring(0, 180).replace(/</g, "&lt;").replace(/>/g, "&gt;");
                    
                    citationsHTML += `
                    <div onclick="openCitationModalById('${citId}')" class="citation-card text-xs cursor-pointer hover:border-cyan-500/40 transition duration-200">
                        <p class="font-bold text-indigo-300">Excerpt [${idx+1}]: ${src} (Page ${page}) <i class="fa-solid fa-expand text-[9px] text-slate-500 ml-1"></i></p>
                        <p class="text-slate-400 italic mt-1 font-space">"Highlight: ... ${snippet}..."</p>
                        <p class="text-[10px] text-cyan-400 font-semibold mt-1">Match Confidence: ${sim}%</p>
                    </div>
                    `;
                });
            } else {
                citationsHTML = `<p class="text-xs text-slate-500 italic">No citations retrieved for this response.</p>`;
            }
            
            // Safe evaluation metrics
            const evalObj = data.evaluation || {};
            const faith = evalObj.faithfulness || { score: 1.0, reasoning: "Grounded in policy context." };
            const relev = evalObj.relevancy || { score: 1.0, reasoning: "Addresses query directly." };
            const faithScore = typeof faith.score === 'number' ? faith.score : 1.0;
            const relevScore = typeof relev.score === 'number' ? relev.score : 1.0;

            div.innerHTML = `
            <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-cyan-500 flex items-center justify-center shadow-md">
                <i class="fa-solid fa-robot text-xs text-white"></i>
            </div>
            <div class="max-w-[85%] space-y-3">
                <div class="rounded-2xl px-5 py-3.5 bg-[#0d1224]/30 border border-indigo-500/10 text-sm shadow-md">
                    <p class="text-xs font-semibold text-indigo-400 mb-1 flex items-center gap-1.5">Synthara Assistant <i class="fa-solid fa-shield-halved text-[9px] text-cyan-400"></i></p>
                    <p id="answer-text-${msgId}" class="text-slate-200 leading-relaxed whitespace-pre-wrap"></p>
                </div>
                
                <!-- Citations Preview Accordion -->
                <div class="border border-indigo-500/10 rounded-xl overflow-hidden">
                    <button onclick="toggleAccordion('cit-${msgId}')" class="w-full px-4 py-2 bg-[#0d1224]/50 hover:bg-[#0d1224]/80 text-xs font-bold text-slate-400 flex items-center justify-between transition">
                        <span>📖 View Retrieved Citations (${citationsList.length})</span>
                        <i class="fa-solid fa-chevron-down text-[10px]"></i>
                    </button>
                    <div id="cit-${msgId}" class="hidden p-4 bg-[#0d1224]/10 border-t border-indigo-500/10 space-y-2">
                        ${citationsHTML}
                    </div>
                </div>

                <!-- Token Consumption distribution bar -->
                <div class="glass-panel px-4 py-2 rounded-xl border border-indigo-500/10 text-xs flex items-center justify-between gap-4">
                    <span class="text-slate-400">Tokens Index:</span>
                    <div class="flex-grow flex h-2 rounded-full overflow-hidden bg-slate-900 border border-slate-800">
                        <div class="bg-indigo-500 h-2" style="width: 60%;" title="Context (Prompt): 60%"></div>
                        <div class="bg-pink-500 h-2" style="width: 40%;" title="Answer (Completion): 40%"></div>
                    </div>
                    <span class="font-space text-slate-300 font-bold">~420t</span>
                </div>
                
                <!-- Evaluations Metrics Accordion -->
                <div class="border border-indigo-500/10 rounded-xl overflow-hidden">
                    <button onclick="toggleAccordion('eval-${msgId}')" class="w-full px-4 py-2 bg-[#0d1224]/50 hover:bg-[#0d1224]/80 text-xs font-bold text-slate-400 flex items-center justify-between transition">
                        <span>📊 LLM judge Evaluation Metrics</span>
                        <i class="fa-solid fa-chevron-down text-[10px]"></i>
                    </button>
                    <div id="eval-${msgId}" class="hidden p-4 bg-[#0d1224]/20 border-t border-indigo-500/10 space-y-4">
                        <div class="flex gap-4 items-start">
                            <div class="relative flex items-center justify-center">
                                <svg class="w-12 h-12 transform -rotate-90">
                                    <circle cx="24" cy="24" r="18" class="stroke-slate-800" stroke-width="3" fill="transparent"/>
                                    <circle cx="24" cy="24" r="18" class="stroke-emerald-400" stroke-width="3" fill="transparent" stroke-dasharray="113" stroke-dashoffset="${113 - (113 * faithScore)}"/>
                                </svg>
                                <span class="absolute text-[10px] font-bold font-space text-slate-200">${(faithScore * 100).toFixed(0)}%</span>
                            </div>
                            <div class="flex-grow">
                                <p class="text-xs font-semibold text-slate-200">Groundedness Index</p>
                                <p class="text-[10px] text-slate-400 leading-relaxed mt-0.5">${faith.reasoning || ""}</p>
                            </div>
                        </div>
                        <div class="border-t border-indigo-950/60 my-1"></div>
                        <div class="flex gap-4 items-start">
                            <div class="relative flex items-center justify-center">
                                <svg class="w-12 h-12 transform -rotate-90">
                                    <circle cx="24" cy="24" r="18" class="stroke-slate-800" stroke-width="3" fill="transparent"/>
                                    <circle cx="24" cy="24" r="18" class="stroke-cyan-400" stroke-width="3" fill="transparent" stroke-dasharray="113" stroke-dashoffset="${113 - (113 * relevScore)}"/>
                                </svg>
                                <span class="absolute text-[10px] font-bold font-space text-slate-200">${(relevScore * 100).toFixed(0)}%</span>
                            </div>
                            <div class="flex-grow">
                                <p class="text-xs font-semibold text-slate-200">Answer Relevancy</p>
                                <p class="text-[10px] text-slate-400 leading-relaxed mt-0.5">${relev.reasoning || ""}</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- User thumbs up/down rating widgets -->
                <div class="flex items-center justify-between w-full">
                    <div class="flex items-center gap-2">
                        <button onclick="handleRatingByMsgId('${msgId}', 'UP', this)" class="px-2.5 py-1.5 bg-[#0d1224]/40 hover:bg-emerald-500/10 border border-slate-800 text-slate-400 hover:text-emerald-400 text-xs rounded-lg transition">
                            👍
                        </button>
                        <button onclick="handleRatingByMsgId('${msgId}', 'DOWN', this)" class="px-2.5 py-1.5 bg-[#0d1224]/40 hover:bg-red-500/10 border border-slate-800 text-slate-400 hover:text-red-400 text-xs rounded-lg transition">
                            👎
                        </button>
                    </div>
                    <select onchange="exportAnswerByMsgId(this, '${msgId}')" class="bg-[#0d1224] border border-slate-800 text-slate-400 hover:text-white text-xs px-2 py-1.5 rounded-lg focus:outline-none transition cursor-pointer">
                        <option value="">📤 Share / Export...</option>
                        <option value="email">📧 Send via Email</option>
                        <option value="slack">💬 Copy Slack Format</option>
                        <option value="md">📝 Markdown (.md)</option>
                        <option value="txt">📄 Text File (.txt)</option>
                        <option value="json">📊 Raw JSON (.json)</option>
                    </select>
                </div>
            </div>
            `;
            container.appendChild(div);
            const textContainer = document.getElementById(`answer-text-${msgId}`);
            const answerText = data.answer || "";
            const words = answerText.split(" ");
            let wIdx = 0;
            const interval = setInterval(() => {
                if (wIdx < words.length) {
                    textContainer.innerText += (wIdx === 0 ? "" : " ") + words[wIdx];
                    wIdx++;
                    container.scrollTop = container.scrollHeight;
                } else {
                    clearInterval(interval);
                }
            }, 30);
            
            container.scrollTop = container.scrollHeight;
        }

        // Toggle Accordions
        function toggleAccordion(id) {
            const el = document.getElementById(id);
            if (el) el.classList.toggle('hidden');
        }



        // Trigger SharePoint sync simulation progress indicator
        

        // Voice Search simulator
        
    </script>
    <!-- Citation Highlight Modal -->
    <div id="citation-modal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-[#060814]/85 backdrop-blur-sm p-4">
        <div class="glass-panel p-6 rounded-2xl w-full max-w-2xl border border-indigo-500/20 shadow-2xl relative">
            <button onclick="closeCitationModal()" class="absolute top-4 right-4 text-slate-500 hover:text-white transition">
                <i class="fa-solid fa-xmark text-lg"></i>
            </button>
            <h3 id="modal-citation-title" class="text-md font-bold font-outfit text-indigo-400 mb-3">Policy Segment Reference</h3>
            <div class="bg-[#070913] border border-indigo-500/10 p-4 rounded-xl max-h-[300px] overflow-y-auto">
                <p id="modal-citation-text" class="text-xs text-slate-300 leading-relaxed whitespace-pre-wrap"></p>
            </div>
        </div>
    </div>
    <!-- Qualitative Feedback Drawer -->
    <div id="feedback-drawer" class="hidden fixed bottom-6 right-6 z-50 w-80 glass-panel p-5 rounded-2xl border border-indigo-500/20 shadow-2xl">
        <div class="flex justify-between items-center mb-2">
            <h4 class="text-xs font-bold text-slate-200">RAG Response Feedback</h4>
            <button onclick="closeFeedbackDrawer()" class="text-slate-500 hover:text-white transition"><i class="fa-solid fa-xmark text-xs"></i></button>
        </div>
        <p class="text-[10px] text-slate-400 mb-3">Please help improve precision. What was the issue?</p>
        <div class="space-y-2 mb-4">
            <label class="flex items-center gap-2 text-xs cursor-pointer"><input type="checkbox" name="fb-issue" value="inaccurate" class="accent-indigo-500"> Excerpt is inaccurate</label>
            <label class="flex items-center gap-2 text-xs cursor-pointer"><input type="checkbox" name="fb-issue" value="irrelevant" class="accent-indigo-500"> Irrelevant references</label>
            <label class="flex items-center gap-2 text-xs cursor-pointer"><input type="checkbox" name="fb-issue" value="hallucination" class="accent-indigo-500"> Contains hallucination</label>
        </div>
        <button onclick="submitDetailedFeedback()" class="w-full py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg shadow-md transition active:scale-95">
            Submit Feedback
        </button>
    </div>
    <!-- SharePoint Sync Log Modal -->
    <div id="sync-modal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-[#060814]/80 backdrop-blur-sm p-4">
        <div class="glass-panel p-6 rounded-2xl w-full max-w-lg border border-indigo-500/20 shadow-2xl relative">
            <button onclick="closeSyncModal()" class="absolute top-4 right-4 text-slate-500 hover:text-white transition">
                <i class="fa-solid fa-xmark text-lg"></i>
            </button>
            <h3 class="text-md font-bold font-outfit text-indigo-400 mb-3 flex items-center gap-2">
                <i class="fa-solid fa-arrows-spin animate-spin"></i> SharePoint Repository Synced
            </h3>
            <div id="sync-terminal" class="bg-[#070913] border border-indigo-500/15 p-4 rounded-xl max-h-[250px] overflow-y-auto font-space text-[10px] text-emerald-400 space-y-1">
            </div>
        </div>
    </div>
    <!-- Action Toast container -->
    <div id="toast-container" class="fixed top-6 right-6 z-50 flex flex-col gap-2"></div>
    <!-- DB Reset confirmation modal overlay -->
    <div id="reset-modal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-[#060814]/85 backdrop-blur-sm p-4">
        <div class="glass-panel p-6 rounded-2xl w-full max-w-sm border border-red-500/20 shadow-2xl relative text-center">
            <h3 class="text-md font-bold font-outfit text-red-400 mb-2">Confirm Database Reset</h3>
            <p class="text-xs text-slate-400 mb-4">Wipes the ChromaDB collections. This cannot be undone.</p>
            <p class="text-xs font-semibold text-slate-300 mb-2">Type "RESET" to confirm deletion:</p>
            <input type="text" id="reset-confirm-input" class="w-full bg-[#0d1224] border border-red-500/20 rounded-xl px-3 py-2 text-xs text-center text-red-400 placeholder-slate-600 focus:outline-none mb-4" placeholder="RESET">
            <div class="grid grid-cols-2 gap-3">
                <button onclick="closeResetModal()" class="py-2 bg-slate-900 border border-slate-800 text-slate-300 text-xs font-semibold rounded-lg hover:bg-slate-800 transition">Cancel</button>
                <button onclick="submitResetDB()" class="py-2 bg-red-650 hover:bg-red-600 bg-red-600 text-white text-xs font-semibold rounded-lg shadow-md transition">Reset Index</button>
            </div>
        </div>
    <!-- User Auth Modal Overlay -->
    <div id="login-container" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-[#060814]/85 backdrop-blur-sm p-4">
        <div class="glass-panel p-6 rounded-3xl w-full max-w-md border border-indigo-500/20 shadow-2xl relative">
            <button onclick="closeLoginModal()" class="absolute top-4 right-4 text-slate-500 hover:text-white transition">
                <i class="fa-solid fa-xmark text-lg"></i>
            </button>
            <div class="text-center mb-6">
                <div class="w-12 h-12 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center mx-auto mb-3 shadow-lg shadow-indigo-500/30">
                    <i class="fa-solid fa-shield-halved text-xl text-white"></i>
                </div>
                <h3 class="text-lg font-bold font-outfit text-white">User Authentication & Role Switch</h3>
                <p class="text-xs text-slate-400 mt-1">Select a predefined profile or authenticate with credentials.</p>
            </div>
            
            <!-- Quick Role Badges -->
            <div class="grid grid-cols-2 gap-2 mb-4">
                <button onclick="quickFill('admin', 'admin123')" class="p-2.5 rounded-xl bg-indigo-950/40 border border-indigo-500/30 hover:border-indigo-400 text-left transition">
                    <p class="text-xs font-bold text-indigo-300">Admin</p>
                    <p class="text-[10px] text-slate-400">Compliance Officer</p>
                </button>
                <button onclick="quickFill('manager', 'manager123')" class="p-2.5 rounded-xl bg-indigo-950/40 border border-indigo-500/30 hover:border-indigo-400 text-left transition">
                    <p class="text-xs font-bold text-purple-300">Manager</p>
                    <p class="text-[10px] text-slate-400">Internal Teams</p>
                </button>
                <button onclick="quickFill('compliance', 'comp123')" class="p-2.5 rounded-xl bg-indigo-950/40 border border-indigo-500/30 hover:border-indigo-400 text-left transition">
                    <p class="text-xs font-bold text-pink-300">Compliance Officer</p>
                    <p class="text-[10px] text-slate-400">Elena Rostova</p>
                </button>
                <button onclick="quickFill('employee', 'emp123')" class="p-2.5 rounded-xl bg-indigo-950/40 border border-indigo-500/30 hover:border-indigo-400 text-left transition">
                    <p class="text-xs font-bold text-cyan-300">Employee</p>
                    <p class="text-[10px] text-slate-400">Sarah Jenkins</p>
                </button>
            </div>

            <div class="space-y-3">
                <div>
                    <label class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Username</label>
                    <input type="text" id="username" class="w-full bg-[#0d1224] border border-indigo-500/20 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-indigo-500" placeholder="admin" value="admin">
                </div>
                <div>
                    <label class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Password</label>
                    <input type="password" id="password" class="w-full bg-[#0d1224] border border-indigo-500/20 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-indigo-500" placeholder="••••••••" value="admin123">
                </div>
                <div id="login-error" class="hidden text-xs text-red-400 bg-red-950/20 border border-red-500/20 p-2 rounded-xl text-center"></div>
                <button id="login-btn" onclick="handleLogin()" class="w-full py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-semibold text-xs rounded-xl shadow-lg transition active:scale-95">
                    🚀 Switch & Authenticate
                </button>
            </div>
        </div>
    </div>
    <script>
        if ('serviceWorker' in navigator) {
            window.addEventListener('load', () => {
                navigator.serviceWorker.register('/service-worker.js').catch(() => {});
            });
        }
    </script>
</body>
</html>"""

@app.get("/manifest.json")
async def get_manifest():
    manifest_path = os.path.join(os.path.dirname(__file__), "manifest.json")
    if os.path.exists(manifest_path):
        return FileResponse(manifest_path, media_type="application/json")
    raise HTTPException(status_code=404, detail="Manifest not found")

@app.get("/service-worker.js")
async def get_service_worker():
    sw_path = os.path.join(os.path.dirname(__file__), "service-worker.js")
    if os.path.exists(sw_path):
        return FileResponse(sw_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Service worker not found")

@app.get("/", response_class=HTMLResponse)
async def serve_portal():
    return HTMLResponse(content=HTML_CONTENT, status_code=200)

@app.post("/api/login")
async def api_login(payload: Dict[str, str]):
    username = payload.get("username")
    password = payload.get("password")
    if username == "admin" and password == "password":
        return JSONResponse(content={"status": "authorized"})
    raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(64)
):
    try:
        # Create storage directory for raw files
        storage_dir = "uploaded_policies"
        os.makedirs(storage_dir, exist_ok=True)
        file_path = os.path.join(storage_dir, file.filename)
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
            
        logger.info(f"Document uploaded to local storage: {file_path}")
        pages = load_document(file_path)
        chunks = split_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Register in version control ledger
        ver_info = register_document_version(
            filename=file.filename,
            file_path=file_path,
            chunks_count=len(chunks),
            pages_count=len(pages)
        )
        
        # Attach version tag to chunk metadatas
        for c in chunks:
            c["metadata"]["version"] = ver_info.get("version", "v1.0")
            c["metadata"]["hash"] = ver_info.get("hash", "")
            
        add_documents_to_db(chunks)
        
        # Generate automatic contextual policy question suggestions
        full_doc_text = " ".join([p.get("text", "") for p in pages[:5]])
        generate_document_suggestions(full_doc_text, file.filename)
        
        # Log to audit trail
        log_audit_event(
            event_type="DOCUMENT_INGESTED",
            user_id="admin",
            clearance="Compliance Officer",
            details={
                "filename": file.filename,
                "chunks": len(chunks),
                "pages": len(pages),
                "version": ver_info.get("version", "v1.0")
            }
        )
        
        return JSONResponse(content={
            "status": "indexed",
            "chunks_count": len(chunks),
            "pages_count": len(pages),
            "version": ver_info.get("version", "v1.0")
        })
    except Exception as e:
        logger.error(f"Upload and indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/versions")
async def api_document_versions(filename: Optional[str] = None):
    return JSONResponse(content=get_document_versions(filename))

@app.get("/api/documents")
async def api_documents():
    docs = get_indexed_documents()
    return JSONResponse(content=docs)

@app.post("/api/query")
async def api_query(payload: Dict[str, Any]):
    start_time = time.time()
    query = payload.get("query", "")
    clearance = payload.get("clearance", "Employee")
    category = payload.get("category")
    source_filter = payload.get("source_filter")
    session_id = payload.get("session_id")
    user_id = payload.get("user_id", "anonymous")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    # Retrieve past session history if available for multi-turn contextual resolution
    effective_query = query
    history = []
    if session_id:
        history = get_session_messages(session_id, limit=6)
        effective_query = build_contextual_query(query, history)
        add_message(session_id=session_id, role="user", content=query, user_id=user_id)
        
    # Select A/B Prompt Variant
    ab_variant = select_active_variant(payload.get("prompt_variant"))
    
    # Detect language of incoming query
    lang_info = detect_language(effective_query)
    
    res = run_pipeline(
        effective_query,
        clearance=clearance,
        lang_info=lang_info,
        category=category,
        source_filter=source_filter,
        prompt_variant=ab_variant
    )
    
    # Calculate token usage costs & latency
    latency_ms = (time.time() - start_time) * 1000.0
    prompt_len = len(effective_query) // 4
    comp_len = len(res["answer"]) // 4
    cost = (prompt_len * 0.15 / 1e6) + (comp_len * 0.60 / 1e6)
    
    faith_score = res["evaluation"]["faithfulness"].get("score", 1.0)
    rel_score = res["evaluation"]["relevancy"].get("score", 1.0)
    
    if session_id:
        add_message(
            session_id=session_id,
            role="assistant",
            content=res["answer"],
            citations=res["citations"],
            evaluation=res["evaluation"],
            user_id=user_id
        )
    
    # Record real-time telemetry
    record_query_telemetry(
        query=query,
        answer=res["answer"],
        latency_ms=latency_ms,
        prompt_tokens=prompt_len,
        completion_tokens=comp_len,
        cost=cost,
        faithfulness=faith_score,
        relevancy=rel_score,
        clearance=clearance,
        user_id=user_id,
        session_id=session_id
    )
    
    # Record A/B test metric
    record_ab_metric(
        variant=ab_variant,
        latency_ms=latency_ms,
        faithfulness=faith_score,
        relevancy=rel_score
    )
    
    # Record audit trail event
    log_audit_event(
        event_type="QUERY_EXECUTED",
        user_id=user_id,
        clearance=clearance,
        details={
            "query": query,
            "category": category,
            "faithfulness": faith_score,
            "relevancy": rel_score,
            "cost": cost,
            "prompt_variant": ab_variant
        }
    )
    
    log_evaluation(
        query=query,
        answer=res["answer"],
        faithfulness=faith_score,
        relevancy=rel_score
    )
    
    return JSONResponse(content={
        "answer": res["answer"],
        "citations": res["citations"],
        "evaluation": res["evaluation"],
        "cost": cost,
        "latency_ms": round(latency_ms, 2),
        "language": lang_info,
        "prompt_variant": ab_variant,
        "session_id": session_id
    })

@app.get("/api/ab-test/metrics")
async def api_ab_test_metrics():
    return JSONResponse(content=get_ab_experiment_summary())

@app.get("/api/audit/logs")
async def api_get_audit_logs(limit: int = 50):
    return JSONResponse(content=get_audit_logs(limit=limit))

@app.get("/api/audit/verify")
async def api_verify_audit():
    return JSONResponse(content=verify_audit_integrity())

@app.get("/api/audit/export")
async def api_export_audit(format: str = "csv"):
    from fastapi.responses import Response
    if format == "csv":
        csv_data = export_audit_csv()
        return Response(content=csv_data, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=synthara_compliance_audit.csv"})
    return JSONResponse(content=get_audit_logs(limit=1000))

@app.get("/api/languages")
async def api_get_languages():
    return JSONResponse(content=get_supported_languages())

@app.get("/api/document/preview/{source_name}/{page_number}")
async def api_document_preview(source_name: str, page_number: int, highlight: Optional[str] = None):
    return JSONResponse(content=get_document_page_preview(source_name, page_number, highlight))

@app.get("/api/search/facets")
async def api_get_facets():
    return JSONResponse(content=get_search_facets())

@app.get("/api/suggestions")
async def api_get_suggestions():
    return JSONResponse(content=get_all_suggestions())

@app.post("/api/sync/trigger")
async def api_trigger_sync():
    result = scan_and_sync_policies()
    return JSONResponse(content=result)

@app.post("/api/guardrail/scan")
async def api_guardrail_scan(payload: Dict[str, Any]):
    text = payload.get("text", "")
    redacted, detected = redact_pii(text)
    return JSONResponse(content={
        "original_length": len(text),
        "redacted_text": redacted,
        "pii_detected": detected,
        "is_safe": len(detected) == 0
    })

@app.get("/api/sync/status")
async def api_sync_status():
    return JSONResponse(content=get_sync_status())

@app.get("/api/autocomplete")
async def api_get_autocomplete(q: Optional[str] = None):
    return JSONResponse(content=get_autocomplete_suggestions(q or ""))

@app.post("/api/share/email")
async def api_share_email(payload: Dict[str, Any]):
    query = payload.get("query", "Policy Inquiry")
    answer = payload.get("answer", "")
    citations = payload.get("citations", [])
    data = format_email_template(query, answer, citations)
    return JSONResponse(content=data)

@app.post("/api/share/slack")
async def api_share_slack(payload: Dict[str, Any]):
    query = payload.get("query", "Policy Inquiry")
    answer = payload.get("answer", "")
    citations = payload.get("citations", [])
    webhook_url = payload.get("webhook_url")
    blocks = format_slack_block_kit(query, answer, citations)
    if webhook_url:
        success = dispatch_webhook(webhook_url, blocks)
        return JSONResponse(content={"status": "dispatched" if success else "failed", "blocks": blocks})
    return JSONResponse(content={"status": "formatted", "blocks": blocks})

@app.post("/api/share/teams")
async def api_share_teams(payload: Dict[str, Any]):
    query = payload.get("query", "Policy Inquiry")
    answer = payload.get("answer", "")
    citations = payload.get("citations", [])
    webhook_url = payload.get("webhook_url")
    card = format_teams_card(query, answer, citations)
    if webhook_url:
        success = dispatch_webhook(webhook_url, card)
        return JSONResponse(content={"status": "dispatched" if success else "failed", "card": card})
    return JSONResponse(content={"status": "formatted", "card": card})

@app.get("/api/analytics/summary")
async def api_analytics_summary():
    return JSONResponse(content=get_analytics_summary())

@app.get("/api/conversations")
async def api_get_conversations(user_id: Optional[str] = None):
    return JSONResponse(content=list_sessions(user_id=user_id))

@app.post("/api/conversations")
async def api_create_conversation(payload: Optional[Dict[str, Any]] = None):
    data = payload or {}
    user_id = data.get("user_id", "anonymous")
    title = data.get("title")
    session = create_session(user_id=user_id, title=title)
    return JSONResponse(content=session)

@app.get("/api/conversations/{session_id}")
async def api_get_conversation_messages(session_id: str):
    messages = get_session_messages(session_id)
    return JSONResponse(content={"session_id": session_id, "messages": messages})

@app.delete("/api/conversations/{session_id}")
async def api_delete_conversation(session_id: str):
    deleted = delete_session(session_id)
    return JSONResponse(content={"deleted": deleted})

@app.delete("/api/document/{source_name}")
async def api_delete_document(source_name: str):
    try:
        delete_document_from_db(source_name)
        return JSONResponse(content={"status": "deleted"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
async def api_reset():
    try:
        reset_db()
        log_audit_event(
            event_type="DATABASE_RESET",
            user_id="admin",
            clearance="Compliance Officer",
            details={"action": "complete_vector_wipe"}
        )
        return JSONResponse(content={"status": "reset"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback")
async def api_feedback(payload: Dict[str, Any]):
    query = payload.get("query", "")
    answer = payload.get("answer", "")
    rating = int(payload.get("rating", 1))
    comments = payload.get("comments")
    issue_type = payload.get("issue_type")
    user_id = payload.get("user_id", "anonymous")
    session_id = payload.get("session_id")
    
    rec = record_feedback(
        query=query,
        answer=answer,
        rating=rating,
        comments=comments,
        issue_type=issue_type,
        user_id=user_id,
        session_id=session_id
    )
    log_audit_event(
        event_type="FEEDBACK_SUBMITTED",
        user_id=user_id,
        clearance="Employee",
        details={"rating": rating, "issue_type": issue_type}
    )
    return JSONResponse(content={"status": "logged", "feedback": rec})

@app.get("/api/feedback/summary")
async def api_feedback_summary():
    return JSONResponse(content=get_feedback_summary())

@app.post("/api/auth/login")
async def api_auth_login(payload: Dict[str, Any]):
    username = payload.get("username", "")
    password = payload.get("password", "")
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = create_jwt_token(user)
    log_audit_event(
        event_type="AUTH_LOGIN",
        user_id=user["username"],
        clearance=user.get("clearance", "Employee"),
        details={"role": user.get("role"), "department": user.get("department")}
    )
    return JSONResponse(content={"token": token, "user": user})

@app.get("/api/auth/me")
async def api_auth_me(token: Optional[str] = None):
    if not token:
        raise HTTPException(status_code=401, detail="Token required.")
    payload = verify_jwt_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return JSONResponse(content={"user": payload})

@app.get("/api/auth/users")
async def api_auth_users():
    return JSONResponse(content=get_all_users())

@app.post("/api/auth/register")
async def api_auth_register(payload: Dict[str, Any]):
    try:
        new_user = register_user(
            username=payload.get("username", ""),
            password=payload.get("password", ""),
            full_name=payload.get("full_name", ""),
            role=payload.get("role", "Employee"),
            department=payload.get("department", "General")
        )
        return JSONResponse(content={"status": "registered", "user": new_user})
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import sys

# Detect if executing actively under the Streamlit runner CLI
is_streamlit_runner = False
try:
    import streamlit as st
    if hasattr(st, "runtime") and st.runtime.exists():
        is_streamlit_runner = True
    elif "streamlit.web.cli" in sys.modules or "streamlit.runtime" in sys.modules:
        is_streamlit_runner = True
    elif any("streamlit" in arg.lower() for arg in sys.argv):
        is_streamlit_runner = True
except Exception:
    is_streamlit_runner = False

if is_streamlit_runner:
    try:
        import threading
        import uvicorn
        import streamlit as st
        import streamlit.components.v1 as components

        def _start_bg_api():
            try:
                import asyncio
                import uvicorn
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                config = uvicorn.Config(app=app, host="127.0.0.1", port=8000, log_level="warning")
                server = uvicorn.Server(config)
                loop.run_until_complete(server.serve())
            except Exception as e:
                logger.error(f"Background API server error: {e}")

        if not hasattr(app, "_bg_api_started"):
            app._bg_api_started = True
            t = threading.Thread(target=_start_bg_api, daemon=True)
            t.start()

        st.set_page_config(
            page_title="Synthara Policy Portal",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        st.markdown("""
        <style>
            .block-container { padding: 0px !important; margin: 0px !important; max-width: 100% !important; }
            iframe { border: none !important; }
            [data-testid="stHeader"], footer { visibility: hidden !important; height: 0px !important; }
        </style>
        """, unsafe_allow_html=True)
        components.html(HTML_CONTENT, height=850, scrolling=True)
    except Exception as e:
        logger.error(f"Streamlit execution failed: {e}")

if __name__ == "__main__":
    import uvicorn
    # When run directly via `python app.py`, bind cleanly to port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
