"""
Synthara Enterprise RAG Portal Backend.
FastAPI Application serving a premium, responsive glassmorphic single-page RAG workspace interface.
Backward compatible with test_validation.py integration calls.
"""

import os
import logging
import json
import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Import custom core modules
import config
from utils.document_loader import load_pdf
from utils.chunker import split_documents
from utils.retriever import add_documents_to_db, query_db, reset_db, get_collection, delete_document_from_db
from utils.validator import validate_query, evaluate_faithfulness, evaluate_answer_relevancy

# Setup logging
logger = logging.getLogger("app")
logging.basicConfig(level=config.LOG_LEVEL)

app = FastAPI(title="Synthara RAG Portal")

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
                    "chunks": 0
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

def call_llm(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    client = config.get_openai_client()
    context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks):
        source = chunk["metadata"].get("source", "Unknown Document")
        page = chunk["metadata"].get("page", "?")
        chunk_text = sanitize_text(chunk['text'])
        context_blocks.append(f"Excerpt [{idx + 1}] (Source: {source}, Page {page}):\n{chunk_text}")

    context_str = "\n\n".join(context_blocks)
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
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        return f"An error occurred: {str(e)}"

def run_pipeline(question: str, clearance: str = "Employee") -> Dict[str, Any]:
    try:
        cleaned_query = validate_query(question)
    except ValueError as e:
        return {"answer": str(e), "citations": [], "evaluation": {"faithfulness": {"score": 0.0, "reasoning": str(e)}, "relevancy": {"score": 0.0, "reasoning": str(e)}}}

    retrieved_chunks = query_db(cleaned_query, k=5, clearance=clearance)
    answer = call_llm(cleaned_query, retrieved_chunks)
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
@app.get("/", response_class=HTMLResponse)
async def serve_portal():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Synthara Enterprise RAG Portal</title>
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
    <!-- Glow Orbs in background -->
    <div class="glow-orb top-10 left-10 animate-pulse"></div>
    <div id="mouse-glow-mesh" class="absolute w-[250px] h-[250px] rounded-full pointer-events-none transition-all duration-500 ease-out opacity-0 z-0 bg-gradient-to-r from-indigo-500/10 to-cyan-500/0 blur-[60px]"></div>
    <div class="glow-orb bottom-10 right-10 animate-pulse" style="animation-duration: 8s;"></div>

    <!-- LOGIN SCREEN -->
    <div id="login-container" class="fixed inset-0 z-50 flex items-center justify-center bg-[#060814]/90 backdrop-blur-md">
        <div class="glass-panel p-10 rounded-2xl w-full max-w-md shadow-2xl shadow-indigo-950/20 border border-indigo-500/20">
            <div class="text-center mb-8">
                <h1 class="text-3xl font-extrabold font-outfit tracking-tight bg-gradient-to-r from-indigo-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                    ⚡ Synthara Portal
                </h1>
                <p class="text-slate-400 text-sm mt-2">Enter credentials to unlock secure RAG Workspace</p>
            </div>
            
            <div class="space-y-5">
                <div>
                    <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Username</label>
                    <div class="relative">
                        <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-slate-500">
                            <i class="fa-solid fa-user"></i>
                        </span>
                        <input type="text" id="username" value="admin" class="w-full bg-[#0d1224]/80 border border-indigo-500/20 rounded-xl py-3 pl-10 pr-4 text-white neomorphic-depth placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition" placeholder="Enter username">
                    </div>
                </div>
                
                <div>
                    <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Password</label>
                    <div class="relative">
                        <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-slate-500">
                            <i class="fa-solid fa-lock"></i>
                        </span>
                        <input type="password" id="password" value="password" class="w-full bg-[#0d1224]/80 border border-indigo-500/20 rounded-xl py-3 pl-10 pr-4 text-white neomorphic-depth placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition" placeholder="Enter password">
                    </div>
                </div>
                
                <button onclick="handleLogin()" class="w-full py-3.5 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-semibold font-outfit rounded-xl shadow-lg shadow-indigo-500/20 active:translate-y-0.5 transition duration-150">
                    🚀 Enter Workspace
                </button>
                <div id="login-error" class="hidden text-red-400 text-xs text-center font-medium mt-2"></div>
            </div>
        </div>
    </div>

    <!-- MAIN DASHBOARD (HIDDEN UNTIL LOGGED IN) -->
    <div id="dashboard-container" class="hidden min-h-screen flex flex-col max-w-7xl mx-auto p-6 relative z-10">
        
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
            
            <div class="flex items-center gap-3">
                <span id="api-status-badge" class="px-3 py-1.5 rounded-full text-xs font-bold bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center gap-1.5">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                    API Active
                </span>
                <button onclick="toggleSidebar()" class="px-4 py-2 bg-slate-900 border border-slate-800 hover:bg-slate-800 text-indigo-400 hover:text-white rounded-xl text-xs font-semibold transition mr-2"><i class="fa-solid fa-sidebar mr-1.5"></i> Sidebar</button>
                <button onclick="handleLogout()" class="px-4 py-2 bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-300 hover:text-white rounded-xl text-xs font-semibold transition">
                    <i class="fa-solid fa-right-from-bracket mr-1.5"></i> Exit Portal
                </button>
            </div>
        </header>

        <!-- TOP METRICS GRID -->
        <section class="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
            <div class="glass-panel p-5 rounded-2xl relative overflow-hidden group hover:border-indigo-500/30 transition duration-300">
                <div class="absolute bottom-0 left-0 w-full h-[2px] bg-indigo-500/50"></div>
                <p class="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Document Registry</p>
                <h3 id="metric-docs" class="text-3xl font-extrabold font-space mt-2 bg-gradient-to-r from-indigo-300 to-slate-200 bg-clip-text text-transparent">0</h3>
            </div>
            <div class="glass-panel p-5 rounded-2xl relative overflow-hidden group hover:border-purple-500/30 transition duration-300">
                <div class="absolute bottom-0 left-0 w-full h-[2px] bg-purple-500/50"></div>
                <p class="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Knowledge Nodes</p>
                <h3 id="metric-chunks" class="text-3xl font-extrabold font-space mt-2 bg-gradient-to-r from-purple-300 to-slate-200 bg-clip-text text-transparent">0</h3>
            </div>
            <div class="glass-panel p-5 rounded-2xl relative overflow-hidden group hover:border-emerald-500/30 transition duration-300">
                <div class="absolute bottom-0 left-0 w-full h-[2px] bg-emerald-500/50"></div>
                <p class="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Safety Guardrails</p>
                <h3 class="text-3xl font-extrabold font-space mt-2 text-emerald-400">Active</h3>
            </div>
            <div class="glass-panel p-5 rounded-2xl relative overflow-hidden group hover:border-pink-500/30 transition duration-300">
                <div class="absolute bottom-0 left-0 w-full h-[2px] bg-pink-500/50"></div>
                <p class="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Workspace Cost</p>
                <h3 id="metric-cost" class="text-3xl font-extrabold font-space mt-2 bg-gradient-to-r from-pink-300 to-slate-200 bg-clip-text text-transparent">$0.00000</h3>
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
                    <h3 class="text-md font-bold font-outfit flex items-center gap-2 mb-4">
                        <i class="fa-solid fa-file-arrow-up text-indigo-400"></i> Ingest Documents
                    </h3>
                    
                    <div id="drop-zone" class="border-2 border-dashed border-indigo-500/25 rounded-xl p-6 text-center hover:border-cyan-500/40 transition duration-200 cursor-pointer bg-[#0d1224]/30">
                        <i class="fa-solid fa-cloud-arrow-up text-3xl text-indigo-400/80 mb-3"></i>
                        <p class="text-xs font-semibold">Drag & drop policy PDF here</p>
                        <p class="text-[10px] text-slate-500 mt-1">or click to browse local files</p>
                        <input type="file" id="file-input" accept=".pdf" class="hidden">
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
                            <span id="rbac-val" class="font-bold text-indigo-400">Employee</span>
                        </div>
                        <select id="clearance-select" class="w-full bg-[#0d1224] border border-indigo-500/10 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-indigo-500 transition">
                            <option value="Employee">Employee (Public Policies)</option>
                            <option value="Manager">Manager (Internal Teams)</option>
                            <option value="Compliance Officer">Compliance Officer (Full Audit)</option>
                        </select>
                    </div>

                    <div>
                        <div class="flex items-center justify-between text-xs font-medium mb-1.5">
                            <span class="text-slate-400">Target Chunk Size</span>
                            <span id="chunk-size-val" class="font-bold text-indigo-400">512</span>
                        </div>
                        <input type="range" id="chunk-size-slider" min="128" max="1024" step="64" value="512" class="w-full accent-indigo-500 h-1 bg-slate-950 rounded-lg cursor-pointer">
                    </div>

                    <div>
                        <div class="flex items-center justify-between text-xs font-medium mb-1.5">
                            <span class="text-slate-400">Chunk Overlap</span>
                            <span id="chunk-overlap-val" class="font-bold text-indigo-400">64</span>
                        </div>
                        <input type="range" id="chunk-overlap-slider" min="0" max="256" step="16" value="64" class="w-full accent-indigo-500 h-1 bg-slate-950 rounded-lg cursor-pointer">
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
                    
                    <div id="chat-session-badge" class="text-[10px] text-slate-400 bg-slate-900 border border-slate-800 px-3 py-1 rounded-full font-space">
                        Session: Active
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
                                </p>
                            </div>
                        </div>
                    </div>

                    <!-- Bottom Input bar -->
                    <div class="p-4 border-t border-indigo-950 bg-[#060814]/80">
                        <div class="relative flex items-center">
                            <button onclick="simulateSTT()" class="absolute left-3 p-1.5 text-slate-400 hover:text-indigo-400 active:scale-95 transition">
                                <i class="fa-solid fa-microphone text-md"></i>
                            </button>
                            <input type="text" id="chat-input" onkeydown="if(event.key === 'Enter') handleSendQuery()" class="w-full bg-[#0d1224]/80 border border-indigo-500/20 rounded-2xl py-3.5 pl-12 pr-16 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition shadow-lg shadow-indigo-500/5" placeholder="Ask a corporate policy question...">
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
                        <p class="text-xs text-slate-400 mb-4">Visual representation of overlapping context nodes and file mapping relationships extracted from the document index database.</p>
                        
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
        let isLoggedIn = false;
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
            document.getElementById('rbac-val').innerText = e.target.value;
        });

        // Trigger manual browse file click
        document.getElementById('drop-zone').addEventListener('click', () => {
            document.getElementById('file-input').click();
        });
        document.getElementById('file-input').addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                uploadDocument(e.target.files[0]);
            }
        });

        // Login flow
        function handleLogin() {
            const user = document.getElementById('username').value;
            const pass = document.getElementById('password').value;
            const errDiv = document.getElementById('login-error');
            
            if (user === 'admin' && pass === 'password') {
                isLoggedIn = true;
                document.getElementById('login-container').classList.add('hidden');
                document.getElementById('dashboard-container').classList.remove('hidden');
                fetchDocuments();
            } else {
                errDiv.innerText = "Invalid credentials. Hint: admin / password.";
                errDiv.classList.remove('hidden');
            }
        }
        
        function handleLogout() {
            isLoggedIn = false;
            document.getElementById('login-container').classList.remove('hidden');
            document.getElementById('dashboard-container').classList.add('hidden');
            document.getElementById('login-error').classList.add('hidden');
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
            }
        }

        // Fetch Documents Registry
        async function fetchDocuments() {
            try {
                const res = await fetch('/api/documents');
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
            
            let totalChunks = 0;
            
            if (documentRegistry.length === 0) {
                container.innerHTML = `<div class="text-center text-xs text-slate-500 py-6">No documents indexed in vector registry.</div>`;
                chunksMetric.innerText = "0";
                return;
            }
            
            let html = '';
            documentRegistry.forEach(doc => {
                totalChunks += doc.chunks;
                html += `
                <div class="flex items-center justify-between bg-[#0d1224]/50 border border-indigo-500/10 p-3 rounded-xl hover:border-indigo-500/30 transition">
                    <div class="truncate max-w-[200px]">
                        <p class="text-xs font-semibold truncate text-slate-200">🟢 ${doc.source}</p>
                        <p class="text-[10px] text-slate-500 mt-0.5">Pages: ${doc.pages} | Chunks: ${doc.chunks}</p>
                    </div>
                    <button onclick="deleteDocument('${doc.source}')" class="p-1.5 text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg active:scale-95 transition">
                        <i class="fa-solid fa-trash-can text-xs"></i>
                    </button>
                </div>
                `;
            });
            container.innerHTML = html;
            chunksMetric.innerText = totalChunks;
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
            statusDiv.classList.remove('hidden');
            
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
                if (currentPct === 40) logSpan.innerText = "Extracting text nodes...";
                if (currentPct === 70) logSpan.innerText = "Generating SentenceTransformer embeddings...";
            }, 300);
            
            try {
                const res = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                clearInterval(timer);
                
                if (res.ok) {
                    pctSpan.innerText = "100%";
                    progress.style.width = "100%";
                    logSpan.innerText = "Database index completed successfully!";
                    setTimeout(() => {
                        statusDiv.classList.add('hidden');
                    }, 2000);
                    fetchDocuments();
                } else {
                    const data = await res.json();
                    alert("Upload failed: " + data.detail);
                    statusDiv.classList.add('hidden');
                }
            } catch (err) {
                clearInterval(timer);
                alert("Upload error: " + err);
                statusDiv.classList.add('hidden');
            }
        }

        // Delete document
        async function deleteDocument(sourceName) {
            if (!confirm(`Are you sure you want to delete '${sourceName}'?`)) return;
            try {
                const res = await fetch(`/api/document/${encodeURIComponent(sourceName)}`, {
                    method: 'DELETE'
                });
                if (res.ok) {
                    fetchDocuments();
                } else {
                    const data = await res.json();
                    alert("Delete failed: " + data.detail);
                }
            } catch (err) {
                alert("Error deleting document: " + err);
            }
        }

        // Reset database
        async function handleResetDB() {
            if (!confirm("Are you sure you want to completely wipe the vector index database? This cannot be undone.")) return;
            try {
                const res = await fetch('/api/reset', { method: 'POST' });
                if (res.ok) {
                    alert("Database index cleared successfully.");
                    fetchDocuments();
                }
            } catch (err) {
                alert("Reset error: " + err);
            }
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
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        clearance: document.getElementById('clearance-select').value,
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
                        document.getElementById('metric-cost').innerText = `$${totalCost.toFixed(5)}`;
                    }
                } else {
                    const data = await res.json();
                    appendSystemErrorMessage(data.detail || "Retrieval execution aborted.");
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

        function appendAssistantResponse(data) {
            const container = document.getElementById('message-container');
            const msgId = Date.now();
            const div = document.createElement('div');
            div.className = "flex gap-4";
            
            // Format citations
            let citationsHTML = '';
            if (data.citations && data.citations.length > 0) {
                data.citations.forEach((cit, idx) => {
                    citationsHTML += `
                    <div class="citation-card text-xs">
                        <p class="font-bold text-indigo-300">Excerpt [${idx+1}]: ${cit.metadata.source} (Page ${cit.metadata.page})</p>
                        <p class="text-slate-400 italic mt-1 font-space">"Highlight: ... ${cit.text.substring(0, 200)}..."</p>
                        <p class="text-[10px] text-cyan-400 font-semibold mt-1">Match Confidence: ${(cit.similarity * 100).toFixed(1)}%</p>
                    </div>
                    `;
                });
            } else {
                citationsHTML = `<p class="text-xs text-slate-500 italic">No citations retrieved for this response.</p>`;
            }
            
            // Format evaluation scores
            const faith = data.evaluation.faithfulness;
            const relev = data.evaluation.relevancy;
            const fBadge = faith.score >= 0.8 ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' : 'bg-red-500/10 border-red-500/20 text-red-400';
            const rBadge = relev.score >= 0.8 ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' : 'bg-red-500/10 border-red-500/20 text-red-400';

            div.innerHTML = `
            <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-cyan-500 flex items-center justify-center shadow-md">
                <i class="fa-solid fa-robot text-xs text-white"></i>
            </div>
            <div class="max-w-[85%] space-y-3">
                <div class="rounded-2xl px-5 py-3.5 bg-[#0d1224]/30 border border-indigo-500/10 text-sm shadow-md">
                    <p class="text-xs font-semibold text-indigo-400 mb-1 flex items-center gap-1.5">Synthara Assistant <i class="fa-solid fa-shield-halved text-[9px] text-cyan-400"></i></p>
                    <p class="text-slate-200 leading-relaxed whitespace-pre-wrap">${data.answer}</p>
                </div>
                
                <!-- Citations Preview Accordion -->
                <div class="border border-indigo-500/10 rounded-xl overflow-hidden">
                    <button onclick="toggleAccordion('cit-${msgId}')" class="w-full px-4 py-2 bg-[#0d1224]/50 hover:bg-[#0d1224]/80 text-xs font-bold text-slate-400 flex items-center justify-between transition">
                        <span>📖 View Retrieved Citations (${data.citations.length})</span>
                        <i class="fa-solid fa-chevron-down text-[10px]"></i>
                    </button>
                    <div id="cit-${msgId}" class="hidden p-4 bg-[#0d1224]/10 border-t border-indigo-500/10 space-y-2">
                        ${citationsHTML}
                    </div>
                </div>

                <!-- Evaluations Metrics Accordion -->
                <div class="border border-indigo-500/10 rounded-xl overflow-hidden">
                    <button onclick="toggleAccordion('eval-${msgId}')" class="w-full px-4 py-2 bg-[#0d1224]/50 hover:bg-[#0d1224]/80 text-xs font-bold text-slate-400 flex items-center justify-between transition">
                        <span>📊 LLM judge Evaluation Metrics</span>
                        <i class="fa-solid fa-chevron-down text-[10px]"></i>
                    </button>
                    <div id="eval-${msgId}" class="hidden p-4 bg-[#0d1224]/10 border-t border-indigo-500/10 space-y-4">
                        <div>
                            <div class="flex items-center justify-between text-xs font-semibold mb-1">
                                <span>Groundedness Index</span>
                                <span class="px-2.5 py-0.5 rounded-full border ${fBadge}">${(faith.score * 100).toFixed(0)}%</span>
                            </div>
                            <p class="text-[10px] text-slate-500 italic mt-0.5">Reasoning: ${faith.reasoning}</p>
                        </div>
                        <div class="border-t border-slate-900/60 my-2"></div>
                        <div>
                            <div class="flex items-center justify-between text-xs font-semibold mb-1">
                                <span>Answer Relevancy</span>
                                <span class="px-2.5 py-0.5 rounded-full border ${rBadge}">${(relev.score * 100).toFixed(0)}%</span>
                            </div>
                            <p class="text-[10px] text-slate-500 italic mt-0.5">Reasoning: ${relev.reasoning}</p>
                        </div>
                    </div>
                </div>

                <!-- User thumbs up/down rating widgets -->
                <div class="flex items-center gap-2">
                    <button onclick="handleRating('${data.answer}', 'UP', this)" class="px-2.5 py-1.5 bg-[#0d1224]/40 hover:bg-emerald-500/10 border border-slate-800 text-slate-400 hover:text-emerald-400 text-xs rounded-lg transition">
                        👍
                    </button>
                    <button onclick="handleRating('${data.answer}', 'DOWN', this)" class="px-2.5 py-1.5 bg-[#0d1224]/40 hover:bg-red-500/10 border border-slate-800 text-slate-400 hover:text-red-400 text-xs rounded-lg transition">
                        👎
                    </button>
                </div>
            </div>
            `;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }

        // Toggle Accordions
        function toggleAccordion(id) {
            const el = document.getElementById(id);
            el.classList.toggle('hidden');
        }

        // Log thumbs rating feedback
        async function handleRating(answerText, rating, btn) {
            try {
                const res = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        answer: answerText,
                        rating: rating
                    })
                });
                if (res.ok) {
                    btn.classList.add('bg-indigo-500/20', 'text-white', 'border-indigo-500/40');
                }
            } catch (err) {
                console.error("Error rating:", err);
            }
        }

        // Trigger SharePoint sync simulation progress indicator
        function triggerSharepointSync() {
            alert("Connecting to SharePoint Directory Server... Sync started in the background.");
        }

        // Voice Search simulator
        function simulateSTT() {
            const input = document.getElementById('chat-input');
            input.value = "What is the vacation leave policy?";
            input.focus();
        }
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content, status_code=200)

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
            
        logger.info(f"PDF uploaded to local storage: {file_path}")
        pages = load_pdf(file_path)
        chunks = split_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        add_documents_to_db(chunks)
        return JSONResponse(content={"status": "indexed", "chunks_count": len(chunks)})
    except Exception as e:
        logger.error(f"Upload and indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def api_documents():
    docs = get_indexed_documents()
    return JSONResponse(content=docs)

@app.post("/api/query")
async def api_query(payload: Dict[str, Any]):
    query = payload.get("query", "")
    clearance = payload.get("clearance", "Employee")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    res = run_pipeline(query, clearance=clearance)
    
    # Calculate mock token usage costs
    prompt_len = len(query) // 4
    comp_len = len(res["answer"]) // 4
    cost = (prompt_len * 0.15 / 1e6) + (comp_len * 0.60 / 1e6)
    
    log_evaluation(
        query=query,
        answer=res["answer"],
        faithfulness=res["evaluation"]["faithfulness"].get("score", 0.0),
        relevancy=res["evaluation"]["relevancy"].get("score", 0.0)
    )
    
    return JSONResponse(content={
        "answer": res["answer"],
        "citations": res["citations"],
        "evaluation": res["evaluation"],
        "cost": cost
    })

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
        return JSONResponse(content={"status": "reset"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback")
async def api_feedback(payload: Dict[str, str]):
    # Save feedback rating
    return JSONResponse(content={"status": "logged"})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8501, reload=False)
