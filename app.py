"""
Main Streamlit Application.
Implements the multi-document policy chatbot interface with high-fidelity UI/UX
inspired by the Synthara project: dark slate HSL variables, glass-modern cards,
animated floating backgrounds, top metrics grids, document registries, and split-layout chats.
Custom CSS overrides are applied to standard Streamlit components to ensure a fully branded web app experience.
"""

import os
import logging
from typing import List, Dict, Any
import streamlit as st
import openai

# Import our custom modules
import config
from utils.document_loader import load_pdf
from utils.chunker import split_documents
from utils.retriever import add_documents_to_db, query_db, reset_db, get_collection
from utils.validator import validate_query, evaluate_faithfulness, evaluate_answer_relevancy

# Setup page config first (removes standard padding to match full-width dashboard)
st.set_page_config(
    page_title="Synthara Policy Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Logger initialization
logger = logging.getLogger("app")
logging.basicConfig(level=config.LOG_LEVEL)

# Tailwind-based glassmorphism, responsive dashboard CSS variables, animations, and Streamlit overrides
CUSTOM_CSS = """
<style>
    /* Google Fonts Loading */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&family=Space+Grotesk:wght@400;600;700&display=swap');

    /* CSS Variables & Global Dark Theme from Synthara project */
    :root {
        --background: #0B0F19;      /* Deep dark slate */
        --card: #131A26;            /* Dark card slate */
        --primary: #3B82F6;         /* Bright primary blue */
        --accent: #8B5CF6;          /* Violet accent */
        --border: #251C35;          /* Dark purple border */
        --text-muted: #94A3B8;      /* Slate muted gray */
        --glass-bg: rgba(255, 255, 255, 0.04);
        --glass-border: rgba(255, 255, 255, 0.08);
    }

    /* Font application across the app */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Shifting background gradient + Synthara SVG Grid Overlay */
    .stApp {
        background-color: var(--background) !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.12) 0px, transparent 50%),
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32' width='32' height='32' fill='none' stroke='rgb(255 255 255 / 0.015)'%3E%3Cpath d='M0 .5H31.5V32'/%3E%3C/svg%3E") !important;
        color: #F8FAFC !important;
    }

    /* Custom Webkit scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--background);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.2);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(148, 163, 184, 0.4);
    }

    /* Keyframe Animations */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    @keyframes pulseGlow {
        0%, 100% { opacity: 0.35; filter: blur(40px); }
        50% { opacity: 0.6; filter: blur(60px); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Floating Header Logo Animation */
    .animate-float-logo {
        animation: float 5s ease-in-out infinite;
        display: inline-block;
    }

    /* Background Floating Blurs */
    .glow-orb {
        position: absolute;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.25) 0%, rgba(59, 130, 246, 0) 70%);
        top: 20%;
        left: 30%;
        z-index: 0;
        pointer-events: none;
        animation: pulseGlow 6s ease-in-out infinite;
    }

    /* Form centering override for login page */
    div[data-testid="stForm"] {
        background: rgba(19, 26, 38, 0.65) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 20px !important;
        padding: 40px !important;
        max-width: 440px !important;
        margin: 15vh auto !important; /* Perfect vertical and horizontal centering */
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5) !important;
        animation: fadeInUp 0.8s cubic-bezier(0.22, 1, 0.36, 1);
    }
    
    .login-logo {
        text-align: center;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85em; /* Cleaner, sleeker font size */
        font-weight: 700;
        background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    .login-desc {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.88em;
        margin-top: 5px;
    }

    /* Header Logo Text Gradient */
    .text-gradient-logo {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Top Grid Dashboard Metrics */
    .stat-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin-bottom: 25px;
        animation: fadeInUp 0.8s ease;
    }
    .stat-card {
        background: rgba(19, 26, 38, 0.45) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        text-align: left;
        transition: transform 0.3s ease, border-color 0.3s ease !important;
    }
    .stat-card:hover {
        transform: translateY(-3px) !important;
        border-color: rgba(96, 165, 250, 0.3) !important;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.1) !important;
    }
    .stat-val {
        font-size: 2.1em;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 5px;
        font-family: 'Space Grotesk', sans-serif;
    }
    .stat-lbl {
        color: var(--text-muted);
        font-size: 0.8em;
        text-transform: uppercase;
        font-weight: bold;
        letter-spacing: 1px;
    }

    /* Glass Panels */
    .glass-panel {
        background: rgba(19, 26, 38, 0.45) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        margin-bottom: 20px;
    }

    /* Citation and evaluation badge styling */
    .citation-card {
        background: rgba(255, 255, 255, 0.01) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-left: 4px solid var(--primary) !important;
        padding: 12px 16px;
        margin: 10px 0px;
        border-radius: 0px 10px 10px 0px;
        font-size: 0.9em;
        color: #E2E8F0;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .citation-card:hover {
        transform: translateX(3px);
        border-color: rgba(59, 130, 246, 0.3) !important;
        background: rgba(255, 255, 255, 0.02) !important;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82em;
        font-weight: bold;
        color: white;
    }
    .badge-green { background-color: #10B981; }
    .badge-yellow { background-color: #F59E0B; }
    .badge-red { background-color: #EF4444; }

    /* Streamlit Components Visual Overrides (Glow effects, fonts) */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Override Chat Message Bubble Styles */
    div[data-testid="stChatMessage"] {
        background-color: rgba(19, 26, 38, 0.45) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 14px !important;
        padding: 18px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
    }
    div[data-testid="stChatMessageAudience-user"] {
        background-color: rgba(59, 130, 246, 0.08) !important;
        border-color: rgba(59, 130, 246, 0.2) !important;
    }

    /* Override Chat Input styling to feel like custom premium React portal */
    div[data-testid="stChatInput"] {
        border-radius: 16px !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
        background-color: rgba(19, 26, 38, 0.8) !important;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.15) !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: rgba(139, 92, 246, 0.6) !important;
        box-shadow: 0 0 25px rgba(139, 92, 246, 0.3) !important;
    }
    div[data-testid="stChatInput"] textarea {
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Override File Uploader border and background */
    div[data-testid="stFileUploader"] {
        border: 2px dashed rgba(139, 92, 246, 0.2) !important;
        border-radius: 14px !important;
        background-color: rgba(19, 26, 38, 0.2) !important;
        padding: 12px !important;
        transition: border-color 0.3s ease !important;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: var(--primary) !important;
    }

    /* Override Expanders */
    .st-expanderHeader {
        background-color: rgba(19, 26, 38, 0.5) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
    }
    .st-expanderContent {
        background-color: rgba(19, 26, 38, 0.1) !important;
        border-radius: 0px 0px 10px 10px !important;
        border: 1px solid var(--glass-border) !important;
        border-top: none !important;
    }

    /* Override Buttons */
    .stButton>button {
        border-radius: 10px !important;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(139, 92, 246, 0.4) !important;
    }
    .stButton>button:active {
        transform: translateY(0px) !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_session_state() -> None:
    """Initializes necessary Streamlit session state variables for chat history and login status."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "db_initialized" not in st.session_state:
        try:
            col = get_collection()
            count = col.count()
            st.session_state.db_initialized = (count > 0)
        except Exception:
            st.session_state.db_initialized = False
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False


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
    except Exception:
        return []


def call_llm(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Constructs the prompt context and calls the configured provider client."""
    client = config.get_openai_client()
    
    context_blocks: List[str] = []
    for idx, chunk in enumerate(retrieved_chunks):
        source = chunk["metadata"].get("source", "Unknown Document")
        page = chunk["metadata"].get("page", "?")
        context_blocks.append(f"Excerpt [{idx + 1}] (Source: {source}, Page {page}):\n{chunk['text']}")

    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an expert corporate policy assistant. Your goal is to answer the employee's question "
        "using ONLY the provided policy excerpts. If the information is not present in the excerpts, "
        "state that you cannot find the answer in the current policy documents. Do not hallucinate or "
        "use general knowledge.\n\n"
        "At the end of your response, list the citations matching the Excerpt bracket numbers (e.g. [1], [2]) "
        "that support your statements."
    )

    user_prompt = (
        f"Context Excerpts:\n{context_str}\n\n"
        f"Question:\n{question}\n\n"
        f"Grounded Response:"
    )

    try:
        logger.info(f"Calling LLM ({config.LLM_MODEL}) chat completion...")
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
        logger.error(f"Unexpected error in call_llm: {e}")
        return f"An unexpected error occurred: {str(e)}"


def run_pipeline(question: str) -> Dict[str, Any]:
    """Runs the full RAG query pipeline: search DB, prompt LLM, evaluate response."""
    retrieved_chunks = query_db(question, k=5)
    
    if not retrieved_chunks:
        return {
            "answer": "No relevant policy documents could be found in the knowledge base. Please upload documents first.",
            "citations": [],
            "evaluation": {
                "faithfulness": {"score": 0.0, "reasoning": "No context retrieved"},
                "relevancy": {"score": 0.0, "reasoning": "No context retrieved"}
            }
        }

    answer = call_llm(question, retrieved_chunks)
    contexts = [chunk["text"] for chunk in retrieved_chunks]
    
    faith_eval = evaluate_faithfulness(contexts, answer)
    rel_eval = evaluate_answer_relevancy(question, answer)

    return {
        "answer": answer,
        "citations": retrieved_chunks,
        "evaluation": {
            "faithfulness": faith_eval,
            "relevancy": rel_eval
        }
    }


def render_login_page() -> None:
    """Renders the glassmorphic login interface with floating mesh background."""
    st.markdown("<div class='glow-orb'></div>", unsafe_allow_html=True)
    
    # Render the login form wrapper directly, styled with auto-margins for absolute screen centering
    with st.form("login_form"):
        st.markdown(
            "<div style='text-align: center; margin-bottom: 25px;'>"
            "<div class='login-logo'>⚡ Synthara Portal</div>"
            "<div class='login-desc'>Enter portal credentials to access RAG Chatbot</div>"
            "</div>",
            unsafe_allow_html=True
        )
        username = st.text_input("Username", value="admin", placeholder="Enter admin username")
        password = st.text_input("Password", type="password", value="password", placeholder="Enter password")
        submit = st.form_submit_button("🚀 Enter Workspace", use_container_width=True)
        
        if submit:
            if username == "admin" and password == "password":
                st.session_state.logged_in = True
                st.success("Authorized successfully. Launching portal...")
                st.rerun()
            else:
                st.error("Invalid credentials. Hint: admin / password.")


def main() -> None:
    init_session_state()

    # Login Gate
    if not st.session_state.logged_in:
        render_login_page()
        return

    # Check api keys
    is_api_key_valid = config.check_keys()

    # Header portal branding
    st.markdown("<h1 style='margin-bottom:5px;'><span class='animate-float-logo'>🛡️</span> <span class='text-gradient-logo'>Synthara Policy Portal</span></h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:var(--text-muted); font-size:1.05em; margin-bottom:25px; font-weight: 500;'>Deploy local SentenceTransformers and OpenAI/Gemini routing to parse, index, and query handbooks.</div>", unsafe_allow_html=True)

    # Top Grid Dashboard Metrics
    total_chunks = 0
    document_list = get_indexed_documents()
    if st.session_state.db_initialized:
        try:
            total_chunks = get_collection().count()
        except Exception:
            pass

    st.markdown(
        f"<div class='stat-container'>"
        f"<div class='stat-card'><div class='stat-lbl'>📄 Document Registry</div><div class='stat-val'>{len(document_list)}</div></div>"
        f"<div class='stat-card'><div class='stat-lbl'>📦 Knowledge Nodes</div><div class='stat-val'>{total_chunks}</div></div>"
        f"<div class='stat-card'><div class='stat-lbl'>🛡️ Safety Check</div><div class='stat-val' style='color:#10B981;'>Grounded</div></div>"
        f"<div class='stat-card'><div class='stat-lbl'>⚙️ Active Model</div><div class='stat-val' style='font-size:1.55em; line-height:2.0; color:#8B5CF6; font-family:\"Space Grotesk\", sans-serif;'>{config.LLM_MODEL}</div></div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Split Workspace Layout (Left: Ingest & Registry, Right: Chat Console)
    col_workspace_left, col_workspace_right = st.columns([1.1, 2.0], gap="large")

    # LEFT PANEL: DOCUMENT MANAGER & CONFIGS
    with col_workspace_left:
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0px; font-family:\"Outfit\", sans-serif;'>📁 Ingest Documents</h3>", unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Select one or more policy handbooks",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
            label_visibility="collapsed"
        )

        if uploaded_files:
            if st.button("🚀 Process & Extract Text", use_container_width=True):
                if not is_api_key_valid:
                    st.error("Please configure a valid GEMINI_API_KEY or OPENAI_API_KEY to start extraction.")
                else:
                    with st.spinner("Processing documents..."):
                        temp_dir = "./temp_uploads"
                        try:
                            os.makedirs(temp_dir, exist_ok=True)
                            all_pages = []

                            for idx, uploaded_file in enumerate(uploaded_files):
                                temp_path = os.path.join(temp_dir, f"temp_{idx}.pdf")
                                with open(temp_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                
                                pages = load_pdf(temp_path, custom_filename=uploaded_file.name)
                                all_pages.extend(pages)
                                os.remove(temp_path)

                            os.rmdir(temp_dir)

                            if all_pages:
                                chunks = split_documents(all_pages)
                                add_documents_to_db(chunks)
                                st.session_state.db_initialized = True
                                st.success(f"Indexed {len(chunks)} chunks successfully!")
                                st.rerun()
                            else:
                                st.error("No extractable text found in files.")
                        except Exception as e:
                            st.error(f"Failed to process documents: {e}")
                            logger.error(f"Document Ingestion failed: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Document Registry List (Metadata Parser)
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0px; font-family:\"Outfit\", sans-serif;'>📖 Document Registry</h3>", unsafe_allow_html=True)
        
        if document_list:
            for doc in document_list:
                st.markdown(
                    f"<div style='background:rgba(255,255,255,0.015); border:1px solid rgba(255,255,255,0.04); padding:12px; border-radius:10px; margin-bottom:8px; font-size:0.9em;'>"
                    f"🟢 <strong>{doc['source']}</strong><br/>"
                    f"<span style='color:var(--text-muted); font-size:0.85em;'>Pages: {doc['pages']} | Nodes: {doc['chunks']}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown("<div style='color:var(--text-muted); font-size:0.9em;'>No documents loaded in the registry.</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Database Management / Reset
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0px; font-family:\"Outfit\", sans-serif;'>🧹 Portal Actions</h3>", unsafe_allow_html=True)
        
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("🗑️ Reset Database", use_container_width=True):
                try:
                    reset_db()
                    st.session_state.db_initialized = False
                    st.session_state.chat_history = []
                    st.success("Database wiped.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Reset failed: {e}")
        with col_act2:
            if st.button("🚪 Exit Workspace", use_container_width=True):
                st.session_state.logged_in = False
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT PANEL: CHAT CONSOLE
    with col_workspace_right:
        st.markdown("<div class='glass-panel' style='min-height:550px;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0px; font-family:\"Outfit\", sans-serif;'>💬 Interactive Chat Console</h3>", unsafe_allow_html=True)

        if not st.session_state.db_initialized:
            st.info("👋 Welcome to the Synthara Portal! To start asking questions, please upload corporate documents in the left panel and click **Process & Extract Text**.")
        else:
            # Display Chat History
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    
                    if msg["role"] == "assistant" and "citations" in msg:
                        # Citations
                        with st.expander("📖 View Retrieved Citations"):
                            for idx, chunk in enumerate(msg["citations"]):
                                source = chunk["metadata"].get("source", "Unknown Document")
                                page = chunk["metadata"].get("page", "?")
                                score = chunk.get("similarity", 0.0)
                                st.markdown(
                                    f"<div class='citation-card'>"
                                    f"<strong>Source {idx+1}:</strong> {source} (Page {page})<br/>"
                                    f"<strong>Similarity Match:</strong> {score * 100:.1f}%<br/>"
                                    f"<p style='margin-top: 5px; color:#94A3B8; font-style:italic;'>\"{chunk['text'][:300]}...\"</p>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                        # Evaluations
                        if "evaluations" in msg:
                            with st.expander("📊 Evaluation Metrics (LLM judge)"):
                                faith = msg["evaluations"].get("faithfulness", {})
                                relev = msg["evaluations"].get("relevancy", {})
                                f_score = faith.get("score", 0.0)
                                r_score = relev.get("score", 0.0)
                                
                                f_badge = "badge-green" if f_score >= 0.8 else ("badge-yellow" if f_score >= 0.5 else "badge-red")
                                r_badge = "badge-green" if r_score >= 0.8 else ("badge-yellow" if r_score >= 0.5 else "badge-red")
                                
                                st.markdown(
                                    f"#### Groundedness Score<br/>"
                                    f"<span class='metric-badge {f_badge}'>{f_score:.2f} / 1.00</span><br/>"
                                    f"<em>Reasoning:</em> {faith.get('reasoning', 'No reasoning supplied.')}",
                                    unsafe_allow_html=True
                                )
                                st.divider()
                                st.markdown(
                                    f"#### Answer Relevancy Score<br/>"
                                    f"<span class='metric-badge {r_badge}'>{r_score:.2f} / 1.00</span><br/>"
                                    f"<em>Reasoning:</em> {relev.get('reasoning', 'No reasoning supplied.')}",
                                    unsafe_allow_html=True
                                )

            # Chat Input Box
            query = st.chat_input("Ask a policy question...")

            if query:
                if not is_api_key_valid:
                    st.error("Please configure a valid API Key to ask questions.")
                    return

                try:
                    cleaned_query = validate_query(query)
                except ValueError as e:
                    st.error(str(e))
                    return

                with st.chat_message("user"):
                    st.write(cleaned_query)
                st.session_state.chat_history.append({"role": "user", "content": cleaned_query})

                with st.chat_message("assistant"):
                    with st.spinner("Retrieving facts and generating response..."):
                        try:
                            res = run_pipeline(cleaned_query)
                            st.write(res["answer"])

                            # Citations Accordion
                            with st.expander("📖 View Retrieved Citations"):
                                for idx, chunk in enumerate(res["citations"]):
                                    source = chunk["metadata"].get("source", "Unknown Document")
                                    page = chunk["metadata"].get("page", "?")
                                    score = chunk.get("similarity", 0.0)
                                    st.markdown(
                                        f"<div class='citation-card'>"
                                        f"<strong>Source {idx+1}:</strong> {source} (Page {page})<br/>"
                                        f"<strong>Similarity Match:</strong> {score * 100:.1f}%<br/>"
                                        f"<p style='margin-top: 5px; color:#94A3B8; font-style:italic;'>\"{chunk['text'][:300]}...\"</p>"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )

                            # Evaluations Accordion
                            with st.expander("📊 Evaluation Metrics (LLM judge)"):
                                faith = res["evaluation"].get("faithfulness", {})
                                relev = res["evaluation"].get("relevancy", {})
                                f_score = faith.get("score", 0.0)
                                r_score = relev.get("score", 0.0)
                                
                                f_badge = "badge-green" if f_score >= 0.8 else ("badge-yellow" if f_score >= 0.5 else "badge-red")
                                r_badge = "badge-green" if r_score >= 0.8 else ("badge-yellow" if r_score >= 0.5 else "badge-red")
                                
                                st.markdown(
                                    f"#### Groundedness Score<br/>"
                                    f"<span class='metric-badge {f_badge}'>{f_score:.2f} / 1.00</span><br/>"
                                    f"<em>Reasoning:</em> {faith.get('reasoning', 'No reasoning supplied.')}",
                                    unsafe_allow_html=True
                                )
                                st.divider()
                                st.markdown(
                                    f"#### Answer Relevancy Score<br/>"
                                    f"<span class='metric-badge {r_badge}'>{r_score:.2f} / 1.00</span><br/>"
                                    f"<em>Reasoning:</em> {relev.get('reasoning', 'No reasoning supplied.')}",
                                    unsafe_allow_html=True
                                )

                            # Download Results
                            download_text = f"Question: {cleaned_query}\n\nAnswer: {res['answer']}\n\nMetrics:\n- Groundedness: {f_score}\n- Relevancy: {r_score}"
                            st.download_button(
                                label="📥 Export Answer",
                                data=download_text,
                                file_name="policy_chat_response.txt",
                                mime="text/plain"
                            )

                            # Add to persistent chat state
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": res["answer"],
                                "citations": res["citations"],
                                "evaluations": res["evaluation"]
                            })
                            st.rerun()

                        except Exception as e:
                            logger.error(f"Error executing chat pipeline: {e}")
                            st.error(f"Failed to query knowledge base: {e}")

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
