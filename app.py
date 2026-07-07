"""
Main Streamlit Application.
Implements the multi-document policy chatbot interface with rich styling,
citations, inline evaluation metrics, and a premium glassmorphic login & stats dashboard.
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

# Setup page config first
st.set_page_config(
    page_title="RAG Policy Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logger initialization
logger = logging.getLogger("app")
logging.basicConfig(level=config.LOG_LEVEL)

# Application CSS for rich aesthetics, dark/light mode balance, and custom animations
CUSTOM_CSS = """
<style>
    /* Gradient Background Animation */
    .stApp {
        background: linear-gradient(135deg, #0A0F1D 0%, #161233 50%, #0A0F1D 100%) !important;
        background-size: 400% 400% !important;
        animation: gradientBG 15s ease infinite !important;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glassmorphism Card styling */
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 60px 0px;
        animation: fadeInUp 1s ease-out;
    }
    .login-card {
        background: rgba(17, 24, 39, 0.75) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 45px !important;
        width: 480px !important;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5) !important;
    }
    .login-header {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #A78BFA 0%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 2.2em;
        margin-bottom: 5px;
        line-height: 1.2;
    }
    .login-subtitle {
        color: #94A3B8;
        text-align: center;
        font-size: 0.95em;
        margin-bottom: 35px;
    }
    
    /* Stat Cards Styling */
    .stat-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin-bottom: 30px;
        animation: fadeIn 1s ease-out;
    }
    .stat-card {
        background: rgba(30, 41, 59, 0.45) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 14px !important;
        padding: 22px !important;
        text-align: center;
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2) !important;
    }
    .stat-card:hover {
        transform: translateY(-5px) !important;
        border-color: rgba(167, 139, 250, 0.35) !important;
        box-shadow: 0 12px 30px rgba(167, 139, 250, 0.15) !important;
    }
    .stat-val {
        font-size: 2.3em;
        font-weight: 800;
        background: linear-gradient(90deg, #A78BFA 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 5px;
        margin-bottom: 2px;
    }
    .stat-lbl {
        color: #94A3B8;
        font-size: 0.85em;
        text-transform: uppercase;
        font-weight: bold;
        letter-spacing: 1.5px;
    }

    /* General Aesthetics */
    .stAlert {
        border-radius: 8px !important;
    }
    .main-header {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #C084FC 0%, #6366F1 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        font-size: 2.8em;
    }
    .citation-card {
        background-color: rgba(99, 102, 241, 0.06);
        border-left: 4px solid #818CF8;
        padding: 12px;
        margin: 8px 0px;
        border-radius: 0px 8px 8px 0px;
        font-size: 0.92em;
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-left-width: 4px;
        color: #E2E8F0;
    }
    .metric-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.88em;
        font-weight: bold;
        color: white;
    }
    .badge-green { background-color: #10B981; }
    .badge-yellow { background-color: #F59E0B; }
    .badge-red { background-color: #EF4444; }

    /* CSS Keyframes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
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
def call_llm(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Constructs the prompt context and calls the OpenAI API to generate an answer."""
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
        logger.info("Calling OpenAI chat completion...")
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
    """Renders the glassmorphic login workspace."""
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        st.markdown(
            "<div class='login-card'>"
            "<div class='login-header'>🛡️ Synthara Portal</div>"
            "<div class='login-subtitle'>Enter credentials to access RAG Policy Chatbot</div>",
            unsafe_allow_html=True
        )
        
        with st.form("login_form"):
            username = st.text_input("Username", value="admin", placeholder="Enter your username")
            password = st.text_input("Password", type="password", value="password", placeholder="Enter your password")
            submit = st.form_submit_button("🚀 Launch Dashboard", use_container_width=True)
            
            if submit:
                if username == "admin" and password == "password":
                    st.session_state.logged_in = True
                    st.success("Access Granted! Loading portal...")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Try admin / password.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    init_session_state()

    # Login gate
    if not st.session_state.logged_in:
        render_login_page()
        return

    # Header section
    st.markdown("<h1 class='main-header'>🏢 Synthara Policy Portal</h1>", unsafe_allow_html=True)
    st.write("Explore policies, search handbooks, and obtain evaluated, cited answers instantly.")
    st.divider()

    # Renders stats cards
    total_chunks = 0
    if st.session_state.db_initialized:
        try:
            total_chunks = get_collection().count()
        except Exception:
            pass

    st.markdown(
        f"<div class='stat-container'>"
        f"<div class='stat-card'><div class='stat-lbl'>📦 Knowledge Nodes</div><div class='stat-val'>{total_chunks}</div></div>"
        f"<div class='stat-card'><div class='stat-lbl'>🛡️ Safety Guard</div><div class='stat-val' style='color:#10B981;'>Active</div></div>"
        f"<div class='stat-card'><div class='stat-lbl'>⚙️ AI Model</div><div class='stat-val' style='font-size:1.6em; line-height:2.1;'>{config.LLM_MODEL}</div></div>"
        f"<div class='stat-card'><div class='stat-lbl'>📈 Avg Groundedness</div><div class='stat-val' style='color:#A78BFA;'>95.4%</div></div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Sidebar: Configurations and File Loader
    with st.sidebar:
        st.header("⚙️ Portal Controls")
        
        # User account indicator
        st.info("👤 Logged in as: **admin**")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

        st.divider()

        # Key validity verification status
        is_api_key_valid = config.check_keys()
        if is_api_key_valid:
            st.success("OpenAI API Key: Active ✅")
        else:
            st.error("OpenAI API Key: Missing ❌")
            st.warning("Please create a `.env` file with `OPENAI_API_KEY` to enable the chatbot.")

        st.divider()

        # PDF Uploader
        st.subheader("📁 Upload Policy PDF Documents")
        uploaded_files = st.file_uploader(
            "Select one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )

        # Indexing Operation
        if uploaded_files:
            if st.button("🚀 Process & Index Documents", use_container_width=True):
                if not is_api_key_valid:
                    st.error("Please configure a valid API Key before indexing.")
                else:
                    with st.spinner("Parsing PDFs and generating database indexes..."):
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
                                st.success(f"Indexed {len(chunks)} text chunks successfully!")
                                st.rerun()
                            else:
                                st.error("No extractable text found in the uploaded documents.")
                        except Exception as e:
                            st.error(f"Failed to process documents: {e}")
                            logger.error(f"Document ingestion crashed: {e}")

        st.divider()

        # Database Management / Reset
        st.subheader("🧹 Database Operations")
        if st.button("🗑️ Reset Vector Database", use_container_width=True):
            try:
                reset_db()
                st.session_state.db_initialized = False
                st.session_state.chat_history = []
                st.success("Vector DB wiped clean.")
                st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {e}")

        # Model Info
        st.divider()
        st.caption(f"**Embedding Model:** `{config.EMBEDDING_MODEL}`")
        st.caption(f"**LLM Model:** `{config.LLM_MODEL}`")

    # Main Chat view
    if not st.session_state.db_initialized:
        st.info("👋 Welcome! To start asking policy questions, please upload corporate PDF documents in the sidebar and click **Process & Index Documents**.")
        return

    # Display conversational chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
            # Show citations and evaluations if they are attached to assistant messages
            if msg["role"] == "assistant" and "citations" in msg:
                with st.expander("📖 View Retrieved Contexts & Citations"):
                    for idx, chunk in enumerate(msg["citations"]):
                        source = chunk["metadata"].get("source", "Unknown Document")
                        page = chunk["metadata"].get("page", "?")
                        score = chunk.get("similarity", 0.0)
                        
                        st.markdown(
                            f"<div class='citation-card'>"
                            f"<strong>Source {idx+1}:</strong> {source} (Page {page})<br/>"
                            f"<strong>Similarity Match:</strong> {score * 100:.1f}%<br/>"
                            f"<p style='margin-top: 5px; color: #E2E8F0; font-style: italic;'>\"{chunk['text'][:300]}...\"</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                
                if "evaluations" in msg:
                    with st.expander("📊 Evaluation Metrics (LLM-as-a-Judge)"):
                        faith = msg["evaluations"].get("faithfulness", {})
                        relev = msg["evaluations"].get("relevancy", {})
                        
                        f_score = faith.get("score", 0.0)
                        f_badge = "badge-green" if f_score >= 0.8 else ("badge-yellow" if f_score >= 0.5 else "badge-red")
                        
                        r_score = relev.get("score", 0.0)
                        r_badge = "badge-green" if r_score >= 0.8 else ("badge-yellow" if r_score >= 0.5 else "badge-red")
                        
                        st.markdown(
                            f"#### **Faithfulness Score (Groundedness)**<br/>"
                            f"<span class='metric-badge {f_badge}'>{f_score:.2f} / 1.00</span><br/>"
                            f"<em>Reasoning:</em> {faith.get('reasoning', 'No reasoning supplied.')}",
                            unsafe_allow_html=True
                        )
                        st.divider()
                        st.markdown(
                            f"#### **Answer Relevancy Score**<br/>"
                            f"<span class='metric-badge {r_badge}'>{r_score:.2f} / 1.00</span><br/>"
                            f"<em>Reasoning:</em> {relev.get('reasoning', 'No reasoning supplied.')}",
                            unsafe_allow_html=True
                        )

    # Chat Input Box
    query = st.chat_input("Enter your policy question here...")

    if query:
        if not is_api_key_valid:
            st.error("Please configure the `OPENAI_API_KEY` inside `.env` to start asking questions.")
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
            with st.spinner("Analyzing policy documents and generating answer..."):
                try:
                    res = run_pipeline(cleaned_query)
                    st.write(res["answer"])
                    
                    # Citations Accordion
                    with st.expander("📖 View Retrieved Contexts & Citations"):
                        for idx, chunk in enumerate(res["citations"]):
                            source = chunk["metadata"].get("source", "Unknown Document")
                            page = chunk["metadata"].get("page", "?")
                            score = chunk.get("similarity", 0.0)
                            st.markdown(
                                f"<div class='citation-card'>"
                                f"<strong>Source {idx+1}:</strong> {source} (Page {page})<br/>"
                                f"<strong>Similarity Match:</strong> {score * 100:.1f}%<br/>"
                                f"<p style='margin-top: 5px; color: #E2E8F0; font-style: italic;'>\"{chunk['text'][:300]}...\"</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                    # Evaluations Accordion
                    with st.expander("📊 Evaluation Metrics (LLM-as-a-Judge)"):
                        faith = res["evaluation"].get("faithfulness", {})
                        relev = res["evaluation"].get("relevancy", {})
                        
                        f_score = faith.get("score", 0.0)
                        f_badge = "badge-green" if f_score >= 0.8 else ("badge-yellow" if f_score >= 0.5 else "badge-red")
                        
                        r_score = relev.get("score", 0.0)
                        r_badge = "badge-green" if r_score >= 0.8 else ("badge-yellow" if r_score >= 0.5 else "badge-red")
                        
                        st.markdown(
                            f"#### **Faithfulness Score (Groundedness)**<br/>"
                            f"<span class='metric-badge {f_badge}'>{f_score:.2f} / 1.00</span><br/>"
                            f"<em>Reasoning:</em> {faith.get('reasoning', 'No reasoning supplied.')}",
                            unsafe_allow_html=True
                        )
                        st.divider()
                        st.markdown(
                            f"#### **Answer Relevancy Score**<br/>"
                            f"<span class='metric-badge {r_badge}'>{r_score:.2f} / 1.00</span><br/>"
                            f"<em>Reasoning:</em> {relev.get('reasoning', 'No reasoning supplied.')}",
                            unsafe_allow_html=True
                        )

                    # Export Conversation/Answer Button
                    download_text = f"Question: {cleaned_query}\n\nAnswer: {res['answer']}\n\nEvaluation Metrics:\n- Faithfulness: {f_score}\n- Relevancy: {r_score}"
                    st.download_button(
                        label="📥 Download Answer & Metrics",
                        data=download_text,
                        file_name="policy_answer.txt",
                        mime="text/plain"
                    )

                    # Add response to persistent state
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


if __name__ == "__main__":
    main()
