"""
Home page for Streamlit interface.
"""

import logging
import uuid
import streamlit as st

st.set_page_config(page_title="AI Document Assistant", page_icon="🤖", layout="centered")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log",
    filemode="a",
)
logger = logging.getLogger(__name__)

st.markdown("""
<style>
    [data-testid="stSidebarNav"] { display: none; }

    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
    }
    [data-testid="stMain"] {
        background-color: #ffffff;
    }
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        max-width: 100% !important;
    }

    /* ── Top black hero section ── */
    .hero {
        background-color: #0a0a0a;
        padding: 80px 40px 60px;
        text-align: center;
        width: 100%;
    }
    .hero-icon {
        font-size: 72px;
        display: block;
        margin-bottom: 20px;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50%       { transform: translateY(-10px); }
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 14px;
        line-height: 1.2;
    }
    .hero-title span {
        color: #e2e2e2;
        font-weight: 300;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #888888;
        max-width: 500px;
        margin: 0 auto;
    }

    /* ── White bottom section ── */
    .bottom-section {
        background-color: #ffffff;
        padding: 48px 40px 40px;
    }

    /* ── Feature cards ── */
    .cards-row {
        display: flex;
        gap: 16px;
        margin-bottom: 36px;
        justify-content: center;
    }
    .card {
        flex: 1;
        background: #f8f8f8;
        border: 1px solid #e5e5e5;
        border-radius: 16px;
        padding: 24px 16px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }
    .card-icon { font-size: 36px; margin-bottom: 12px; }
    .card-title {
        color: #0a0a0a;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .card-desc { color: #888888; font-size: 0.82rem; line-height: 1.5; }

    /* ── Input ── */
    .stTextInput > div > div > input {
        background: #f8f8f8 !important;
        border: 1.5px solid #e0e0e0 !important;
        border-radius: 12px !important;
        color: #0a0a0a !important;
        padding: 14px 18px !important;
        font-size: 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #0a0a0a !important;
        box-shadow: none !important;
    }
    .stTextInput > div > div > input::placeholder { color: #aaaaaa !important; }

    /* ── Button ── */
    .stButton > button {
        background: #0a0a0a !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        transition: background 0.2s !important;
        margin-top: 6px !important;
    }
    .stButton > button:hover {
        background: #333333 !important;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #bbbbbb;
        font-size: 0.78rem;
        margin-top: 36px;
        padding-bottom: 16px;
        border-top: 1px solid #f0f0f0;
        padding-top: 20px;
    }
    .footer span {
        display: inline-block;
        margin: 0 6px;
        background: #f0f0f0;
        border-radius: 20px;
        padding: 2px 10px;
        color: #666;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Black Hero Section ─────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-icon">🤖</span>
    <div class="hero-title">AI Document <span>Assistant</span></div>
    <div class="hero-subtitle">
        Upload your PDFs &amp; text files — ask anything,
        get instant intelligent answers
    </div>
</div>
""", unsafe_allow_html=True)

# ── White Bottom Section ───────────────────────────────
st.markdown("""
<div class="bottom-section">
    <div class="cards-row">
        <div class="card">
            <div class="card-icon">📄</div>
            <div class="card-title">Document Upload</div>
            <div class="card-desc">PDF &amp; TXT support with smart chunking</div>
        </div>
        <div class="card">
            <div class="card-icon">🧠</div>
            <div class="card-title">Adaptive RAG</div>
            <div class="card-desc">Routes queries to the best knowledge source</div>
        </div>
        <div class="card">
            <div class="card-icon">💬</div>
            <div class="card-title">Chat Memory</div>
            <div class="card-desc">Remembers your full conversation history</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Form ──────────────────────────────────────────────
username = st.text_input(
    "Your name",
    placeholder="Enter your name (optional)...",
    label_visibility="collapsed",
)

if st.button(" Start Chatting", use_container_width=True):
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
        logger.info(f"New session: {st.session_state['session_id']}")
    st.session_state["username"] = username.strip() or "Guest"
    st.switch_page("pages/chat.py")

# ── Footer ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Powered by
    <span>Groq</span>
    <span>LangGraph</span>
    <span>Qdrant</span>
    <span>HuggingFace</span>
</div>
""", unsafe_allow_html=True)

# ── Debug Logs ─────────────────────────────────────────
with st.expander("📜 Debug Logs"):
    try:
        with open("app.log", "r") as f:
            content = f.read()
            st.text(content if content else "No logs yet.")
    except FileNotFoundError:
        st.warning("Log file not found yet.")