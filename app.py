"""
app.py
Streamlit UI for the Agentic Documentation Generator.
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Doc Generator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

    .stApp { background: #0a0f1e; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }

    /* Main area */
    .main .block-container {
        padding: 2rem 2.5rem;
        max-width: 1100px;
    }

    /* Cards */
    .doc-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Titles */
    h1, h2, h3 { color: #f1f5f9 !important; }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
    }
    .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(99,102,241,0.5) !important;
    }

    /* Status box */
    .status-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #94a3b8;
    }

    /* Agent badges */
    .agent-badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 0.25rem;
    }
    .badge-analyzer { background: #1e3a5f; color: #7dd3fc; }
    .badge-generator { background: #312e81; color: #a5b4fc; }
    .badge-examples { background: #1c3a2c; color: #86efac; }
    .badge-validator { background: #422006; color: #fdba74; }

    /* Input styling */
    .stTextInput input, .stSelectbox select {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
    }

    /* Radio */
    .stRadio label { color: #cbd5e1 !important; }

    /* Hero section */
    .hero-section {
        text-align: center;
        padding: 3rem 1rem 2rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.04em;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Feature pills */
    .pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 9999px;
        font-size: 0.8rem;
        color: #94a3b8;
        margin: 0.2rem;
    }

    /* Tab */
    .stTabs [data-baseweb="tab"] { color: #94a3b8 !important; }
    .stTabs [aria-selected="true"] { color: #6366f1 !important; border-bottom-color: #6366f1 !important; }

    /* Alert boxes */
    .success-banner {
        background: linear-gradient(135deg, #065f46, #064e3b);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: #6ee7b7;
        font-weight: 500;
        margin: 1rem 0;
    }
    .error-banner {
        background: linear-gradient(135deg, #7f1d1d, #450a0a);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: #fca5a5;
        margin: 1rem 0;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def get_api_key() -> Optional[str]:
    """Get Groq API key from env or session state."""
    return st.session_state.get("groq_api_key") or os.getenv("GROQ_API_KEY")


def render_status_log(messages: list):
    """Render a scrollable status log."""
    if not messages:
        return
    content = "\n".join(f"▶ {m}" for m in messages)
    st.markdown(f'<div class="status-box">{content}</div>', unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")

        # API Key
        api_key_env = os.getenv("GROQ_API_KEY", "")
        if api_key_env:
            st.success("✅ GROQ_API_KEY loaded from .env")
        else:
            key_input = st.text_input(
                "Groq API Key",
                type="password",
                placeholder="gsk_...",
                help="Get your key at console.groq.com",
            )
            if key_input:
                st.session_state["groq_api_key"] = key_input

        st.markdown("---")
        st.markdown("### 🤖 Agents")
        st.markdown("""
        <span class="agent-badge badge-analyzer">Analyzer</span> Understands code<br><br>
        <span class="agent-badge badge-generator">DocGen</span> Writes docs<br><br>
        <span class="agent-badge badge-examples">Examples</span> Creates examples<br><br>
        <span class="agent-badge badge-validator">Validator</span> Improves quality
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📊 Output Formats")
        st.markdown("""
        - 📝 **Markdown** — Raw `.md` file
        - 🌐 **HTML** — Interactive docs site
        - 🗂️ **JSON** — Structured data
        """)

        st.markdown("---")
        st.caption("Agentic Documentation Generator\nPowered by Groq LLM + RAG")


# ── Main UI ───────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    # Hero
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">⚡ Agentic Doc Generator</div>
        <div class="hero-subtitle">Generate professional documentation from any codebase using AI agents + RAG</div>
        <div>
            <span class="pill">🔍 RAG-enhanced</span>
            <span class="pill">🤖 Multi-agent</span>
            <span class="pill">⚡ Groq LLM</span>
            <span class="pill">🌐 Interactive HTML</span>
            <span class="pill">📦 ZIP / Files / Git</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input section ─────────────────────────────────────────────────────────
    st.markdown("## 📥 Input Source")

    input_type = st.radio(
        "Select input method:",
        ["📦 ZIP File", "📄 Individual Files", "🔗 Git Repository"],
        horizontal=True,
        label_visibility="collapsed",
    )

    files = None
    project_name = "my_project"
    ingest_error = None

    # ── ZIP Upload ────────────────────────────────────────────────────────────
    if input_type == "📦 ZIP File":
        st.markdown('<div class="doc-card">', unsafe_allow_html=True)
        st.markdown("### 📦 Upload ZIP Archive")
        st.caption("Upload a .zip file containing your project source code.")
        zip_file = st.file_uploader(
            "Choose a ZIP file",
            type=["zip"],
            label_visibility="collapsed",
        )
        if zip_file:
            project_name = Path(zip_file.name).stem
            st.info(f"📦 Ready: **{zip_file.name}** ({zip_file.size:,} bytes)")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Individual Files ──────────────────────────────────────────────────────
    elif input_type == "📄 Individual Files":
        st.markdown('<div class="doc-card">', unsafe_allow_html=True)
        st.markdown("### 📄 Upload Source Files")
        st.caption("Upload one or more source code files directly.")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            st.info(f"📄 {len(uploaded_files)} file(s) selected: {', '.join(f.name for f in uploaded_files[:5])}")
        project_name = st.text_input("Project name", value="my_project", placeholder="my_project")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Git Repository ────────────────────────────────────────────────────────
    elif input_type == "🔗 Git Repository":
        st.markdown('<div class="doc-card">', unsafe_allow_html=True)
        st.markdown("### 🔗 Clone Git Repository")

        col1, col2 = st.columns([3, 1])
        with col1:
            repo_url = st.text_input(
                "Repository URL",
                placeholder="https://github.com/owner/repository",
            )
        with col2:
            branch = st.text_input("Branch (optional)", placeholder="main")

        git_token = st.text_input(
            "🔒 Git Token (for private repos)",
            type="password",
            placeholder="ghp_... (optional)",
            help="GitHub Personal Access Token for private repositories",
        )

        if not git_token:
            git_token = os.getenv("GIT_TOKEN")

        if repo_url:
            st.info(f"🔗 Repository: `{repo_url}`" + (f" branch: `{branch}`" if branch else ""))
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Output format ──────────────────────────────────────────────────────────
    st.markdown("## 📤 Output Format")
    st.markdown('<div class="doc-card">', unsafe_allow_html=True)

    fmt_col1, fmt_col2, fmt_col3 = st.columns(3)
    with fmt_col1:
        md_selected = st.button("📝 Markdown", use_container_width=True)
    with fmt_col2:
        html_selected = st.button("🌐 HTML (Interactive)", use_container_width=True)
    with fmt_col3:
        json_selected = st.button("🗂️ JSON", use_container_width=True)

    if "output_format" not in st.session_state:
        st.session_state["output_format"] = "markdown"

    if md_selected:
        st.session_state["output_format"] = "markdown"
    elif html_selected:
        st.session_state["output_format"] = "html"
    elif json_selected:
        st.session_state["output_format"] = "json"

    fmt = st.session_state["output_format"]
    fmt_labels = {"markdown": "📝 Markdown", "html": "🌐 HTML (Interactive)", "json": "🗂️ JSON"}
    st.info(f"Selected format: **{fmt_labels[fmt]}**")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Generate button ────────────────────────────────────────────────────────
    st.markdown("## 🚀 Generate")

    api_key = get_api_key()
    if not api_key:
        st.warning("⚠️ No Groq API key found. Add GROQ_API_KEY to your .env file or enter it in the sidebar.")

    generate_btn = st.button(
        "⚡ Generate Documentation",
        use_container_width=True,
        disabled=not api_key,
    )

    # ── Pipeline execution ─────────────────────────────────────────────────────
    if generate_btn:
        # Validate API key format
        if api_key and not api_key.startswith("gsk_"):
            st.warning("⚠️ Groq API keys typically start with 'gsk_'. Proceeding anyway...")

        status_messages = []
        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        def update_progress(msg: str, pct: int):
            status_messages.append(msg)
            progress_bar.progress(pct / 100)
            with status_placeholder.container():
                render_status_log(status_messages[-8:])

        try:
            from ingestion.ingestor import Ingestor
            from pipeline import DocumentationPipeline

            ingestor = Ingestor()

            # ── Ingest ───────────────────────────────────────────────────────
            update_progress("Ingesting source files...", 2)

            if input_type == "📦 ZIP File":
                if not zip_file:
                    st.error("❌ Please upload a ZIP file first.")
                    st.stop()
                files, project_name = ingestor.ingest_zip(zip_file.read())

            elif input_type == "📄 Individual Files":
                if not uploaded_files:
                    st.error("❌ Please upload at least one file.")
                    st.stop()
                files, _ = ingestor.ingest_files(uploaded_files)
                if not project_name.strip():
                    project_name = "my_project"

            elif input_type == "🔗 Git Repository":
                if not repo_url:
                    st.error("❌ Please enter a repository URL.")
                    st.stop()
                files, project_name = ingestor.ingest_git(
                    repo_url=repo_url,
                    branch=branch if branch else None,
                    token=git_token if git_token else None,
                )

            update_progress(f"✅ Ingested {len(files)} files from '{project_name}'", 5)

            # ── Pipeline ─────────────────────────────────────────────────────
            pipeline = DocumentationPipeline(groq_api_key=api_key)

            result = pipeline.run(
                files=files,
                project_name=project_name,
                output_format=fmt,
                progress_callback=update_progress,
            )

            # ── Store result ──────────────────────────────────────────────────
            st.session_state["generated_doc"] = result
            st.session_state["generated_fmt"] = fmt
            st.session_state["generated_project"] = project_name

            ingestor.cleanup()

            # ── Success ───────────────────────────────────────────────────────
            progress_bar.progress(1.0)
            st.markdown(
                '<div class="success-banner">✅ Documentation generated successfully!</div>',
                unsafe_allow_html=True,
            )

        except ValueError as e:
            st.markdown(f'<div class="error-banner">❌ Input Error: {e}</div>', unsafe_allow_html=True)
            logger.error(f"Input error: {e}")
        except RuntimeError as e:
            if "authentication" in str(e).lower() or "401" in str(e) or "403" in str(e):
                st.markdown(
                    '<div class="error-banner">❌ Groq API authentication failed. Check your GROQ_API_KEY.</div>',
                    unsafe_allow_html=True,
                )
            elif "rate" in str(e).lower():
                st.markdown(
                    '<div class="error-banner">⚠️ Groq rate limit hit. Please wait a moment and try again.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f'<div class="error-banner">❌ LLM Error: {e}</div>', unsafe_allow_html=True)
            logger.error(f"Runtime error: {e}\n{traceback.format_exc()}")
        except Exception as e:
            st.markdown(
                f'<div class="error-banner">❌ Unexpected error: {e}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("🔍 Error details"):
                st.code(traceback.format_exc())
            logger.error(f"Unexpected: {e}\n{traceback.format_exc()}")

    # ── Result display ─────────────────────────────────────────────────────────
    if "generated_doc" in st.session_state:
        doc = st.session_state["generated_doc"]
        doc_fmt = st.session_state["generated_fmt"]
        doc_project = st.session_state["generated_project"]

        st.markdown("## 📖 Generated Documentation")

        # Download button
        ext_map = {"markdown": "md", "html": "html", "json": "json"}
        mime_map = {
            "markdown": "text/markdown",
            "html": "text/html",
            "json": "application/json",
        }
        filename = f"{doc_project}_docs.{ext_map[doc_fmt]}"

        col_dl, col_clear = st.columns([3, 1])
        with col_dl:
            st.download_button(
                label=f"⬇️ Download {ext_map[doc_fmt].upper()}",
                data=doc.encode("utf-8"),
                file_name=filename,
                mime=mime_map[doc_fmt],
                use_container_width=True,
            )
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                del st.session_state["generated_doc"]
                del st.session_state["generated_fmt"]
                del st.session_state["generated_project"]
                st.rerun()

        # Preview tabs
        tab1, tab2 = st.tabs(["👁️ Preview", "📋 Raw Source"])

        with tab1:
            if doc_fmt == "html":
                st.markdown("*Rendering interactive HTML preview:*")
                st.components.v1.html(doc, height=700, scrolling=True)
            elif doc_fmt == "markdown":
                st.markdown(doc)
            else:
                st.json(doc)

        with tab2:
            st.code(doc, language=doc_fmt if doc_fmt != "markdown" else "markdown", line_numbers=True)

        st.caption(f"📄 {len(doc):,} characters | {len(doc.splitlines()):,} lines | Format: {doc_fmt.upper()}")


if __name__ == "__main__":
    main()