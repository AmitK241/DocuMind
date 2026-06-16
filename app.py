"""
app.py  ·  DocuMind — Streamlit Frontend
─────────────────────────────────────────────────────────────────────────────
DocuMind: Chat with your documents. Ask the world.
"""

# ── SQLite monkey-patch (MUST be first) ───────────────────────────────────────
# Streamlit Cloud's Ubuntu image ships with SQLite < 3.35 but chromadb requires
# >= 3.35.  pysqlite3-binary bundles a modern SQLite; we redirect the stdlib
# sqlite3 module to it before any chromadb/langchain import touches the DB.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # local dev: system sqlite3 is recent enough

import os
import shutil
import tempfile
from datetime import datetime

import streamlit as st

from rag import (
    ask,
    build_rag_chain,
    build_vectorstore,
    load_and_split,
    summarise_pdf,
)


# ── Secret resolution (local .env  →  Streamlit Cloud secrets) ────────────────
def _get_secret(key: str) -> str | None:
    """Read a secret: os.environ first, then st.secrets (Streamlit Cloud).
    Called at use-time so st.secrets is always fully initialised.
    """
    # 1. Local .env / shell environment
    value = os.getenv(key)
    if value:
        return value
    # 2. Streamlit Cloud secrets (TOML secrets dashboard)
    try:
        secrets = st.secrets
        if key in secrets:
            return secrets[key]
    except Exception:
        pass
    return None


def _tavily_key() -> str | None:
    return _get_secret("TAVILY_API_KEY")


def _groq_key() -> str | None:
    return _get_secret("GROQ_API_KEY")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Page background ───────────────────────────────────────────── */
.stApp { background: #060d1a; }

/* ── Sidebar ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090f1e 0%, #0d1526 100%);
    border-right: 1px solid #1e2d45;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Sidebar buttons ───────────────────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: #fff !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.82rem;
    padding: 0.45rem 0.75rem;
    transition: all 0.2s;
    box-shadow: 0 2px 8px #1d4ed840;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    box-shadow: 0 4px 16px #2563eb50;
    transform: translateY(-1px);
}

/* ── Main container ────────────────────────────────────────────── */
.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 860px;
}

/* ── Hero header ───────────────────────────────────────────────── */
.dm-hero {
    background: linear-gradient(135deg, #0f1f3d 0%, #0a1628 50%, #060d1a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.dm-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, #2563eb18, transparent 70%);
    border-radius: 50%;
}
.dm-hero-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    margin: 0 0 0.25rem 0;
}
.dm-hero-sub {
    color: #64748b;
    font-size: 0.88rem;
    font-weight: 400;
    margin: 0;
}
.dm-mode-pills {
    display: flex;
    gap: 8px;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.dm-pill {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
    letter-spacing: 0.04em;
}
.dm-pill-pdf   { background:#064e3b33; border:1px solid #10b981; color:#10b981; }
.dm-pill-web   { background:#78350f33; border:1px solid #f59e0b; color:#f59e0b; }
.dm-pill-ai    { background:#4c1d9533; border:1px solid #8b5cf6; color:#8b5cf6; }

/* ── No-PDF banner ─────────────────────────────────────────────── */
.dm-no-pdf {
    background: linear-gradient(135deg, #0f1f3d, #060d1a);
    border: 1px solid #1e3a5f;
    border-left: 4px solid #3b82f6;
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}
.dm-no-pdf b { color: #93c5fd; }
.dm-no-pdf span { color: #64748b; font-size: 0.875rem; }

/* ── PDF summary box ───────────────────────────────────────────── */
.dm-summary {
    background: linear-gradient(135deg, #0f1e38, #0a1525);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 1rem 1.25rem;
    color: #cbd5e1;
    font-size: 0.9rem;
    line-height: 1.7;
}

/* ── Chat messages ─────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border-radius: 16px;
    padding: 0.5rem 0;
    margin-bottom: 0.25rem;
}

/* ── Mode badges ───────────────────────────────────────────────── */
.badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 2px 9px;
    border-radius: 6px;
    margin-right: 6px;
    vertical-align: middle;
    text-transform: uppercase;
}
.badge-pdf     { background:#064e3b33; border:1px solid #10b981; color:#10b981; }
.badge-web     { background:#78350f33; border:1px solid #f59e0b; color:#f59e0b; }
.badge-general { background:#4c1d9533; border:1px solid #8b5cf6; color:#8b5cf6; }
.badge-error   { background:#7f1d1d33; border:1px solid #ef4444; color:#ef4444; }

/* ── Answer text ───────────────────────────────────────────────── */
.dm-answer {
    color: #e2e8f0;
    font-size: 0.93rem;
    line-height: 1.75;
    margin-top: 0.4rem;
}

/* ── Source pills ──────────────────────────────────────────────── */
.src-row { margin-top: 0.5rem; }
.src-pill {
    display: inline-block;
    background: #1e3a5f33;
    border: 1px solid #3b82f6;
    color: #93c5fd;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px 4px 2px 0;
}

/* ── Web source cards ──────────────────────────────────────────── */
.web-card {
    background: #0f1e38;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #f59e0b;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    margin: 5px 0;
}
.web-card a {
    color: #fbbf24 !important;
    font-weight: 600;
    font-size: 0.85rem;
    text-decoration: none;
}
.web-card a:hover { text-decoration: underline; }
.web-snip { color: #64748b; font-size: 0.78rem; margin-top: 3px; line-height: 1.5; }

/* ── Snippet block ─────────────────────────────────────────────── */
.snip-block {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #64748b;
    background: #0a1525;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.5rem 0.75rem;
    margin: 4px 0 8px 0;
    line-height: 1.6;
}

/* ── Warning card ──────────────────────────────────────────────── */
.warn-card {
    background: #7f1d1d22;
    border: 1px solid #ef444488;
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    font-size: 0.82rem;
    color: #fca5a5;
    margin-top: 6px;
}
.warn-card a { color: #f87171 !important; font-weight: 700; }

/* ── Sidebar logo ──────────────────────────────────────────────── */
.dm-logo {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}
.dm-tagline { font-size: 0.72rem; color: #334155 !important; margin-top: 2px; }

/* ── Status indicators ─────────────────────────────────────────── */
.status-ok  { font-size: 0.75rem; color: #34d399 !important; }
.status-err { font-size: 0.75rem; color: #f87171 !important; }

/* ── Stats ─────────────────────────────────────────────────────── */
.dm-stats { font-size: 0.75rem; color: #334155 !important; padding: 2px 0; }

/* ── Divider ───────────────────────────────────────────────────── */
hr { border-color: #1e2d45 !important; }

/* ── Expander ──────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0a1525 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
}

/* ── Upload area ───────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0a1525;
    border: 1px dashed #1e3a5f;
    border-radius: 12px;
    padding: 0.5rem;
}

/* ── Radio ─────────────────────────────────────────────────────── */
[data-testid="stRadio"] label { font-size: 0.83rem !important; }

/* ── Chat input ────────────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    background: #0a1525 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px #3b82f620 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "pdfs":          {},
    "active_pdf":    None,
    "messages":      [],
    "total_queries": 0,
    "show_snippets": True,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ────────────────────────────────────────────────────────────────────
def _col_name(name: str) -> str:
    import re as _re
    base = os.path.splitext(name)[0]
    # Replace invalid chars with underscore
    safe = _re.sub(r"[^a-zA-Z0-9_-]", "_", base)
    # Collapse multiple consecutive underscores/hyphens
    safe = _re.sub(r"[_-]{2,}", "_", safe)
    # Strip leading/trailing underscores and hyphens (must start+end alphanumeric)
    safe = safe.strip("_-")
    # Truncate to 63 chars then re-strip
    safe = safe[:63].strip("_-")
    # Must be at least 3 chars
    if len(safe) < 3:
        safe = (safe + "pdf")[:63]
    return safe or "docpdf"

def _export_chat() -> str:
    lines = [f"DocuMind Chat Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    if st.session_state.active_pdf:
        lines.append(f"Document: {st.session_state.active_pdf}\n")
    lines.append("=" * 60 + "\n")
    for msg in st.session_state.messages:
        role = "You" if msg["role"] == "user" else "DocuMind"
        lines.append(f"[{role}]\n{msg['content']}\n")
        if msg.get("sources"):
            lines.append(f"Sources: {', '.join(msg['sources'])}\n")
        lines.append("-" * 40 + "\n")
    return "\n".join(lines)

def _badge(mode: str) -> str:
    return {
        "pdf":        "<span class='badge badge-pdf'>📄 PDF</span>",
        "web":        "<span class='badge badge-web'>🔍 Web</span>",
        "general":    "<span class='badge badge-general'>🤖 AI</span>",
        "web_no_key": "<span class='badge badge-error'>⚠️ No Key</span>",
        "web_failed": "<span class='badge badge-error'>⚠️ Web Failed</span>",
    }.get(mode, "<span class='badge badge-general'>🤖 AI</span>")

def _render_sources(sources, snippets, mode, web_sources):
    if not sources:
        return
    if mode == "pdf":
        pills = "".join(f"<span class='src-pill'>{s}</span>" for s in sources)
        st.markdown(f"<div class='src-row'>{pills}</div>", unsafe_allow_html=True)
    if st.session_state.show_snippets and snippets:
        label = "📄 Source chunks" if mode == "pdf" else "🌐 Web sources"
        with st.expander(label, expanded=False):
            for src, snip in zip(sources, snippets):
                if mode == "web":
                    title = next((r.get("title", src) for r in web_sources if r.get("url") == src), src)
                    st.markdown(
                        f"<div class='web-card'><a href='{src}' target='_blank'>🔗 {title}</a>"
                        f"<div class='web-snip'>{snip}</div></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"<div class='snip-block'><b style='color:#93c5fd'>{src}</b><br>{snip}</div>",
                                unsafe_allow_html=True)

def _render_warning(mode: str):
    if mode == "web_no_key":
        st.markdown(
            "<div class='warn-card'>⚠️ This needs web search but <code>TAVILY_API_KEY</code> "
            "is missing. Get a free key at "
            "<a href='https://app.tavily.com' target='_blank'>app.tavily.com</a> "
            "and add it to your <code>.env</code> file or Streamlit Cloud Secrets.</div>",
            unsafe_allow_html=True,
        )
    elif mode == "web_failed":
        st.markdown(
            "<div class='warn-card'>⚠️ Web search returned no results. "
            "Try rephrasing or check your Tavily key.</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div class='dm-logo'>🧠 DocuMind</div>", unsafe_allow_html=True)
    st.markdown("<div class='dm-tagline'>Chat with docs · Ask the world</div>", unsafe_allow_html=True)
    st.divider()

    # ── Upload ─────────────────────────────────────────────────────────────
    st.markdown("**📂 Upload PDF**")
    uploaded = st.file_uploader("PDF file", type=["pdf"], label_visibility="collapsed")

    if uploaded:
        if uploaded.name not in st.session_state.pdfs:
            with st.spinner("Indexing…"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                chunks      = load_and_split(tmp_path)
                vectorstore = build_vectorstore(chunks, collection_name=_col_name(uploaded.name))
                chain       = build_rag_chain(vectorstore)
                summary     = summarise_pdf(chunks)
                os.unlink(tmp_path)

                st.session_state.pdfs[uploaded.name] = {
                    "chain": chain, "vectorstore": vectorstore,
                    "chunks": len(chunks), "summary": summary,
                }
            st.success(f"✅ {len(chunks)} chunks indexed")

        if st.session_state.active_pdf != uploaded.name:
            st.session_state.active_pdf = uploaded.name
            st.session_state.messages   = []

    # ── PDF selector ───────────────────────────────────────────────────────
    if st.session_state.pdfs:
        st.divider()
        st.markdown("**📚 Loaded PDFs**")
        options = list(st.session_state.pdfs.keys())
        chosen  = st.radio("", options,
                           index=options.index(st.session_state.active_pdf)
                                 if st.session_state.active_pdf in options else 0,
                           label_visibility="collapsed")
        if chosen != st.session_state.active_pdf:
            st.session_state.active_pdf = chosen
            st.session_state.messages   = []
            st.rerun()

        info = st.session_state.pdfs.get(st.session_state.active_pdf, {})
        st.caption(f"Chunks: **{info.get('chunks','—')}**")

    st.divider()

    # ── Settings ───────────────────────────────────────────────────────────
    st.markdown("**⚙️ Settings**")
    st.session_state.show_snippets = st.toggle("Show source snippets", value=st.session_state.show_snippets)

    st.divider()

    # ── Actions ────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            # LCEL chains are stateless callables — history is passed explicitly
            # on every call, so clearing session_state.messages is sufficient.
            st.rerun()
    with c2:
        if st.button("❌ Remove", use_container_width=True):
            active = st.session_state.active_pdf
            if active and active in st.session_state.pdfs:
                del st.session_state.pdfs[active]
                st.session_state.active_pdf = (
                    list(st.session_state.pdfs.keys())[-1]
                    if st.session_state.pdfs else None
                )
                st.session_state.messages = []
                st.rerun()

    if st.session_state.messages:
        st.download_button(
            "💾 Export Chat",
            data=_export_chat(),
            file_name=f"documind_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.divider()

    # ── Status ─────────────────────────────────────────────────────────────
    st.markdown(
        f"<div class='dm-stats'>Queries this session: <b>{st.session_state.total_queries}</b></div>",
        unsafe_allow_html=True,
    )
    if _tavily_key():
        st.markdown("<div class='status-ok'>🟢 Web search active</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='status-err'>🔴 No Tavily key · web search off<br>"
            "<small>Add TAVILY_API_KEY to .env or Streamlit Secrets</small></div>",
            unsafe_allow_html=True,
        )

    # ── API key debug (no key values shown) ────────────────────────────────────
    with st.expander("🔑 API Key Status", expanded=False):
        st.write("Groq:",   "✅ Connected" if _groq_key()   else "❌ Missing")
        st.write("Tavily:", "✅ Enabled"   if _tavily_key() else "❌ Disabled")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='dm-hero'>
  <div class='dm-hero-title'>🧠 DocuMind</div>
  <p class='dm-hero-sub'>Chat with your documents · Search the web · Ask anything — powered by Groq &amp; LangChain</p>
  <div class='dm-mode-pills'>
    <span class='dm-pill dm-pill-pdf'>📄 PDF RAG</span>
    <span class='dm-pill dm-pill-web'>🔍 Live Web Search</span>
    <span class='dm-pill dm-pill-ai'>🤖 General AI</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Resolve active PDF ─────────────────────────────────────────────────────────
active_pdf  = st.session_state.active_pdf
active_data = st.session_state.pdfs.get(active_pdf, {}) if active_pdf else {}
chain       = active_data.get("chain")
vectorstore = active_data.get("vectorstore")

# ── No-PDF banner OR summary ───────────────────────────────────────────────────
if not active_pdf:
    st.markdown("""
    <div class='dm-no-pdf'>
      <b>💡 No document loaded</b><br>
      <span>You can still ask <b>any question</b> or get <b>live web search results</b>.
      Upload a PDF from the sidebar to also chat with documents.</span>
    </div>
    """, unsafe_allow_html=True)
else:
    with st.expander(f"📋 Document Summary — {active_pdf}", expanded=False):
        st.markdown(f"<div class='dm-summary'>{active_data.get('summary','')}</div>",
                    unsafe_allow_html=True)

st.divider()

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🧠"):
        if msg["role"] == "assistant":
            mode = msg.get("mode", "general")
            st.markdown(
                f"{_badge(mode)}<span class='dm-answer'>{msg['content']}</span>",
                unsafe_allow_html=True,
            )
            _render_warning(mode)
            _render_sources(msg.get("sources", []), msg.get("snippets", []),
                            mode, msg.get("web_sources", []))
        else:
            st.markdown(f"<span style='color:#e2e8f0'>{msg['content']}</span>",
                        unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────────
placeholder = (
    "Ask about the document, search the web, or ask anything…"
    if active_pdf else
    "Ask anything — web search & AI work without a PDF too…"
)

if prompt := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"<span style='color:#e2e8f0'>{prompt}</span>", unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("DocuMind is thinking…"):
            result = ask(
                chain=chain,
                question=prompt,
                chat_history=st.session_state.messages[:-1],
                vectorstore=vectorstore,
            )

        mode = result["mode"]
        st.markdown(
            f"{_badge(mode)}<span class='dm-answer'>{result['answer']}</span>",
            unsafe_allow_html=True,
        )
        _render_warning(mode)
        _render_sources(result.get("sources", []), result.get("snippets", []),
                        mode, result.get("web_sources", []))

    st.session_state.messages.append({
        "role":        "assistant",
        "content":     result["answer"],
        "sources":     result.get("sources", []),
        "snippets":    result.get("snippets", []),
        "mode":        result["mode"],
        "web_sources": result.get("web_sources", []),
    })
    st.session_state.total_queries += 1