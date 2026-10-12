"""Streamlit chat interface for the local CollegeNotesAI backend."""
from __future__ import annotations
import streamlit as st

from college_notes_ai.backend import RAGService
from college_notes_ai.config import LOGGER, RagError, Settings


@st.cache_resource(show_spinner=False)
def get_service() -> RAGService:
    """Cache expensive local resources for the lifetime of the Streamlit process."""
    return RAGService.load(Settings.from_env())

st.set_page_config(page_title="CollegeNotesAI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
:root { --cn-primary: #6558d3; }
.block-container {
    width: 100%;
    max-width: 1000px;
    padding: 2rem clamp(1rem, 3vw, 2rem) 3rem;
}
.cn-hero {
    padding: 1.3rem 1.5rem;
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 18px;
    margin-bottom: 1rem;
    overflow-wrap: anywhere;
}
.cn-title {
    font-size: 2rem;
    line-height: 1.2;
    font-weight: 750;
    letter-spacing: -.025em;
}
.cn-subtitle {
    color: var(--text-color);
    opacity: .72;
    margin-top: .25rem;
    margin-bottom: .75rem;
    line-height: 1.5;
}
.cn-chip {
    display: inline-block;
    border: 1px solid rgba(128,128,128,.28);
    border-radius: 999px;
    padding: .25rem .65rem;
    font-size: .8rem;
    margin: .15rem .35rem .15rem 0;
}
[data-testid="stChatMessage"] { border-radius: 14px; min-width: 0; }
[data-testid="stChatMessageContent"],
[data-testid="stMarkdownContainer"] {
    min-width: 0;
    overflow-wrap: anywhere;
}
[data-testid="stMarkdownContainer"] pre,
[data-testid="stMarkdownContainer"] table {
    max-width: 100%;
    overflow-x: auto;
}
@media (max-width: 768px) {
    [data-testid="stAppViewContainer"] .main .block-container {
        width: 100%;
        max-width: 100%;
        padding: 1rem .85rem calc(5rem + env(safe-area-inset-bottom));
    }
    .cn-hero { padding: 1rem; border-radius: 14px; margin-bottom: .85rem; }
    .cn-title { font-size: 1.55rem !important; line-height: 1.25; }
    .cn-subtitle { font-size: .94rem; margin-top: .45rem; }
    .cn-chip { font-size: .75rem; padding: .25rem .55rem; margin-right: .2rem; }
    h1, h2, h3 { overflow-wrap: anywhere; }
    [data-testid="stChatMessage"] { padding: .7rem .75rem; gap: .5rem; }
    [data-testid="stChatInput"] { padding-bottom: env(safe-area-inset-bottom); }
    [data-testid="stExpander"] { border-radius: 12px; }
    [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
        flex-wrap: wrap !important;
        gap: .45rem !important;
    }
    [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="column"] {
        flex: 1 1 100% !important;
        width: 100% !important;
        min-width: 0 !important;
    }
    [data-testid="stMetric"] {
        padding: .65rem .75rem;
        border: 1px solid rgba(128,128,128,.18);
        border-radius: 12px;
    }
    [data-testid="stMetricValue"] { font-size: 1.25rem; overflow-wrap: anywhere; }
    [data-testid="stSidebar"] { min-width: 0; }
    button, input, textarea { font-size: 16px !important; }
}
@media (max-width: 380px) {
    [data-testid="stAppViewContainer"] .main .block-container {
        padding-left: .65rem;
        padding-right: .65rem;
    }
    .cn-title { font-size: 1.4rem !important; }
    [data-testid="stChatMessage"] { padding: .55rem; }
}
</style>
<div class="cn-hero">
  <div class="cn-title">🎓 CollegeNotesAI</div>
  <p class="cn-subtitle">Your local AI study assistant — ask questions grounded in your college notes.</p>
  <span class="cn-chip">Runs locally</span><span class="cn-chip">Source pages</span><span class="cn-chip">No paid API</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Study assistant")
    st.caption("Private local workflow")
    st.markdown("**How it works**\n\n1. Your question is embedded locally.\n2. Relevant note chunks are retrieved from ChromaDB.\n3. Ollama generates an answer from those excerpts.")
    if st.button("＋ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Model settings are read from environment variables. The app does not upload your notes to a hosted AI API.")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("Ask your notes")
st.caption("Try: “What is an operating system?” or “Explain process scheduling in detail.”")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
            if message.get("sources"):
                pages = sorted({source["page"] for source in message["sources"]})
                with st.expander(f"📚 Sources from your notes · {len(pages)} page(s)"):
                    st.write("Pages: " + ", ".join(str(page) for page in pages))
                    for i, source in enumerate(message["sources"], start=1):
                        st.markdown(f"**Source {i} · Page {source['page']}**")
                        st.write(source["text"])
            if message.get("timings"):
                t = message["timings"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Retrieval", f"{t['retrieval_seconds']:.2f} s")
                c2.metric("Model response", f"{t['model_seconds']:.2f} s")
                c3.metric("Total", f"{t['total_seconds']:.2f} s")

question = st.chat_input("Ask a question about your notes…", max_chars=2000)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Preparing your study answer… first startup may take longer"):
                # Streamlit caches the initialized index and embedding model for this app process.
                service = get_service()
                answer, sources, timings = service.ask(question)
            st.markdown(answer)
            pages = sorted({source.page for source in sources})
            with st.expander(f"📚 Sources from your notes · {len(pages)} page(s)"):
                st.write("Pages: " + ", ".join(str(page) for page in pages))
                for i, source in enumerate(sources, start=1):
                    st.markdown(f"**Source {i} · Page {source.page}**")
                    st.write(source.text)
            c1, c2, c3 = st.columns(3)
            c1.metric("Retrieval", f"{timings.retrieval_seconds:.2f} s")
            c2.metric("Model response", f"{timings.model_seconds:.2f} s")
            c3.metric("Total", f"{timings.total_seconds:.2f} s")
            st.session_state.messages.append({
                "role": "assistant", "content": answer,
                "sources": [{"page": s.page, "text": s.text, "distance": s.distance} for s in sources],
                "timings": {"retrieval_seconds": timings.retrieval_seconds,
                            "model_seconds": timings.model_seconds,
                            "total_seconds": timings.total_seconds},
            })
        except RagError as exc:
            friendly = str(exc)
            st.error(friendly)
            st.session_state.messages.append({"role": "assistant", "content": friendly, "error": True})
        except Exception:
            # Keep diagnostic details in the terminal rather than exposing internals in the UI.
            LOGGER.exception("Unexpected error while answering a question.")
            friendly = "An unexpected error occurred. Check the terminal output and retry."
            st.error(friendly)
            st.session_state.messages.append({"role": "assistant", "content": friendly, "error": True})
