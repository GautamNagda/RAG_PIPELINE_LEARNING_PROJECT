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
.block-container { max-width: 1000px; padding-top: 2rem; padding-bottom: 3rem; }
.cn-hero { padding: 1.3rem 1.5rem; border: 1px solid rgba(128,128,128,.22); border-radius: 18px; margin-bottom: 1rem; }
.cn-subtitle { color: var(--text-color); opacity: .72; margin-top: -.4rem; }
.cn-chip { display: inline-block; border: 1px solid rgba(128,128,128,.28); border-radius: 999px; padding: .25rem .65rem; font-size: .8rem; margin-right: .35rem; }
[data-testid="stChatMessage"] { border-radius: 14px; }
</style>
<div class="cn-hero">
  <div style="font-size:2rem;font-weight:750;">🎓 CollegeNotesAI</div>
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
