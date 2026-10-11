# CollegeNotesAI — Milestone 6 (modular Streamlit UI)

## What changed

- `app.py`: Streamlit chat interface, session chat history, source excerpts/pages, and timing metrics.
- `college_notes_ai/config.py`: validated settings, errors, and shared data contracts.
- `college_notes_ai/rag.py`: retrieval, prompt policy, Ollama generation, and timing.
- `college_notes_ai/backend.py`: reusable `RAGService`; loads the persistent index and embedding model once per cached service.
- `rag_chat.py`: backward-compatible CLI entry point; existing CLI usage remains supported.
- `tests/`: existing offline RAG tests plus backend tests.

The app uses the existing `chroma_db` folder and collection `college_notes`. It does not rebuild the index.

## Install (Windows PowerShell)

From `D:\CollegeNotesAI`, after copying these files into your existing project:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

If Streamlit and the original RAG dependencies are not installed, install them in the same virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit chromadb sentence-transformers ollama
```

## Start the UI

Make sure Ollama is running and `ollama list` includes `llama3.2:3b`. Then run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Or double-click `start_app.bat` from the project folder. Streamlit prints a local URL (normally `http://localhost:8501`) in the terminal and may open it in your browser.

## Keep the existing CLI

```powershell
.\.venv\Scripts\python.exe rag_chat.py --question "What is an operating system?"
```

## Configuration

Defaults remain compatible with Milestone 5. Optional environment variables include `COLLEGE_NOTES_DB_PATH`, `RAG_COLLECTION`, `RAG_EMBEDDING_MODEL`, `RAG_OLLAMA_MODEL`, `OLLAMA_HOST`, `RAG_TOP_K`, `RAG_MAX_QUESTION_CHARS`, `RAG_MAX_CONTEXT_CHARS`, `RAG_TEMPERATURE`, and `RAG_MAX_DISTANCE`.

## Notes and limitations

- The Streamlit app is intended for local, single-user use on this computer. It is not authenticated for public access.
- Chat history lives in Streamlit session memory and is cleared when the session is reset; no chat database is added in this milestone.
- The first embedding-model load can be slower and may need internet if the model is not cached. The model itself and ChromaDB run locally.
- The UI shows retrieved pages/chunks; page citations generated in answer text are still model-generated and should be checked against the displayed sources.
- Offline tests use fakes. They do not prove the real Ollama model or your local index works; run a real question after starting the app.
