# CollegeNotesAI

CollegeNotesAI is a Retrieval-Augmented Generation (RAG) chatbot that helps students ask questions about college notes and receive answers based on relevant content retrieved from their study material.

## Project Overview

The application processes college notes in PDF format, indexes their content, retrieves relevant sections for a student's question, and uses a language model to generate a response.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main application logic |
| Streamlit | Chat interface and web application |
| PyMuPDF | Extracts text from PDF notes |
| Sentence Transformers | Converts text into semantic embeddings |
| `all-MiniLM-L6-v2` | Embedding model used for questions and note chunks |
| ChromaDB | Stores embeddings and retrieves relevant note chunks |
| Groq API | Connects the application to a hosted language model |
| `openai/gpt-oss-20b` | Generates answers using the question and retrieved context |
| Ollama + `llama3.2:3b` | Earlier local model setup used during development |

## How the Project Works

### 1. PDF Processing

PyMuPDF extracts text from the uploaded or provided PDF notes.

### 2. Text Chunking

The extracted text is divided into smaller sections called chunks. Chunking helps the application retrieve focused pieces of information rather than searching one large document.

### 3. Embedding Generation

Sentence Transformers uses `sentence-transformers/all-MiniLM-L6-v2` to convert each chunk into a numerical vector, also called an embedding.

### 4. Vector Storage

ChromaDB stores the embeddings, text chunks, and available metadata such as page information in a collection named `college_notes`.

### 5. Question Retrieval

When a student asks a question, the same embedding model converts the question into a vector. ChromaDB searches for the most relevant chunks. The project configuration previously used `top_k = 5`.

### 6. Answer Generation

The application combines the student's question with the retrieved context and sends it to `openai/gpt-oss-20b` through the Groq API.

### 7. Answer Display

Streamlit displays the generated response to the student. Page citations are shown when supported by the retrieved metadata and response pipeline.

## Architecture

```text
College Notes PDF
       |
       v
PyMuPDF Text Extraction
       |
       v
Text Chunking
       |
       v
Sentence Transformers
(all-MiniLM-L6-v2)
       |
       v
ChromaDB Vector Collection
(college_notes)
       |
       |                  Student Question
       |                         |
       |                         v
       |                 Streamlit Interface
       |                         |
       |                         v
       |                 Question Embedding
       |                         |
       v                         v
       +----> Relevant Chunks <---+
                       |
                       v
             Question + Retrieved Context
                       |
                       v
                    Groq API
             (openai/gpt-oss-20b)
                       |
                       v
                Generated Answer
                       |
                       v
                Streamlit Interface
```

## Main Components

### Streamlit Interface
Provides the user-facing chatbot interface where students submit questions and view answers.

### PDF Extraction Module
Reads PDF notes and extracts their text for processing.

### Chunking Module
Splits extracted text into smaller sections suitable for embedding and retrieval.

### Embedding Model
The `all-MiniLM-L6-v2` model converts text into vectors so semantically related questions and passages can be matched.

### ChromaDB Vector Database
Stores vector embeddings with their associated text and metadata, then retrieves relevant chunks for a question.

### RAG Backend
Coordinates retrieval, prompt construction, model invocation, and the answer returned to the interface.

### Groq and the Generative Model
Groq provides API access to `openai/gpt-oss-20b`, which generates answers using the question and retrieved note content.

## Project Structure

The following is a guide to the files used during development. File names may differ in the current repository.

```text
CollegeNotesAI/
├── app.py                       # Streamlit entry point, if named this way
├── rag_chat.py                  # RAG / command-line entry point
├── extract_pdf.py               # PDF text extraction
├── chunk_text.py                # Text chunking
├── build_index.py               # Build the vector index
├── requirements.txt             # Python dependencies
├── college_notes_ai/
│   ├── backend.py               # Backend orchestration
│   └── rag.py                   # Retrieval and answer generation
├── documents/
│   └── Operating_System_Notes.pdf
├── output/
│   └── extracted_notes.txt
├── chroma_db/                   # ChromaDB data, if stored in the project
└── tests/
    ├── test_backend.py
    └── test_rag_chat.py
```

## Configuration

The application uses these configuration values for hosted answer generation:

```text
GROQ_API_KEY = Groq API key
GROQ_MODEL = openai/gpt-oss-20b
```

The API key is provided through the local environment or Streamlit Cloud Secrets. The model name identifies the generative model used for answering questions.

## Run the Application Locally

From the project root on Windows, run:

```cmd
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Replace `app.py` with the actual Streamlit entry-point filename if it differs.

## Prepare the Knowledge Base

The project used these scripts during development:

```cmd
.\.venv\Scripts\python.exe extract_pdf.py
.\.venv\Scripts\python.exe chunk_text.py
.\.venv\Scripts\python.exe build_index.py
```

These scripts extract PDF text, split it into chunks, and build the ChromaDB index. Check the paths configured in the scripts before running them.

## Testing

Run the project's test suite from the project root:

```cmd
.\.venv\Scripts\python.exe -m pytest -q
```

The project previously reached 32 passing tests during development.

## Deployment

The application can be deployed using Streamlit Community Cloud. The deployment uses the project's GitHub repository, the configured Streamlit entry-point file, and the `GROQ_API_KEY` and `GROQ_MODEL` values supplied in Streamlit Cloud Secrets.
The PDF notes and ChromaDB index must be available to the deployed application for retrieval to work.
