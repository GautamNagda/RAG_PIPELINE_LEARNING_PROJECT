"""Validated configuration and shared data contracts for CollegeNotesAI."""
from __future__ import annotations
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger("college_notes_ai")


class RagError(RuntimeError):
    """Base class for expected application failures."""


class ConfigurationError(RagError):
    """Raised when a setting or question is invalid."""


class IndexErrorRag(RagError):
    """Raised when the vector index is missing, empty, or malformed."""


class RetrievalError(RagError):
    """Raised when vector search fails or returns invalid data."""


class GenerationError(RagError):
    """Raised when Ollama cannot generate a usable answer."""


@dataclass(frozen=True)
class Settings:
    """Validated settings for local retrieval and generation."""

    base_dir: Path
    db_path: Path
    collection_name: str = "college_notes"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_model: str = "llama3.2:3b"
    ollama_host: str = "http://127.0.0.1:11434"
    top_k: int = 5
    max_question_chars: int = 2000
    max_context_chars: int = 10000
    temperature: float = 0.0
    max_retrieval_distance: float | None = None

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> "Settings":
        """Read environment overrides and validate them before loading models."""
        root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
        db_path = Path(os.getenv("COLLEGE_NOTES_DB_PATH", str(root / "chroma_db"))).expanduser()
        try:
            top_k = int(os.getenv("RAG_TOP_K", "5"))
            max_question_chars = int(os.getenv("RAG_MAX_QUESTION_CHARS", "2000"))
            max_context_chars = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "10000"))
            temperature = float(os.getenv("RAG_TEMPERATURE", "0.0"))
            raw_max_distance = os.getenv("RAG_MAX_DISTANCE", "").strip()
            max_distance = float(raw_max_distance) if raw_max_distance else None
        except ValueError as exc:
            raise ConfigurationError(
                "RAG_TOP_K, RAG_MAX_QUESTION_CHARS, RAG_MAX_CONTEXT_CHARS, "
                "RAG_TEMPERATURE, and RAG_MAX_DISTANCE must contain valid numbers."
            ) from exc

        collection_name = os.getenv("RAG_COLLECTION", "college_notes").strip()
        embedding_model = os.getenv(
            "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip()
        ollama_model = os.getenv("RAG_OLLAMA_MODEL", "llama3.2:3b").strip()
        ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()

        if not collection_name or not embedding_model or not ollama_model or not ollama_host:
            raise ConfigurationError(
                "Collection, embedding model, Ollama model, and Ollama host must not be empty."
            )
        if top_k < 1 or top_k > 20:
            raise ConfigurationError("RAG_TOP_K must be between 1 and 20.")
        if not 1 <= max_question_chars <= 20000:
            raise ConfigurationError("RAG_MAX_QUESTION_CHARS must be between 1 and 20000.")
        if not 500 <= max_context_chars <= 50000:
            raise ConfigurationError("RAG_MAX_CONTEXT_CHARS must be between 500 and 50000.")
        if not math.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
            raise ConfigurationError("RAG_TEMPERATURE must be between 0.0 and 2.0.")
        if max_distance is not None and (not math.isfinite(max_distance) or max_distance < 0):
            raise ConfigurationError("RAG_MAX_DISTANCE must be a finite, non-negative number.")

        return cls(
            base_dir=root,
            db_path=db_path.resolve(),
            collection_name=collection_name,
            embedding_model=embedding_model,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            top_k=top_k,
            max_question_chars=max_question_chars,
            max_context_chars=max_context_chars,
            temperature=temperature,
            max_retrieval_distance=max_distance,
        )


@dataclass(frozen=True)
class SourceChunk:
    """A retrieved note passage and its source metadata."""

    text: str
    page: int
    distance: float | None = None


@dataclass(frozen=True)
class AnswerTimings:
    """Wall-clock timings for one question after application startup."""

    retrieval_seconds: float
    model_seconds: float
    total_seconds: float


