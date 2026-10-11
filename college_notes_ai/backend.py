
"""Application backend: resource initialization and question-answer service."""

from __future__ import annotations

from typing import Any

from .config import Settings
from .rag import (
    AnswerTimings,
    SourceChunk,
    answer_question_with_timings,
    load_embedding_model,
    open_index,
)


class RAGService:
    """Reusable service that keeps the index and embedding model alive in memory."""

    def __init__(
        self,
        settings: Settings,
        collection: Any,
        embedding_model: Any,
    ) -> None:
        self.settings = settings
        self.collection = collection
        self.embedding_model = embedding_model

    @classmethod
    def load(cls, settings: Settings | None = None) -> "RAGService":
        """Open the existing index and load embeddings once per service."""
        resolved_settings = settings or Settings.from_env()
        collection = open_index(resolved_settings)
        embedding_model = load_embedding_model(
            resolved_settings.embedding_model
        )

        return cls(
            resolved_settings,
            collection,
            embedding_model,
        )

    def ask(
        self, question: str
    ) -> tuple[str, list[SourceChunk], AnswerTimings]:
        """Return an answer, its retrieved source chunks, and timings."""
        return answer_question_with_timings(
            question,
            self.settings,
            self.collection,
            self.embedding_model,
        )
