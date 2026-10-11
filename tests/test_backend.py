"""Offline tests for the application backend service."""
from pathlib import Path
import unittest
from unittest.mock import patch

from college_notes_ai.backend import RAGService
from college_notes_ai.config import AnswerTimings, Settings, SourceChunk


class BackendTests(unittest.TestCase):
    def test_load_initializes_index_and_embedding_model(self):
        settings = Settings(base_dir=Path("/tmp/project"), db_path=Path("/tmp/project/chroma_db"))
        collection, embedding = object(), object()
        with patch("college_notes_ai.backend.open_index", return_value=collection) as open_index, \
             patch("college_notes_ai.backend.load_embedding_model", return_value=embedding) as load_model:
            service = RAGService.load(settings)
        open_index.assert_called_once_with(settings)
        load_model.assert_called_once_with(settings.embedding_model)
        self.assertIs(service.collection, collection)
        self.assertIs(service.embedding_model, embedding)

    def test_ask_delegates_and_returns_answer_sources_and_timings(self):
        settings = Settings(base_dir=Path("/tmp/project"), db_path=Path("/tmp/project/chroma_db"))
        collection, embedding = object(), object()
        service = RAGService(settings, collection, embedding)
        expected = ("Answer", [SourceChunk("Evidence", page=3)], AnswerTimings(.1, 2.0, 2.1))
        with patch("college_notes_ai.backend.answer_question_with_timings", return_value=expected) as ask:
            actual = service.ask("What is an OS?")
        ask.assert_called_once_with("What is an OS?", settings, collection, embedding, ollama_client=None)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
