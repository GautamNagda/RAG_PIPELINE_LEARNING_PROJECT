"""UI control-flow smoke tests using a tiny Streamlit test double.

These tests do not replace a real browser launch; they verify the app's Python
control flow without requiring Streamlit to be installed in the test environment.
"""
from __future__ import annotations
import importlib
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from college_notes_ai.config import AnswerTimings, GenerationError, Settings, SourceChunk


class _Context:
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        return False


class _Column:
    def metric(self, *_args, **_kwargs):
        pass


class _SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
    def __setattr__(self, key, value):
        self[key] = value


class _FakeStreamlit(types.ModuleType):
    def __init__(self, question=None):
        super().__init__("streamlit")
        self.session_state = _SessionState()
        self._question = question
        self.sidebar = _Context()
        self.errors = []

    def set_page_config(self, **_kwargs): pass
    def markdown(self, *_args, **_kwargs): pass
    def title(self, *_args, **_kwargs): pass
    def caption(self, *_args, **_kwargs): pass
    def button(self, *_args, **_kwargs): return False
    def divider(self): pass
    def subheader(self, *_args, **_kwargs): pass
    def chat_input(self, *_args, **_kwargs):
        question, self._question = self._question, None
        return question
    def chat_message(self, *_args, **_kwargs): return _Context()
    def expander(self, *_args, **_kwargs): return _Context()
    def columns(self, count): return [_Column() for _ in range(count)]
    def spinner(self, *_args, **_kwargs): return _Context()
    def write(self, *_args, **_kwargs): pass
    def rerun(self): pass
    def error(self, message): self.errors.append(message)
    def cache_resource(self, **_kwargs):
        def decorate(function):
            return function
        return decorate


class _FakeService:
    def __init__(self, failure=None): self.failure = failure
    @classmethod
    def load(cls, _settings): return cls()
    def ask(self, _question):
        if self.failure:
            raise self.failure
        return "A short answer.", [SourceChunk("Supporting note excerpt", page=2)], AnswerTimings(.02, 1.2, 1.23)


class UISmokeTests(unittest.TestCase):
    def run_app(self, question=None, service=None):
        fake_st = _FakeStreamlit(None)
        previous_st = sys.modules.get("streamlit")
        sys.modules["streamlit"] = fake_st
        try:
            import app as app_module
            fake_st._question = question
            with patch.object(app_module.RAGService, "load", return_value=service or _FakeService()):
                importlib.reload(app_module)
            return fake_st
        finally:
            if previous_st is None:
                sys.modules.pop("streamlit", None)
            else:
                sys.modules["streamlit"] = previous_st
            sys.modules.pop("app", None)

    def test_app_loads_without_a_question(self):
        fake_st = self.run_app()
        self.assertEqual(fake_st.session_state["messages"], [])

    def test_submitted_question_saves_answer_sources_and_timings(self):
        fake_st = self.run_app("What is an OS?")
        self.assertEqual(len(fake_st.session_state["messages"]), 2)
        assistant = fake_st.session_state["messages"][1]
        self.assertEqual(assistant["content"], "A short answer.")
        self.assertEqual(assistant["sources"][0]["page"], 2)
        self.assertAlmostEqual(assistant["timings"]["model_seconds"], 1.2)

    def test_rag_failure_is_shown_as_an_error_message(self):
        fake_st = _FakeStreamlit(None)
        previous_st = sys.modules.get("streamlit")
        sys.modules["streamlit"] = fake_st
        try:
            import app as app_module
            fake_st._question = "question"
            service = _FakeService(GenerationError("Ollama unavailable"))
            with patch.object(app_module.RAGService, "load", return_value=service):
                importlib.reload(app_module)
            self.assertIn("Ollama unavailable", fake_st.errors)
            self.assertTrue(fake_st.session_state["messages"][-1]["error"])
        finally:
            if previous_st is None: sys.modules.pop("streamlit", None)
            else: sys.modules["streamlit"] = previous_st
            sys.modules.pop("app", None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
