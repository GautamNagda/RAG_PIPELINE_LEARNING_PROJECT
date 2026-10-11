"""Backward-compatible command-line entry point for CollegeNotesAI."""
from __future__ import annotations
import argparse
import logging
import time
from typing import Sequence

from college_notes_ai.config import (
    AnswerTimings, ConfigurationError, GenerationError, IndexErrorRag,
    LOGGER, RagError, RetrievalError, Settings, SourceChunk,
)
from college_notes_ai.rag import (
    SYSTEM_PROMPT, answer_question, answer_question_with_timings, answer_style,
    build_prompt, format_result, generate_answer, generate_answer_with_timing,
    load_embedding_model, open_index, retrieve_context, validate_question,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask questions about your local college notes.")
    parser.add_argument("--question", help="Ask one question and exit. Without this flag, starts interactive mode.")
    parser.add_argument("--verbose", action="store_true", help="Show diagnostic logs.")
    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")
    try:
        settings = Settings.from_env()
        collection = open_index(settings)
        model = load_embedding_model(settings.embedding_model)
        if args.question is not None:
            answer, sources, timings = answer_question_with_timings(args.question, settings, collection, model)
            print(format_result(answer, sources, timings))
            return 0
        print("CollegeNotesAI is ready. Ask a question, or type 'exit' to quit.")
        while True:
            try:
                question = input("\nYou: ").strip()
            except EOFError:
                print()
                return 0
            if question.lower() in {"exit", "quit"}:
                return 0
            if not question:
                print("Please enter a question, or type 'exit' to quit.")
                continue
            try:
                answer, sources, timings = answer_question_with_timings(question, settings, collection, model)
                print(format_result(answer, sources, timings))
            except RagError as exc:
                LOGGER.error("Question failed: %s", exc)
                return 1
    except RagError as exc:
        LOGGER.error("Startup failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
