"""Search the notes index: embed a question and show the most relevant chunks."""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb

# Single source of truth: names, paths and model loading live in index_chunks.py,
# so the indexer and the searcher can never drift apart.
from build_index import COLLECTION_NAME, DB_PATH, load_model

TOP_K = 3
# Cosine similarity: 1.0 = identical meaning, ~0 = unrelated. Results below this
# are dropped so an off-topic question doesn't get "best of the worst" chunks.
# 0.2 is a starting point for all-MiniLM-L6-v2: tune it on your own questions.
MIN_SCORE = 0.2

log = logging.getLogger("search_notes")


@dataclass(frozen=True)
class Hit:
    chunk: int | None
    page: int | None
    score: float
    text: str


# ----------------------------------------------------------------- index
def open_collection(db_path: Path, name: str):
    # PersistentClient CREATES the folder if missing, which would leave an empty
    # fake database behind and a confusing "collection not found" error.
    if not db_path.is_dir():
        raise FileNotFoundError(f"No database at {db_path}. Run index_chunks.py first.")

    client = chromadb.PersistentClient(path=str(db_path))
    names = {getattr(c, "name", c) for c in client.list_collections()}
    if name not in names:
        raise RuntimeError(f"Collection '{name}' not found in {db_path}. Run index_chunks.py first.")

    collection = client.get_collection(name)
    meta = collection.metadata or {}

    if "embedding_model" not in meta or meta.get("hnsw:space") != "cosine":
        raise RuntimeError(
            "This index was built by an older indexer (no model or cosine setting "
            "recorded), so scores can't be trusted. Re-run index_chunks.py to rebuild it."
        )
    if collection.count() == 0:
        raise RuntimeError("The index is empty. Re-run index_chunks.py.")
    return collection


# ---------------------------------------------------------------- search
def search(collection, model, question: str, top_k: int = TOP_K,
           min_score: float = MIN_SCORE) -> list[Hit]:
    """Return the chunks most similar to `question`, best first."""
    question = question.strip()
    if not question:
        raise ValueError("The question is empty.")

    vector = model.encode(
        [question],
        normalize_embeddings=True,  # same as the indexer, or scores are wrong
        show_progress_bar=False,
    ).tolist()

    result = collection.query(
        query_embeddings=vector,
        n_results=min(top_k, collection.count()),  # asking for more than exist can error
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        score = 1.0 - distance  # cosine distance -> cosine similarity
        if score >= min_score:
            hits.append(Hit(meta.get("chunk"), meta.get("page"), score, text))
    return hits


def print_hits(hits: list[Hit], min_score: float) -> None:
    if not hits:
        print(f"\nNo sufficiently relevant notes found (nothing scored >= {min_score}).")
        print("Try rephrasing, or lower --min-score.")
        return
    print()
    for i, hit in enumerate(hits, start=1):
        print(f"Result {i} | page {hit.page} | chunk {hit.chunk} | match {hit.score:.2f}")
        print(hit.text)
        print("-" * 50)


# ------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("question", nargs="?", help="ask once and exit (omit for interactive mode)")
    p.add_argument("--db", type=Path, default=DB_PATH)
    p.add_argument("--collection", default=COLLECTION_NAME)
    p.add_argument("--model", help="must match the model that built the index (default: auto)")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--min-score", type=float, default=MIN_SCORE)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # logs -> stderr

    if args.top_k < 1:
        log.error("--top-k must be at least 1.")
        return 1

    try:
        collection = open_collection(args.db, args.collection)
        indexed_with = collection.metadata["embedding_model"]
        if args.model and args.model != indexed_with:
            raise RuntimeError(
                f"Model mismatch: the index was built with '{indexed_with}' but you asked for "
                f"'{args.model}'. Vectors from different models can't be compared."
            )
        log.info("Loading embedding model...")
        model = load_model(indexed_with)  # always the model that built the index
    except (FileNotFoundError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1

    # One-shot mode (scriptable)
    if args.question is not None:
        try:
            print_hits(search(collection, model, args.question, args.top_k, args.min_score),
                       args.min_score)
        except ValueError as exc:
            log.error("%s", exc)
            return 1
        return 0

    # Interactive mode: the model is loaded once, then questions are cheap.
    log.info("Ready. Ask about your notes ('exit' to quit).")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            log.warning("Please type a question.")
            continue
        print_hits(search(collection, model, question, args.top_k, args.min_score),
                   args.min_score)
    return 0


if __name__ == "__main__":
    sys.exit(main())