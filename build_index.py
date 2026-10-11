"""Embed chunks.txt and store the vectors in a persistent ChromaDB collection."""

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Must be set BEFORE sentence_transformers / huggingface_hub are imported,
# otherwise the model downloads to the default cache (~/.cache/huggingface).
os.environ.setdefault("HF_HOME", str(BASE_DIR / "hf_cache"))

import chromadb  # noqa: E402  (import order matters, see HF_HOME above)

CHUNKS_PATH = BASE_DIR / "output" / "chunks.txt"
DB_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "college_notes"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64

# "--- Chunk 12 | Page 3 ---" on its own line (contract with create_chunks.py)
CHUNK_HEADER = re.compile(r"^--- Chunk (\d+) \| Page (\d+) ---[ \t]*$", re.MULTILINE)

log = logging.getLogger("index_chunks")


@dataclass(frozen=True)
class Chunk:
    id: int
    page: int
    text: str


# ---------------------------------------------------------------- loading
def load_chunks(path: Path) -> list[Chunk]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run the chunking script first.")

    content = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    # split() with 2 capture groups -> [preamble, id, page, body, id, page, body, ...]
    parts = CHUNK_HEADER.split(content)

    chunks, seen = [], set()
    for raw_id, raw_page, body in zip(parts[1::3], parts[2::3], parts[3::3]):
        chunk_id = int(raw_id)
        if chunk_id in seen:
            raise ValueError(
                f"Duplicate chunk id {chunk_id} in {path.name}. A chunk's text may "
                "contain a fake '--- Chunk N | Page N ---' line, or the file was "
                "edited by hand."
            )
        seen.add(chunk_id)

        text = body.strip()
        if text:
            chunks.append(Chunk(chunk_id, int(raw_page), text))
    return chunks


# -------------------------------------------------------------- embedding
def load_model(model_name: str):
    # Imported here so `--help` and file errors don't wait for torch to load.
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"Could not load embedding model '{model_name}'. The first run needs "
            f"internet to download it (~90 MB into {os.environ['HF_HOME']}).\n"
            f"Original error: {exc}"
        ) from exc


def warn_if_truncated(model, chunks: list[Chunk]) -> None:
    """Embedding models silently ignore text past their token limit."""
    tokenizer = getattr(model, "tokenizer", None)
    limit = getattr(model, "max_seq_length", None)
    if tokenizer is None or not limit:
        return

    ids = tokenizer([c.text for c in chunks], truncation=False, add_special_tokens=True)
    too_long = [c.id for c, toks in zip(chunks, ids["input_ids"]) if len(toks) > limit]
    if too_long:
        log.warning(
            "%d chunk(s) exceed the model's %d-token limit; the extra text is NOT "
            "embedded and cannot be found by search (chunk ids: %s%s). "
            "Use a smaller chunk size.",
            len(too_long), limit, too_long[:10], " ..." if len(too_long) > 10 else "",
        )


# --------------------------------------------------------------- indexing
def _collection_names(client) -> set[str]:
    # older chromadb returns Collection objects, newer returns plain names
    return {getattr(c, "name", c) for c in client.list_collections()}


def build_index(chunks, model, model_name, db_path, collection_name, batch_size):
    """Build a fresh collection, then swap it in. A crash never corrupts the old one."""
    client = chromadb.PersistentClient(path=str(db_path))
    tmp_name = f"{collection_name}__building"

    if tmp_name in _collection_names(client):  # leftover from a crashed run
        client.delete_collection(tmp_name)

    collection = client.create_collection(
        name=tmp_name,
        metadata={
            "hnsw:space": "cosine",  # all-MiniLM is trained for cosine similarity
            "embedding_model": model_name,  # so the query script can verify it matches
        },
    )

    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start:start + batch_size]
        texts = [c.text for c in batch]
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        collection.add(  # add (not upsert): ids are unique, a clash is a bug
            ids=[f"chunk_{c.id}" for c in batch],
            documents=texts,
            metadatas=[{"page": c.page, "chunk": c.id} for c in batch],
            embeddings=vectors.tolist(),
        )
        log.info("Indexed %d/%d chunks", min(start + batch_size, total), total)

    if collection.count() != total:
        raise RuntimeError(f"Expected {total} chunks in the index, found {collection.count()}.")

    # Swap in the new index. Old data (including chunks that no longer exist) goes away.
    if collection_name in _collection_names(client):
        client.delete_collection(collection_name)
    collection.modify(name=collection_name)
    return collection


# -------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    p.add_argument("--db", type=Path, default=DB_PATH)
    p.add_argument("--collection", default=COLLECTION_NAME)
    p.add_argument("--model", default=EMBED_MODEL)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        chunks = load_chunks(args.chunks)
        if not chunks:
            raise ValueError(f"No chunks found in {args.chunks}.")
        log.info("Loaded %d chunks.", len(chunks))

        log.info("Loading embedding model...")
        model = load_model(args.model)
        warn_if_truncated(model, chunks)

        collection = build_index(
            chunks, model, args.model, args.db, args.collection, args.batch_size
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1

    log.info("Index created successfully!")
    log.info("Total indexed chunks: %d", collection.count())
    log.info("Database location: %s", args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())