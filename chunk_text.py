"""Split page-marked text (output of extract_pdf.py) into overlapping chunks."""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "output" / "extracted_notes.txt"
OUTPUT_PATH = BASE_DIR / "output" / "chunks.txt"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Matches "--- Page 12 ---" on its own line. MULTILINE means it also matches
# on the very first line of the file (no leading "\n" required).
PAGE_MARKER = re.compile(r"^--- Page (\d+) ---[ \t]*$", re.MULTILINE)

# Preferred break points, strongest first.
BREAK_SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    page: int
    start: int  # character offset inside the page text
    text: str


def load_pages(file_path: Path) -> list[Page]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}\nRun extract_pdf.py first."
        )

    # newline normalisation so Windows-style files parse the same way
    text = file_path.read_text(encoding="utf-8").replace("\r\n", "\n")

    # re.split with one capture group -> [preamble, num, body, num, body, ...]
    parts = PAGE_MARKER.split(text)
    return [
        Page(int(num), body.strip())
        for num, body in zip(parts[1::2], parts[2::2])
        if body.strip()
    ]


def _find_break(text: str, start: int, end: int, min_end: int) -> int:
    """Pull `end` back to the nicest boundary inside [min_end, end)."""
    for sep in BREAK_SEPARATORS:
        idx = text.rfind(sep, min_end, end)
        if idx != -1:
            return idx + len(sep)
    return end  # no boundary found (e.g. one giant token): hard cut


def split_text(text: str, size: int, overlap: int) -> Iterator[tuple[int, str]]:
    """Yield (start_offset, chunk) pairs, cutting on natural boundaries."""
    n = len(text)
    start = 0

    while start < n:
        end = min(start + size, n)
        if end < n:
            # never shrink a chunk below half its target size
            end = _find_break(text, start, end, min_end=start + size // 2)

        chunk = text[start:end].strip()
        if chunk:
            yield start, chunk

        if end >= n:
            break

        # start the next chunk `overlap` chars back, but always make progress
        nxt = end - overlap
        if nxt <= start:
            nxt = end
        # snap forward to a word start so chunks don't begin mid-word
        while nxt < end and not text[nxt - 1].isspace():
            nxt += 1
        start = nxt


def create_chunks(pages: list[Page], size: int, overlap: int) -> Iterator[Chunk]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than chunk size")

    for page in pages:
        for start, text in split_text(page.text, size, overlap):
            yield Chunk(page.number, start, text)


def write_chunks(chunks: Iterator[Chunk], output_path: Path) -> list[Chunk]:
    """Stream chunks to disk; return them so the caller can report stats."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    with output_path.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"\n--- Chunk {i} | Page {chunk.page} ---\n{chunk.text}\n")
            written.append(chunk)
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=INPUT_PATH)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--size", type=int, default=CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pages = load_pages(args.input)
    chunks = write_chunks(create_chunks(pages, args.size, args.overlap), args.output)

    print(f"Pages processed: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Saved chunks to: {args.output}")

    if chunks:
        print("\nFirst chunk preview:\n")
        print(chunks[0].text)


if __name__ == "__main__":
    main()