
from pathlib import Path
import pymupdf

# Locate our project folders
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "documents" / "Operating_System_Notes.pdf"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "extracted_notes.txt"


def extract_pdf_text(pdf_path):
    """Extract text from a PDF, keeping page boundaries."""

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Check the filename and documents folder."
        )

    extracted_pages = []

    with pymupdf.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf)}")

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            extracted_pages.append(
                f"\n--- Page {page_number} ---\n{text}"
            )

            print(
                f"Processed page {page_number}: "
                f"{len(text)} characters"
            )

    return "\n".join(extracted_pages)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading your college notes...")

    text = extract_pdf_text(PDF_PATH)

    if not text.strip():
        print(
            "No selectable text found. "
            "Your PDF may contain scanned images."
        )
        return

    OUTPUT_PATH.write_text(text, encoding="utf-8")

    print("\nPDF extraction successful!")
    print(f"Saved extracted text to: {OUTPUT_PATH}")
    print("\nPreview of extracted text:\n")
    print(text[:1500])


if __name__ == "__main__":
    main()
