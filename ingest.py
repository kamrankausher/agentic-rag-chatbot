from pathlib import Path

import fitz


PDF_PATH = Path("data/agentic_ai_ebook.pdf")


def load_pdf(pdf_path: Path) -> str:
    """Load a PDF and return all extracted text."""

    document = fitz.open(pdf_path)

    pages_text = []

    for page in document:
        text = page.get_text()
        pages_text.append(text)

    document.close()

    return "\n".join(pages_text)


def main():
    text = load_pdf(PDF_PATH)

    print("=" * 60)
    print("PDF Loaded Successfully")
    print("=" * 60)
    print(f"Total Characters: {len(text):,}")
    print()
    print("First 500 characters:\n")
    print(text[:500])


if __name__ == "__main__":
    main()