from pathlib import Path

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Path to the PDF
PDF_PATH = Path("data/agentic_ai_ebook.pdf")


def load_pdf(pdf_path: Path) -> str:
    """
    Load a PDF and return all extracted text as a single string.
    """

    document = fitz.open(pdf_path)

    pages_text = []

    for page in document:
        text = page.get_text()
        pages_text.append(text)

    document.close()

    return "\n".join(pages_text)


def split_text(text: str) -> list[str]:
    """
    Split extracted text into smaller chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )

    chunks = splitter.split_text(text)

    return chunks


def main():
    # Load the PDF
    text = load_pdf(PDF_PATH)

    # Split into chunks
    chunks = split_text(text)

    print("=" * 60)
    print("PDF Loaded Successfully")
    print("=" * 60)

    print(f"Total Characters : {len(text):,}")
    print(f"Total Chunks     : {len(chunks)}")

    print("\n" + "=" * 60)
    print("First Chunk")
    print("=" * 60)
    print(chunks[0])

    print("\n" + "=" * 60)
    print("Last Chunk")
    print("=" * 60)
    print(chunks[-1])


if __name__ == "__main__":
    main()