from pathlib import Path

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


class PDFIngestor:
    """
    Handles PDF loading and document chunking.
    """

    def __init__(self, pdf_path: Path = PDF_PATH):
        self.pdf_path = pdf_path

    def load_documents(self) -> list[Document]:
        """
        Read the PDF and convert each page into a LangChain Document.
        """

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {self.pdf_path}"
            )

        pdf = fitz.open(self.pdf_path)

        documents = []

        try:
            for page_number, page in enumerate(pdf, start=1):

                text = page.get_text("text").strip()

                if not text:
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "page": page_number,
                            "source": self.pdf_path.name,
                        },
                    )
                )

        finally:
            pdf.close()

        return documents

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """
        Split documents into smaller overlapping chunks.
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

        chunks = splitter.split_documents(documents)

        for chunk_id, chunk in enumerate(chunks):

            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        return chunks

    def ingest(self) -> list[Document]:
        """
        Complete ingestion pipeline.
        """

        documents = self.load_documents()

        chunks = self.split_documents(documents)

        return chunks


if __name__ == "__main__":

    ingestor = PDFIngestor()

    chunks = ingestor.ingest()

    print("=" * 60)
    print("PDF INGESTION SUMMARY")
    print("=" * 60)

    print(f"Total Chunks : {len(chunks)}")

    print("\nFirst Chunk\n")
    print(chunks[0].page_content[:500])

    print("\nMetadata\n")
    print(chunks[0].metadata)