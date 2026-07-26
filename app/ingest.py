from pathlib import Path

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFIngestor:
    """
    Read a PDF and convert it into LangChain Documents.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)

    def load_documents(self) -> list[Document]:
        """
        Load every page of the PDF.
        """

        pdf = fitz.open(self.pdf_path)

        documents = []

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            if len(text) == 0:
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

        pdf.close()

        return documents

    def split_documents(
        self,
        documents: list[Document],
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> list[Document]:
        """
        Split documents into chunks while preserving metadata.
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks = splitter.split_documents(documents)

        for chunk_id, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        return chunks


def main():

    ingestor = PDFIngestor("data/agentic_ai_ebook.pdf")

    documents = ingestor.load_documents()

    chunks = ingestor.split_documents(documents)

    print("=" * 60)
    print("PDF INGESTION SUMMARY")
    print("=" * 60)

    print(f"Pages Loaded   : {len(documents)}")
    print(f"Chunks Created : {len(chunks)}")

    print("\nFirst Chunk\n")
    print(chunks[0].page_content[:300])

    print("\nMetadata\n")
    print(chunks[0].metadata)


if __name__ == "__main__":
    main()