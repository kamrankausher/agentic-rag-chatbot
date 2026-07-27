from pathlib import Path

from app.config import PDF_PATH
from app.ingest import PDFIngestor
from app.retriever import VectorStore


def main():
    print("=" * 70)
    print("Agentic AI RAG Chatbot - PDF Indexing")
    print("=" * 70)

    # Check PDF exists
    if not Path(PDF_PATH).exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    # Load and chunk PDF
    print("\nLoading and chunking PDF...")

    ingestor = PDFIngestor()
    chunks = ingestor.ingest()

    print(f"✓ Generated {len(chunks)} chunks")

    # Initialize vector store
    print("\nInitializing ChromaDB...")

    vector_store = VectorStore()

    # Remove previous embeddings
    print("Removing existing vectors...")
    vector_store.reset_database()

    # Add new embeddings
    print("Creating embeddings and indexing documents...")
    vector_store.add_documents(chunks)

    total_vectors = vector_store.document_count()

    print("\n" + "=" * 70)
    print("Indexing completed successfully!")
    print("=" * 70)
    print(f"PDF             : {PDF_PATH.name}")
    print(f"Chunks Indexed  : {len(chunks)}")
    print(f"Stored Vectors  : {total_vectors}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nIndexing failed!")
        print(f"Error: {e}")
        raise