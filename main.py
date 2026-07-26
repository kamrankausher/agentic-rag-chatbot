from app.ingest import PDFIngestor
from app.retriever import VectorStore


def main():

    print("=" * 60)
    print("BUILDING VECTOR DATABASE")
    print("=" * 60)

    ingestor = PDFIngestor("data/agentic_ai_ebook.pdf")

    documents = ingestor.load_documents()

    chunks = ingestor.split_documents(documents)

    print(f"Loaded {len(chunks)} chunks.")

    vector_store = VectorStore()

    vector_store.add_documents(chunks)

    print("Knowledge base created successfully.")

    print("\n")

    query = "What is Agentic AI?"

    vector_store.print_search_results(query)


if __name__ == "__main__":
    main()