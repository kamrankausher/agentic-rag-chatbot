from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.embeddings import get_embedding_model


class VectorStore:
    """
    Create and manage the Chroma vector database.
    """

    def __init__(self):
        self.embedding_model = get_embedding_model()

        self.vector_db = Chroma(
            persist_directory="chroma_db",
            embedding_function=self.embedding_model,
        )

    def add_documents(self, documents: list[Document]):
        """
        Add documents to the vector database.
        """

        if documents:
            self.vector_db.add_documents(documents)

    def similarity_search(self, query: str, k: int = 4):
        """
        Return the most similar chunks with similarity scores.
        """

        return self.vector_db.similarity_search_with_score(
            query=query,
            k=k,
        )

    def get_retriever(self, k: int = 4):
        """
        Return a LangChain retriever.
        """

        return self.vector_db.as_retriever(
            search_kwargs={"k": k}
        )

    def print_search_results(self, query: str, k: int = 4):
        """
        Print retrieved chunks for testing.
        """

        results = self.similarity_search(query, k)

        print("=" * 80)
        print(f"QUESTION : {query}")
        print("=" * 80)

        for i, (doc, score) in enumerate(results, start=1):

            print(f"\nResult {i}")
            print("-" * 80)

            print(f"Similarity Score : {score:.4f}")
            print(f"Page             : {doc.metadata.get('page')}")
            print(f"Chunk ID         : {doc.metadata.get('chunk_id')}")

            print("\nContent:\n")
            print(doc.page_content[:500])

            print("\n")