from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import CHROMA_DB_DIR, TOP_K
from app.embeddings import EmbeddingModel


class VectorStore:
    """
    Handles all interactions with the Chroma vector database.
    """

    def __init__(self):
        self.embedding_model = EmbeddingModel.get_model()

        self.vector_db = Chroma(
            persist_directory=str(CHROMA_DB_DIR),
            embedding_function=self.embedding_model,
        )

    def reset_database(self):
        """
        Delete all existing vectors from the database.
        This is useful during development to avoid duplicate embeddings.
        """

        try:
            ids = self.vector_db.get()["ids"]

            if ids:
                self.vector_db.delete(ids=ids)

        except Exception:
            pass

    def add_documents(self, documents: list[Document]):
        """
        Store document chunks inside ChromaDB.
        """

        if not documents:
            return

        self.vector_db.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: int = TOP_K,
    ):
        """
        Return the most relevant document chunks
        together with similarity scores.
        """

        return self.vector_db.similarity_search_with_score(
            query=query,
            k=k,
        )

    def get_retriever(self, k: int = TOP_K):
        """
        Return a LangChain Retriever.
        This will later be used by LangGraph.
        """

        return self.vector_db.as_retriever(
            search_kwargs={
                "k": k
            }
        )

    def document_count(self):
        """
        Return the total number of vectors stored.
        """

        return self.vector_db._collection.count()