from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import CHROMA_DB_DIR, TOP_K
from app.embeddings import EmbeddingModel


class VectorStore:
    """
    Handles all ChromaDB operations.
    """

    def __init__(self):
        self.embedding_model = EmbeddingModel.get_model()

        self.vector_db = Chroma(
            persist_directory=str(CHROMA_DB_DIR),
            embedding_function=self.embedding_model,
            collection_name="agentic_ai",
        )

    def reset_database(self):
        """
        Remove every existing document from Chroma.
        Call this BEFORE indexing.
        """

        try:
            data = self.vector_db.get()

            ids = data.get("ids", [])

            if ids:
                self.vector_db.delete(ids=ids)

        except Exception:
            pass

    def add_documents(self, documents: list[Document]):
        """
        Add chunks with deterministic IDs.
        """

        if not documents:
            return

        ids = []

        for doc in documents:
            ids.append(
                f"{doc.metadata['source']}_page_{doc.metadata['page']}_chunk_{doc.metadata['chunk_id']}"
            )

        self.vector_db.add_documents(
            documents=documents,
            ids=ids,
        )

    def similarity_search(
        self,
        query: str,
        k: int = TOP_K,
    ):
        """
        Return the top-k unique results.
        """

        results = self.vector_db.similarity_search_with_score(
            query=query,
            k=max(k * 3, 10),
        )

        unique = []
        seen = set()

        for doc, score in results:

            uid = (
                doc.metadata.get("page"),
                doc.metadata.get("chunk_id"),
            )

            if uid in seen:
                continue

            seen.add(uid)
            unique.append((doc, score))

            if len(unique) == k:
                break

        return unique

    def get_retriever(self):
        return self.vector_db.as_retriever(
            search_kwargs={
                "k": TOP_K
            }
        )

    def document_count(self):
        return self.vector_db._collection.count()