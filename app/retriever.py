from langchain_chroma import Chroma

from app.embeddings import EmbeddingModel


class VectorStore:

    def __init__(self):

        self.embedding_model = EmbeddingModel().get_model()

        self.vector_db = Chroma(
            persist_directory="chroma_db",
            embedding_function=self.embedding_model,
        )

    def add_documents(self, documents):

        self.vector_db.add_documents(documents)

    def similarity_search(self, query, k=4):

        return self.vector_db.similarity_search(query, k=k)