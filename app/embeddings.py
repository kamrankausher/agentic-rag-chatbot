from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL


class EmbeddingModel:
    """
    Singleton-style wrapper for the HuggingFace embedding model.
    """

    _embedding_model = None

    @classmethod
    def get_model(cls):
        """
        Load the embedding model only once.
        """

        if cls._embedding_model is None:
            cls._embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL
            )

        return cls._embedding_model