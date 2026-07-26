from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
)
from app.prompts import SYSTEM_PROMPT
from app.retriever import VectorStore


class RAGPipeline:
    """
    Handles the complete Retrieval-Augmented Generation (RAG) pipeline.
    """

    def __init__(self):

        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found. "
                "Please add it to your .env file."
            )

        self.vector_store = VectorStore()

        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
        )

    def retrieve(self, question: str):
        """
        Retrieve the most relevant chunks from ChromaDB.
        """

        return self.vector_store.similarity_search(question)

    def build_context(self, retrieved_docs):
        """
        Convert retrieved documents into a single context string.
        """

        context_parts = []

        for doc, _score in retrieved_docs:
            context_parts.append(doc.page_content)

        return "\n\n".join(context_parts)

    def generate_answer(
        self,
        question: str,
        context: str,
    ):
        """
        Generate a grounded answer using Gemini.
        """

        prompt = SYSTEM_PROMPT.format(
            context=context,
            question=question,
        )

        response = self.llm.invoke(prompt)

        return response.content

    def ask(self, question: str):
        """
        Complete RAG pipeline.

        Returns:
            answer
            retrieved context
            similarity scores
        """

        retrieved_docs = self.retrieve(question)

        context = self.build_context(retrieved_docs)

        answer = self.generate_answer(
            question=question,
            context=context,
        )

        retrieved_chunks = []

        for doc, score in retrieved_docs:

            retrieved_chunks.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                }
            )

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
        }