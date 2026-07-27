from google import genai

from app.config import GEMINI_MODEL, GOOGLE_API_KEY
from app.prompts import SYSTEM_PROMPT
from app.retriever import VectorStore


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    """

    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please set it in your .env file."
            )

        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.vector_store = VectorStore()

    def retrieve(self, question: str):
        """
        Retrieve relevant chunks from ChromaDB.
        """

        results = self.vector_store.similarity_search(question)

        retrieved_chunks = []

        context = []

        for doc, score in results:
            retrieved_chunks.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                }
            )

            context.append(doc.page_content)

        return {
            "context": "\n\n".join(context),
            "retrieved_chunks": retrieved_chunks,
        }

    def generate(self, question: str, context: str):
        """
        Generate answer using Gemini.
        """

        prompt = SYSTEM_PROMPT.format(
            context=context,
            question=question,
        )

        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        return response.text.strip()

    def ask(self, question: str):
        """
        Complete RAG pipeline.
        """

        retrieval = self.retrieve(question)

        answer = self.generate(
            question=question,
            context=retrieval["context"],
        )

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieval["retrieved_chunks"],
        }