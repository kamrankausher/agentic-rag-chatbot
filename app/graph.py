from typing import TypedDict

from langgraph.graph import StateGraph, END

from app.rag import RAGPipeline


class GraphState(TypedDict):
    """
    State shared across the LangGraph workflow.
    """

    question: str
    answer: str
    retrieved_chunks: list


class AgenticRAGGraph:
    """
    LangGraph workflow for the RAG chatbot.
    """

    def __init__(self):
        self.rag = RAGPipeline()
        self.graph = self._build_graph()

    def retrieve_and_generate(self, state: GraphState):
        """
        Retrieve relevant context and generate an answer.
        """

        result = self.rag.ask(state["question"])

        return {
            "question": state["question"],
            "answer": result["answer"],
            "retrieved_chunks": result["retrieved_chunks"],
        }

    def _build_graph(self):
        """
        Create the LangGraph workflow.
        """

        workflow = StateGraph(GraphState)

        workflow.add_node(
            "retrieve_and_generate",
            self.retrieve_and_generate,
        )

        workflow.set_entry_point("retrieve_and_generate")

        workflow.add_edge(
            "retrieve_and_generate",
            END,
        )

        return workflow.compile()

    def invoke(self, question: str):
        """
        Execute the LangGraph workflow.
        """

        return self.graph.invoke(
            {
                "question": question,
                "answer": "",
                "retrieved_chunks": [],
            }
        )