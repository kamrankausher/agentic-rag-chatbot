from typing import TypedDict

from langgraph.graph import StateGraph, END

from app.rag import RAGPipeline


class GraphState(TypedDict):
    question: str
    context: str
    retrieved_chunks: list
    answer: str


class AgenticRAGGraph:
    """
    Multi-node LangGraph workflow.
    """

    def __init__(self):
        self.rag = RAGPipeline()
        self.graph = self._build_graph()

    def retrieve_node(self, state: GraphState):
        """
        Retrieve relevant documents.
        """

        retrieval = self.rag.retrieve(state["question"])

        return {
            "context": retrieval["context"],
            "retrieved_chunks": retrieval["retrieved_chunks"],
        }

    def generate_node(self, state: GraphState):
        """
        Generate grounded answer.
        """

        answer = self.rag.generate(
            question=state["question"],
            context=state["context"],
        )

        return {
            "answer": answer,
        }

    def _build_graph(self):

        workflow = StateGraph(GraphState)

        workflow.add_node(
            "retrieve",
            self.retrieve_node,
        )

        workflow.add_node(
            "generate",
            self.generate_node,
        )

        workflow.set_entry_point("retrieve")

        workflow.add_edge(
            "retrieve",
            "generate",
        )

        workflow.add_edge(
            "generate",
            END,
        )

        return workflow.compile()

    def invoke(self, question: str):

        return self.graph.invoke(
            {
                "question": question,
                "context": "",
                "retrieved_chunks": [],
                "answer": "",
            }
        )