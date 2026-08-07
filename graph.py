"""
graph.py

LangGraph workflow for the planner-driven Agentic RAG app.

Flow:

    triage ──needs research?──yes──> research ─┐
       └──────────────────────no──────────────┴─> build_plan
    build_plan -> execute_tools -> accumulate -> reflect
                       ^                            |
                       |                   SUFFICIENT -> generate_answer -> END
                       |                            |
                       └──── replan (gaps) ─── INSUFFICIENT and budget left
                                                    |
                                    budget spent -> generate_answer -> END

triage and build_plan are separate nodes so research lands between question
decomposition and plan construction.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from config import RAGConfig
from graph_state import AgenticRAGState, initial_state
from nodes import AgenticRAGNodes


class AgenticRAGGraph:
    """
    Builds and runs the LangGraph workflow.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.nodes = AgenticRAGNodes(config)
        self.app = self._build_graph()

    def _build_graph(self):
        """
        Build the LangGraph workflow.
        """
        workflow = StateGraph(AgenticRAGState)

        workflow.add_node("triage", self.nodes.triage)
        workflow.add_node("research", self.nodes.research)
        workflow.add_node("build_plan", self.nodes.build_plan)
        workflow.add_node("execute_tools", self.nodes.execute_tools)
        workflow.add_node("accumulate", self.nodes.accumulate)
        workflow.add_node("reflect", self.nodes.reflect)
        workflow.add_node("generate_answer", self.nodes.generate_answer)

        workflow.set_entry_point("triage")

        workflow.add_conditional_edges(
            "triage",
            self.nodes.decide_research,
            {
                "research": "research",
                "build_plan": "build_plan",
            },
        )

        workflow.add_edge("research", "build_plan")
        workflow.add_edge("build_plan", "execute_tools")
        workflow.add_edge("execute_tools", "accumulate")
        workflow.add_edge("accumulate", "reflect")

        workflow.add_conditional_edges(
            "reflect",
            self.nodes.decide_after_reflection,
            {
                "generate_answer": "generate_answer",
                "build_plan": "build_plan",
            },
        )

        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def ask(self, question: str) -> dict:
        """
        Ask a question.

        Args:
            question: User question.

        Returns:
            Final LangGraph state.
        """
        return self.app.invoke(initial_state(question))
