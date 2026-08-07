"""
graph_state.py

State definition for the planner-driven Agentic RAG workflow.
"""

from __future__ import annotations

from typing import List, TypedDict


class EvidenceItem(TypedDict):
    """
    A single piece of retrieved evidence, carrying its provenance.

    Provenance is kept so generate_answer can name the sources it used and so
    the accumulator can deduplicate overlapping documents.
    """

    tool: str  # csv_lookup | vector_search | metadata_search
    sub_question: str
    content: str
    source: str  # filename, or the CSV path for structured lookups
    cluster: str  # "" when the evidence is not cluster-scoped
    parent: str  # "" when the evidence is not parent-group-scoped


class PlanStep(TypedDict):
    """
    One executable step of an evidence plan.

    This is the validated, plain-dict form of planner.EvidenceStep. The planner
    never emits a retrieval query string; it names a tool plus entities, and
    tools.py builds the actual query.
    """

    tool: str
    sub_question: str
    cluster_ids: List[str]
    parent_groups: List[str]
    keywords: List[str]


class AgenticRAGState(TypedDict):
    """
    Shared state passed between LangGraph nodes.
    """

    question: str

    # Triage output.
    sub_questions: List[str]
    needs_external_research: bool

    # Tavily output. PLANNER-ONLY.
    #
    # research_notes is a bounded list of domain keywords derived from web
    # research. It is read in exactly two places: the build_plan prompt, and as
    # extra retrieval terms in tools.vector_search. It must never reach the
    # reflect or generate_answer prompts -- answers are grounded only in the
    # local cluster knowledge base. See research.py for the full contract.
    research_notes: List[str]
    research_done: bool

    # Planning and execution.
    plan: List[PlanStep]
    executed_steps: List[str]  # step keys already run, so replans cannot repeat
    pending_evidence: List[EvidenceItem]  # this pass's raw hits, pre-merge
    evidence: List[EvidenceItem]

    # Reflection.
    verdict: str  # SUFFICIENT | INSUFFICIENT
    verdict_reason: str
    gaps: List[str]
    iterations: int

    # Set when the question can never be answered no matter how many times we
    # replan (for example it names a cluster ID that does not exist). Stops the
    # loop immediately and reports the reason instead of burning the budget.
    terminal: bool

    answer: str


def initial_state(question: str) -> AgenticRAGState:
    """
    Build the starting state for a question.
    """
    return {
        "question": question,
        "sub_questions": [],
        "needs_external_research": False,
        "research_notes": [],
        "research_done": False,
        "plan": [],
        "executed_steps": [],
        "pending_evidence": [],
        "evidence": [],
        "verdict": "",
        "verdict_reason": "",
        "gaps": [],
        "iterations": 0,
        "terminal": False,
        "answer": "",
    }
