"""
nodes.py

LangGraph nodes for the planner-driven Agentic RAG workflow.

Nodes stay thin: planning lives in planner.py, retrieval in tools.py, web
research in research.py. This file wires them to state.

Note on containment: build_reflection_prompt and build_answer_prompt are
module-level and accept only (question, evidence). They have no way to receive
state["research_notes"], which makes the planning-only rule for Tavily
structural rather than a convention. See research.py.
"""

from __future__ import annotations

import hashlib
from typing import List, Literal, Sequence

from pydantic import BaseModel, Field

from config import RAGConfig
from csv_lookup import CSVKnowledgeBase
from graph_state import AgenticRAGState, EvidenceItem
from llm import build_llm, message_to_text
from planner import Planner, step_key
from research import WebResearcher
from retriever import VectorRetriever
from taxonomy import load_taxonomy
from tools import ToolExecutor


class Verdict(BaseModel):
    """
    Structured output for the reflection judge.
    """

    status: Literal["SUFFICIENT", "INSUFFICIENT"] = "INSUFFICIENT"
    reason: str = Field(default="", description="One short sentence.")
    gaps: List[str] = Field(
        default_factory=list,
        description="What is still missing. Empty when sufficient.",
    )


# ----------------------------------------------------------------------
# Prompt builders (question + evidence only, by design)
# ----------------------------------------------------------------------


# Excerpt length used when grading. The judge decides whether evidence is
# on-topic, which an excerpt answers as well as the full text -- and prompt
# processing costs ~1.7s per 1k characters, paid on every call.
GRADING_EXCERPT_CHARS = 400


def format_evidence(
    evidence: Sequence[EvidenceItem],
    max_chars_per_item: int = 0,
) -> str:
    """
    Render accumulated evidence for a prompt, preserving provenance.

    max_chars_per_item truncates each item; 0 means full text.
    """
    if not evidence:
        return "No evidence was retrieved."

    blocks = []

    for index, item in enumerate(evidence, start=1):
        header = f"Evidence {index} (via {item['tool']}, source: {item['source']}"

        if item["cluster"]:
            header += f", cluster: {item['cluster']}"
        elif item["parent"]:
            header += f", group: {item['parent']}"

        content = item["content"]

        if max_chars_per_item and len(content) > max_chars_per_item:
            content = content[:max_chars_per_item] + " [...]"

        blocks.append(f"{header})\n{content}")

    return "\n\n".join(blocks)


def build_reflection_prompt(question: str, evidence: Sequence[EvidenceItem]) -> str:
    """
    Prompt for judging evidence sufficiency.

    Evidence is excerpted: relevance is judged from the opening of a document,
    and sending full text here would double the cost of every grading call.
    """
    return f"""You are grading retrieved evidence for an aircraft maintenance
cluster knowledge base.

Question:
{question}

Evidence:
{format_evidence(evidence, max_chars_per_item=GRADING_EXCERPT_CHARS)}

Decide whether this evidence is enough to answer the question.

Rules:
- SUFFICIENT if the evidence contains what is needed to answer.
- For summary questions, SUFFICIENT if relevant entries for the requested group
  or topic are present.
- For pattern questions, SUFFICIENT if examples, retrieval keywords, inclusion
  rules, or summaries for the requested group are present.
- For comparison questions, SUFFICIENT if there is evidence about each item
  being compared.
- The evidence does not need to already contain the finished summary, pattern
  analysis, or comparison.
- Never treat one cluster as a substitute for another, and never assume one
  cluster is "likely" another.
- If the question names a specific cluster ID, that exact ID must appear in the
  evidence.
- If the evidence is related but not enough, answer INSUFFICIENT and say what
  is missing in gaps.
"""


def build_answer_prompt(
    question: str,
    evidence: Sequence[EvidenceItem],
    partial: bool = False,
) -> str:
    """
    Prompt for the final answer.

    Grounded strictly in the supplied evidence.
    """
    prompt = f"""You are answering questions about an aircraft maintenance
cluster knowledge base.

Question:
{question}

Evidence:
{format_evidence(evidence)}

Rules:
- Use only the evidence above. Do not add outside facts.
- Never infer that one cluster is "likely" another cluster.
- Never substitute a similar cluster for the one that was asked about.
- If a requested cluster ID does not appear in the evidence, say the evidence
  is insufficient for it.
- For a summary, synthesize the entries clearly.
- For patterns, group the common problem, action, and symptom patterns.
- For a comparison, compare the items using the evidence for each.
- Give a clear, direct answer.
- Name the sources you used.

Be concise. Answer in at most 200 words, using short bullets rather than
long prose. Do not restate the question or pad the answer with headings.
"""

    if partial:
        prompt += """
The evidence was judged incomplete and no further retrieval is possible.
Answer what the evidence does support, then state plainly what is missing.
"""

    return prompt


class AgenticRAGNodes:
    """
    Node container for the workflow.
    """

    def __init__(self, config: RAGConfig):
        self.config = config

        self.taxonomy = load_taxonomy(config.cluster_taxonomy_path)
        self.knowledge_base = CSVKnowledgeBase(config)
        self.retriever = VectorRetriever(config)
        self.llm = build_llm(config)

        self.planner = Planner(config, self.llm, self.taxonomy)
        self.researcher = WebResearcher(config, self.llm)
        self.executor = ToolExecutor(
            config,
            self.taxonomy,
            self.knowledge_base,
            self.retriever,
        )

    # ------------------------------------------------------------------
    # Triage
    # ------------------------------------------------------------------

    def triage(self, state: AgenticRAGState) -> dict:
        """
        Decompose the question and decide whether research is warranted.
        """
        result = self.planner.triage(state["question"])

        return {
            "sub_questions": result.sub_questions,
            "needs_external_research": result.needs_external_research,
        }

    def decide_research(self, state: AgenticRAGState) -> str:
        """
        Conditional edge after triage.
        """
        if state["needs_external_research"] and self.researcher.enabled:
            return "research"

        return "build_plan"

    # ------------------------------------------------------------------
    # Research (planning-only)
    # ------------------------------------------------------------------

    def research(self, state: AgenticRAGState) -> dict:
        """
        Gather planning keywords from the web. Never fails the run.
        """
        notes = self.researcher.research(state["question"])

        if notes:
            print(f"[research] planning keywords: {', '.join(notes)}")

        return {"research_notes": notes, "research_done": True}

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def build_plan(self, state: AgenticRAGState) -> dict:
        """
        Build the evidence plan for this pass.
        """
        unknown = self.taxonomy.unknown_clusters(state["question"])

        if unknown:
            # Nothing can ground an answer about a cluster that does not exist,
            # so skip retrieval entirely; reflect reports why.
            print(f"[plan] skipped: {', '.join(unknown)} not in the taxonomy")
            return {"plan": []}

        # Clusters may be scoped to only if they appear in the question or in
        # evidence already gathered, so a replan can legitimately narrow to
        # clusters that metadata_search just discovered.
        grounding_text = state["question"] + " " + " ".join(
            item["content"] for item in state["evidence"]
        )

        plan = self.planner.build_plan(
            question=state["question"],
            sub_questions=state["sub_questions"],
            research_notes=state["research_notes"],
            gaps=state["gaps"],
            executed_steps=state["executed_steps"],
            grounding_text=grounding_text,
        )

        for step in plan:
            scope = ", ".join(step["cluster_ids"] + step["parent_groups"]) or "unscoped"
            print(f"[plan] {step['tool']} ({scope}): {step['sub_question']}")

        return {"plan": plan}

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_tools(self, state: AgenticRAGState) -> dict:
        """
        Run every step of the plan and collect this pass's raw evidence.
        """
        pending: List[EvidenceItem] = []
        executed = list(state["executed_steps"])

        for step in state["plan"]:
            items = self.executor.execute(step, state["research_notes"])
            pending.extend(items)

            key = step_key(step)

            if key not in executed:
                executed.append(key)

            print(f"[execute] {step['tool']} returned {len(items)} item(s)")

        return {"pending_evidence": pending, "executed_steps": executed}

    def accumulate(self, state: AgenticRAGState) -> dict:
        """
        Merge new evidence into the pool: deduplicate, then cap the total size.

        Deduplication matters because data/ has overlapping documents for some
        clusters (cluster_c_10.txt and cluster_c10_baffle_spring.txt).
        """
        merged = list(state["evidence"])
        seen = {self._fingerprint(item) for item in merged}

        for item in state["pending_evidence"]:
            fingerprint = self._fingerprint(item)

            if fingerprint in seen:
                continue

            seen.add(fingerprint)
            merged.append(item)

        capped: List[EvidenceItem] = []
        budget = self.config.max_evidence_chars

        for item in merged:
            length = len(item["content"])

            if length > budget:
                continue

            capped.append(item)
            budget -= length

        if len(capped) < len(merged):
            print(f"[accumulate] kept {len(capped)}/{len(merged)} items within budget")

        return {"evidence": capped, "pending_evidence": []}

    @staticmethod
    def _fingerprint(item: EvidenceItem) -> str:
        digest = hashlib.sha1(item["content"][:200].encode()).hexdigest()
        return f"{item['source']}|{digest}"

    # ------------------------------------------------------------------
    # Reflection
    # ------------------------------------------------------------------

    def reflect(self, state: AgenticRAGState) -> dict:
        """
        Judge evidence sufficiency.

        A deterministic pre-check runs first: if the question names cluster IDs
        that are absent from the evidence, the answer cannot be grounded, so
        there is no point paying for an LLM call to discover that.
        """
        question = state["question"]
        evidence = state["evidence"]
        iterations = state["iterations"] + 1

        # A cluster ID that is not in the taxonomy cannot ever be answered, and
        # no amount of replanning will help. Say so instead of letting the
        # judge reason over whatever unscoped retrieval happened to return --
        # the validator strips invalid IDs from plan steps, so those searches
        # run unscoped and surface unrelated clusters.
        unknown = self.taxonomy.unknown_clusters(question)

        if unknown:
            listed = ", ".join(unknown)

            return {
                "verdict": "INSUFFICIENT",
                "verdict_reason": (
                    f"{listed} is not a cluster in this knowledge base. "
                    f"Valid cluster IDs run c_0 to c_{len(self.taxonomy.clusters) - 1}."
                ),
                "gaps": [f"{listed} does not exist in the taxonomy."],
                "iterations": iterations,
                "terminal": True,
            }

        if not evidence:
            return {
                "verdict": "INSUFFICIENT",
                "verdict_reason": "No evidence was retrieved.",
                "gaps": ["No evidence was retrieved for this question."],
                "iterations": iterations,
            }

        asked_clusters = self.taxonomy.match_clusters(question)

        if asked_clusters:
            covered = {item["cluster"] for item in evidence if item["cluster"]}
            blob = " ".join(item["content"] for item in evidence).lower()

            missing = [
                cluster_id
                for cluster_id in asked_clusters
                if cluster_id not in covered and cluster_id not in blob
            ]

            if missing:
                return {
                    "verdict": "INSUFFICIENT",
                    "verdict_reason": (
                        "Evidence does not mention "
                        f"{', '.join(missing)}, which the question asks about."
                    ),
                    "gaps": [f"Evidence for cluster {c}" for c in missing],
                    "iterations": iterations,
                }

        print(f"[reflect] grading {len(evidence)} item(s)...", flush=True)

        try:
            result = self.llm.with_structured_output(Verdict).invoke(
                build_reflection_prompt(question, evidence)
            )
        except Exception as error:  # noqa: BLE001
            # Evidence exists but grading failed; answer from it rather than
            # discarding a good retrieval over a parsing problem.
            print(f"[reflect] grading failed, treating evidence as usable: {error}")
            return {
                "verdict": "SUFFICIENT",
                "verdict_reason": "Grading unavailable; answering from evidence.",
                "gaps": [],
                "iterations": iterations,
            }

        status = result.status if result.status in {"SUFFICIENT", "INSUFFICIENT"} else "INSUFFICIENT"

        print(f"[reflect] {status}", flush=True)

        return {
            "verdict": status,
            "verdict_reason": result.reason or "",
            "gaps": [gap for gap in (result.gaps or []) if gap.strip()],
            "iterations": iterations,
        }

    def decide_after_reflection(self, state: AgenticRAGState) -> str:
        """
        Conditional edge after reflect: answer, or replan once more.
        """
        if state["verdict"] == "SUFFICIENT":
            return "generate_answer"

        if state["terminal"]:
            return "generate_answer"

        if state["iterations"] > self.config.max_iterations:
            print("[reflect] iteration budget exhausted, answering with what we have")
            return "generate_answer"

        return "build_plan"

    # ------------------------------------------------------------------
    # Answer
    # ------------------------------------------------------------------

    def generate_answer(self, state: AgenticRAGState) -> dict:
        """
        Produce the final answer from accumulated evidence.
        """
        evidence = state["evidence"]

        if state["terminal"]:
            # Whatever evidence was gathered is about something the question
            # did not ask for, so reporting the reason is the honest answer.
            return {"answer": state["verdict_reason"]}

        if not evidence:
            return {
                "answer": (
                    "The retrieved evidence is insufficient to answer this "
                    f"question.\n\nReason: {state['verdict_reason']}"
                )
            }

        partial = state["verdict"] != "SUFFICIENT"

        print(
            f"[generate] writing answer from {len(evidence)} item(s)"
            f"{' (partial)' if partial else ''}...",
            flush=True,
        )

        try:
            response = self.llm.invoke(
                build_answer_prompt(state["question"], evidence, partial=partial)
            )
        except Exception as error:  # noqa: BLE001
            print(f"[generate] answer generation failed: {error}")
            return {
                "answer": (
                    "Evidence was retrieved but the answer could not be "
                    f"generated.\n\nError: {error}"
                )
            }

        return {"answer": message_to_text(response)}
