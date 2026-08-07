"""
planner.py

Task decomposition and evidence planning.

Design: the planner is a *constrained selector*, not a generator. The model
picks a tool from a closed enum and names entities; it never writes a retrieval
query string. Every entity it names is validated against cluster_taxonomy.csv,
so a hallucinated cluster ID is dropped before it can reach a tool.

A deterministic plan is always computed first and used as the floor. If the
model errors, returns nothing, or returns only invalid steps, the deterministic
plan is used instead -- so the workflow is never worse than the keyword routing
it replaces.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence

from pydantic import BaseModel, Field

from config import RAGConfig
from csv_lookup import CSVKnowledgeBase
from graph_state import PlanStep
from taxonomy import Taxonomy


TOOLS = ("csv_lookup", "vector_search", "metadata_search")


# ----------------------------------------------------------------------
# Structured output schemas
# ----------------------------------------------------------------------


class Triage(BaseModel):
    """
    First planner pass: decompose the question and decide on research.
    """

    sub_questions: List[str] = Field(
        default_factory=list,
        description="The question broken into 1-3 answerable parts.",
    )
    needs_external_research: bool = Field(
        default=False,
        description=(
            "True only if answering requires general aircraft maintenance "
            "domain knowledge that a cluster knowledge base would not contain."
        ),
    )


class EvidenceStep(BaseModel):
    """
    One tool call the planner wants executed.
    """

    tool: Literal["csv_lookup", "vector_search", "metadata_search"]
    sub_question: str = Field(description="What this step should find out.")
    cluster_ids: List[str] = Field(
        default_factory=list,
        description="Cluster IDs copied verbatim from the question, e.g. c_38.",
    )
    parent_groups: List[str] = Field(
        default_factory=list,
        description="Parent group names, chosen only from the allowed list.",
    )
    keywords: List[str] = Field(default_factory=list)


class EvidencePlan(BaseModel):
    """
    An ordered set of evidence-gathering steps.
    """

    steps: List[EvidenceStep] = Field(default_factory=list)


# ----------------------------------------------------------------------
# Keyword signals, carried over from the routing this replaces
# ----------------------------------------------------------------------

STRUCTURED_SIGNALS = (
    "how many",
    "count",
    "number of",
    "total",
    "list all",
    "show all",
    "records",
    "record",
    "raw",
    "work order",
    "work orders",
    "workorder",
    "workorders",
    "registration",
    "tail number",
    "columns",
    "schema",
    "ata",
    "contains",
    "contain",
    "mentions",
    "mention",
)

EXPLANATORY_SIGNALS = (
    "explain",
    "describe",
    "summarize",
    "summary",
    "what does",
    "common problem",
    "common action",
    "common terms",
    "pattern",
    "patterns",
    "symptom",
    "symptoms",
    "corrective action",
    "corrective actions",
    "inclusion guidance",
    "why",
    "what kind of",
    "give examples",
    "meaning",
)

COMPARISON_SIGNALS = (
    "compare",
    "comparison",
    "difference between",
    "differences between",
    "differentiate",
    "versus",
    " vs ",
    "similarities",
    "similar and different",
)


def step_key(step: PlanStep) -> str:
    """
    Stable identity for a step, so replans cannot repeat the same retrieval.
    """
    return "|".join(
        [
            step["tool"],
            ",".join(sorted(step["cluster_ids"])),
            ",".join(sorted(step["parent_groups"])),
            step["sub_question"].strip().lower(),
        ]
    )


def _make_step(
    tool: str,
    sub_question: str,
    cluster_ids: Optional[Sequence[str]] = None,
    parent_groups: Optional[Sequence[str]] = None,
    keywords: Optional[Sequence[str]] = None,
) -> PlanStep:
    return {
        "tool": tool,
        "sub_question": sub_question.strip(),
        "cluster_ids": list(cluster_ids or []),
        "parent_groups": list(parent_groups or []),
        "keywords": list(keywords or []),
    }


class Planner:
    """
    Builds evidence plans, with a deterministic floor under the LLM.
    """

    def __init__(self, config: RAGConfig, llm, taxonomy: Taxonomy):
        self.config = config
        self.llm = llm
        self.taxonomy = taxonomy

    # ------------------------------------------------------------------
    # Triage
    # ------------------------------------------------------------------

    def triage(self, question: str) -> Triage:
        """
        Decompose the question and decide whether web research would help.

        Falls back to a single sub-question with no research on any failure.
        """
        prompt = f"""You plan evidence gathering for an aircraft maintenance
cluster knowledge base. The knowledge base contains maintenance work-order
records and one document per maintenance issue cluster.

Question:
{question}

Break the question into at most {self.config.max_sub_questions} self-contained
sub-questions. A simple question stays as one sub-question -- do not invent
extra parts.

Set needs_external_research to true ONLY if answering needs general aircraft
maintenance domain knowledge that would not appear in a cluster knowledge base
(for example, what a component physically does, or standard terminology).
Set it to false for questions about counts, records, cluster contents, or
comparisons between clusters.
"""

        try:
            result = self.llm.with_structured_output(Triage).invoke(prompt)
        except Exception as error:  # noqa: BLE001 - local model, any failure falls back
            print(f"[planner] triage failed, using question as-is: {error}")
            return Triage(sub_questions=[question], needs_external_research=False)

        sub_questions = [
            text.strip()
            for text in (result.sub_questions or [])
            if text and text.strip()
        ][: self.config.max_sub_questions]

        if not sub_questions:
            sub_questions = [question]

        return Triage(
            sub_questions=sub_questions,
            needs_external_research=bool(result.needs_external_research),
        )

    # ------------------------------------------------------------------
    # Deterministic plan (the floor)
    # ------------------------------------------------------------------

    def deterministic_plan(self, question: str) -> List[PlanStep]:
        """
        Build a plan from keyword signals and taxonomy matching, with no LLM.

        This is the same intent as the keyword routing this workflow replaces,
        except it may now emit several steps instead of choosing one arm.
        """
        lowered = question.lower().strip()

        clusters = self.taxonomy.match_clusters(lowered)
        parents = self.taxonomy.match_parents(lowered)

        structured = any(signal in lowered for signal in STRUCTURED_SIGNALS)
        explanatory = any(signal in lowered for signal in EXPLANATORY_SIGNALS)
        comparison = any(signal in lowered for signal in COMPARISON_SIGNALS)

        steps: List[PlanStep] = []

        if comparison and len(clusters) >= 2:
            # Taxonomy definitions first: some clusters (c_38, c_50) have no
            # document and no records, so this is their only evidence.
            steps.append(
                _make_step(
                    "metadata_search",
                    f"Definitions of {', '.join(clusters)}.",
                    cluster_ids=clusters,
                )
            )

            # Then fan out, so each side of the comparison is retrieved on its
            # own and cannot be crowded out by the other.
            for cluster_id in clusters:
                steps.append(
                    _make_step(
                        "vector_search",
                        f"What does {cluster_id} cover?",
                        cluster_ids=[cluster_id],
                    )
                )
        elif clusters:
            if structured or not explanatory:
                steps.append(
                    _make_step("csv_lookup", question, cluster_ids=clusters)
                )

            # Scoped by cluster metadata, so this cannot return a different
            # cluster's document -- the substitution risk that kept exact IDs
            # away from vector search before.
            steps.append(
                _make_step("vector_search", question, cluster_ids=clusters)
            )
        elif structured:
            steps.append(_make_step("csv_lookup", question))

        if parents and not clusters:
            steps.append(
                _make_step("metadata_search", question, parent_groups=parents)
            )
            steps.append(
                _make_step("vector_search", question, parent_groups=parents)
            )

        if not steps:
            steps.append(_make_step("vector_search", question))

        return steps[: self.config.max_plan_steps]

    # ------------------------------------------------------------------
    # LLM plan
    # ------------------------------------------------------------------

    def build_plan(
        self,
        question: str,
        sub_questions: Sequence[str],
        research_notes: Sequence[str],
        gaps: Sequence[str],
        executed_steps: Sequence[str],
        grounding_text: str = "",
    ) -> List[PlanStep]:
        """
        Ask the model for an evidence plan, validate it, and fall back.

        research_notes is read here and nowhere downstream -- see research.py.

        grounding_text is the question plus any evidence already gathered; it
        bounds which cluster IDs the plan may scope to.
        """
        baseline = self.deterministic_plan(question)
        grounding_text = grounding_text or question
        fallback_parents = self.taxonomy.match_parents(question)

        try:
            raw = self.llm.with_structured_output(EvidencePlan).invoke(
                self._plan_prompt(question, sub_questions, research_notes, gaps)
            )
            plan = self.validate_plan(
                raw, executed_steps, grounding_text, fallback_parents
            )
        except Exception as error:  # noqa: BLE001 - fall back rather than fail
            print(f"[planner] plan generation failed, using baseline: {error}")
            plan = []

        if not plan:
            plan = self.validate_steps(
                baseline, executed_steps, grounding_text, fallback_parents
            )

        if not plan:
            # Everything was already executed: retry the baseline unfiltered so
            # the graph always makes a retrieval attempt.
            plan = baseline

        return plan

    def _plan_prompt(
        self,
        question: str,
        sub_questions: Sequence[str],
        research_notes: Sequence[str],
        gaps: Sequence[str],
    ) -> str:
        parts = [
            """You plan evidence gathering for an aircraft maintenance cluster
knowledge base. Choose which tools to run. You do NOT answer the question.

Tools:
- csv_lookup: exact structured facts from maintenance records. Use for counts,
  totals, raw records, work orders, registrations, ATA codes, and schema.
- vector_search: narrative cluster documents. Use for meaning, explanations,
  patterns, symptoms, corrective actions, and comparisons.
- metadata_search: lists and definitions for a whole parent group. Use to find
  which clusters belong to a group before searching within it.

Rules:
- Emit at most %d steps. Prefer the fewest steps that answer the question.
- cluster_ids may ONLY contain IDs written literally in the question, copied
  exactly (for example c_38). Never invent or guess a cluster ID.
- parent_groups may ONLY be chosen from this list:
  %s
- Leave cluster_ids and parent_groups empty if the question names none.
- sub_question states what the step should find out, in plain words.

Examples:
- "How many records are in cluster c_22?" -> one csv_lookup step, cluster_ids=["c_22"]
- "Compare c_37 and c_38." -> two vector_search steps, one per cluster ID
- "What are common problems across baffle clusters?" -> metadata_search then
  vector_search, both with parent_groups=["baffle"]
"""
            % (self.config.max_plan_steps, ", ".join(self.taxonomy.parents)),
            f"Question:\n{question}",
        ]

        if sub_questions:
            joined = "\n".join(f"- {text}" for text in sub_questions)
            parts.append(f"Sub-questions:\n{joined}")

        if research_notes:
            parts.append(
                "Domain terms that may help retrieval (background only, not "
                "facts to report):\n" + ", ".join(research_notes)
            )

        if gaps:
            joined = "\n".join(f"- {text}" for text in gaps)
            parts.append(
                "A previous attempt was judged insufficient. Target these "
                f"gaps and do not repeat the same searches:\n{joined}"
            )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_plan(
        self,
        plan: EvidencePlan,
        executed_steps: Sequence[str] = (),
        grounding_text: str = "",
        fallback_parents: Optional[Sequence[str]] = None,
    ) -> List[PlanStep]:
        """
        Convert a model plan into validated steps.
        """
        candidates: List[PlanStep] = []

        for step in plan.steps or []:
            if step.tool not in TOOLS:
                continue

            candidates.append(
                _make_step(
                    step.tool,
                    step.sub_question or "",
                    cluster_ids=step.cluster_ids,
                    parent_groups=step.parent_groups,
                    keywords=step.keywords,
                )
            )

        return self.validate_steps(
            candidates, executed_steps, grounding_text, fallback_parents
        )

    def validate_steps(
        self,
        steps: Sequence[PlanStep],
        executed_steps: Sequence[str] = (),
        grounding_text: str = "",
        fallback_parents: Optional[Sequence[str]] = None,
    ) -> List[PlanStep]:
        """
        Drop invalid entities and already-executed steps, dedupe, and cap.

        grounding_text, when supplied, restricts cluster IDs to those actually
        mentioned in the question or in evidence gathered so far. Existence
        alone is not enough: asked for "engine clusters" the model happily
        names c_12-c_15, which are real clusters but belong to cowling and
        cylinder groups. Scoping a search to invented-but-real IDs silently
        searches the wrong documents, so the prompt rule is enforced here
        rather than trusted to the model.
        """
        already_run = set(executed_steps)
        grounded = (
            self.taxonomy.match_clusters(grounding_text) if grounding_text else None
        )
        seen: set = set()
        kept: List[PlanStep] = []

        for step in steps:
            cluster_ids = self.taxonomy.valid_clusters(step["cluster_ids"])
            parent_groups = self.taxonomy.valid_parents(step["parent_groups"])

            if grounded is not None:
                dropped = [c for c in cluster_ids if c not in grounded]

                if dropped:
                    print(
                        f"[planner] dropped ungrounded cluster ids: {', '.join(dropped)}"
                    )

                cluster_ids = [c for c in cluster_ids if c in grounded]

                # The model expressed scope through the wrong field. Recover it
                # from the question's parent groups rather than letting the step
                # fall back to an unscoped search over everything.
                if dropped and not cluster_ids and not parent_groups:
                    parent_groups = fallback_parents or []

                    if parent_groups:
                        print(
                            f"[planner] rescoped to group(s): {', '.join(parent_groups)}"
                        )

            cleaned = _make_step(
                step["tool"],
                step["sub_question"],
                cluster_ids=cluster_ids,
                parent_groups=parent_groups,
                keywords=[
                    keyword.strip()
                    for keyword in step["keywords"]
                    if keyword and keyword.strip()
                ][:8],
            )

            if not cleaned["sub_question"]:
                continue

            key = step_key(cleaned)

            if key in seen or key in already_run:
                continue

            seen.add(key)
            kept.append(cleaned)

            if len(kept) >= self.config.max_plan_steps:
                break

        return kept


def extract_structured_hints(question: str) -> dict:
    """
    Pull structured identifiers out of a question.

    Reuses the extractors already written for the CSV layer rather than
    duplicating the regexes here.
    """
    return {
        "workorders": CSVKnowledgeBase._extract_workorders(question),
        "registration": CSVKnowledgeBase._extract_registration(question.lower()),
        "ata_codes": CSVKnowledgeBase._extract_ata_codes(question),
        "keyword": CSVKnowledgeBase._extract_keyword_for_search(question.lower()),
    }
