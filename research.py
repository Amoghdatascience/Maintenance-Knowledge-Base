"""
research.py

Optional Tavily web research, used to inform PLANNING ONLY.

The contract, in full:

1. Tavily runs only when all three hold: triage asked for research, a
   TAVILY_API_KEY is set, and ENABLE_WEB_RESEARCH is not disabled. Missing key
   or missing package means the feature is simply off.
2. The only thing that ever leaves this module is a bounded list of short
   domain keywords -- never prose, never a claim, never a URL. The return type
   is List[str] and the post-filter enforces length and count caps.
3. Those keywords are consumed in exactly two places: the build_plan prompt
   (planner.Planner._plan_prompt) and as extra retrieval terms in
   tools.vector_search. They must never reach the reflect or generate_answer
   prompts, because answers are grounded only in the local knowledge base.
   test_workflow.py asserts this.
4. Every failure path is non-fatal. Any exception yields an empty list and the
   graph continues on the local-only path.

Because the output can only widen retrieval, a bad research result costs recall
tuning at worst -- it can never put an unsupported fact into an answer.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

from pydantic import BaseModel, Field

from config import RAGConfig


class DomainKeywords(BaseModel):
    """
    Structured extraction target: keywords only, deliberately not prose.
    """

    keywords: List[str] = Field(
        default_factory=list,
        description="Short technical terms, 1-3 words each.",
    )


class WebResearcher:
    """
    Gated Tavily client that yields planning keywords and nothing else.
    """

    def __init__(self, config: RAGConfig, llm):
        self.config = config
        self.llm = llm
        self._cache: Dict[str, List[str]] = {}
        self._client = None
        self._unavailable_reason = ""

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """
        Whether research can run at all.
        """
        if not self.config.enable_web_research:
            self._unavailable_reason = "ENABLE_WEB_RESEARCH is false"
            return False

        if not self.config.tavily_api_key:
            self._unavailable_reason = "TAVILY_API_KEY is not set"
            return False

        return True

    def _get_client(self):
        """
        Lazily construct the Tavily client. Import failure disables research.
        """
        if self._client is not None:
            return self._client

        try:
            from tavily import TavilyClient
        except ImportError:
            self._unavailable_reason = "tavily-python is not installed"
            return None

        try:
            self._client = TavilyClient(api_key=self.config.tavily_api_key)
        except Exception as error:  # noqa: BLE001
            self._unavailable_reason = f"Tavily client init failed: {error}"
            return None

        return self._client

    # ------------------------------------------------------------------
    # Research
    # ------------------------------------------------------------------

    def research(self, question: str) -> List[str]:
        """
        Return bounded planning keywords for a question. Never raises.
        """
        if not self.enabled:
            print(f"[research] skipped: {self._unavailable_reason}")
            return []

        cache_key = hashlib.sha1(question.strip().lower().encode()).hexdigest()

        if cache_key in self._cache:
            return list(self._cache[cache_key])

        raw_text = self._search(question)

        if not raw_text:
            self._cache[cache_key] = []
            return []

        keywords = self._extract_keywords(question, raw_text)
        self._cache[cache_key] = keywords

        return list(keywords)

    def _search(self, question: str) -> str:
        """
        Run the search and return truncated raw text, or "" on any failure.
        """
        client = self._get_client()

        if client is None:
            print(f"[research] skipped: {self._unavailable_reason}")
            return ""

        query = f"aircraft maintenance {question}"

        kwargs = {
            "query": query,
            "max_results": self.config.tavily_max_results,
            "search_depth": "basic",
        }

        try:
            try:
                response = client.search(
                    timeout=self.config.tavily_timeout_seconds,
                    **kwargs,
                )
            except TypeError:
                # Older clients do not accept a timeout argument.
                response = client.search(**kwargs)
        except Exception as error:  # noqa: BLE001
            print(f"[research] search failed, continuing locally: {error}")
            return ""

        contents = [
            str(result.get("content", ""))
            for result in (response or {}).get("results", [])
        ]

        return " ".join(text for text in contents if text)[
            : self.config.tavily_max_chars
        ]

    def _extract_keywords(self, question: str, raw_text: str) -> List[str]:
        """
        Reduce search text to a bounded keyword list. Never returns prose.
        """
        prompt = f"""Extract technical aircraft maintenance terms from the text
below that would help search a maintenance knowledge base.

Question:
{question}

Text:
{raw_text}

Return ONLY short technical terms of 1-3 words each, such as component names,
failure modes, and symptoms. Do not return sentences, explanations, claims,
numbers, or URLs. Return at most {self.config.research_max_keywords} terms.
"""

        try:
            result = self.llm.with_structured_output(DomainKeywords).invoke(prompt)
        except Exception as error:  # noqa: BLE001
            print(f"[research] keyword extraction failed: {error}")
            return []

        return self._filter_keywords(result.keywords or [])

    def _filter_keywords(self, keywords: Sequence[str]) -> List[str]:
        """
        Enforce the output caps: short, lowercase, unique, bounded in count.
        """
        kept: List[str] = []

        for keyword in keywords:
            cleaned = " ".join(str(keyword).strip().lower().split())

            if not cleaned:
                continue

            if len(cleaned) > self.config.research_keyword_max_len:
                continue

            if cleaned in kept:
                continue

            kept.append(cleaned)

            if len(kept) >= self.config.research_max_keywords:
                break

        return kept
