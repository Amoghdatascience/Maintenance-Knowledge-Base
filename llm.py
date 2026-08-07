"""
llm.py

Shared local LLM factory.

Every LLM call in this project goes to a local Ollama model. There is no cloud
model provider anywhere in the pipeline.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_ollama import ChatOllama

from config import RAGConfig


THINKING_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def build_llm(config: RAGConfig) -> ChatOllama:
    """
    Build the local chat model.

    reasoning=False disables qwen3's thinking mode. Left on, the model emits
    <think> blocks that corrupt structured-output parsing and roughly triple
    latency for what is mostly classification work.

    num_ctx must be raised explicitly: Ollama defaults to 4096 tokens, which is
    smaller than a full evidence prompt (max_evidence_chars alone is ~3k
    tokens). Left at the default, prompts are silently truncated and the judge
    grades evidence it never saw.
    """
    return ChatOllama(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        temperature=0,
        reasoning=False,
        num_ctx=config.ollama_num_ctx,
        num_predict=config.ollama_num_predict,
    )


def warm_up(llm) -> None:
    """
    Force the model into memory before the first real question.

    Ollama loads ~5.6GB on first invoke. Without this the first question pays
    a two-minute penalty that looks like a hang.
    """
    try:
        llm.invoke("ok")
    except Exception as error:  # noqa: BLE001 - warming is best-effort
        print(f"[llm] warm-up skipped: {error}")


def message_to_text(message: Any) -> str:
    """
    Convert a LangChain message to plain text, dropping any thinking block.
    """
    text = str(getattr(message, "content", message))
    return THINKING_BLOCK.sub("", text).strip()
