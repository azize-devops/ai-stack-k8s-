"""Node functions for the LangGraph query-routing workflow.

Each public function in this module corresponds to a node in the state
graph.  Every node receives the current ``WorkflowState`` dict and
returns a partial state update.

All LLM calls go through ``langchain_openai.ChatOpenAI`` with a
configurable ``base_url`` so the same code works with LocalAI, Ollama,
or the OpenAI API.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from state import WorkflowState

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared LLM helper
# ---------------------------------------------------------------------------


def _get_llm() -> ChatOpenAI:
    """Build a ``ChatOpenAI`` instance from environment variables."""
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo"),
        base_url=os.getenv("LLM_BASE_URL", "http://localai:8080/v1"),
        api_key=os.getenv("LLM_API_KEY", "sk-no-key-required"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        request_timeout=int(os.getenv("LLM_TIMEOUT", "120")),
    )


# ---------------------------------------------------------------------------
# Node: classify_query
# ---------------------------------------------------------------------------

_CLASSIFICATION_PROMPT = """\
You are a query classifier. Analyse the user query and return a JSON object
with exactly two keys:

  "classification": one of "factual", "analytical", or "creative"
  "route": "rag" if the query would benefit from external context retrieval,
           otherwise "direct"

Definitions:
- factual: the query asks for specific facts, data, or definitions.
- analytical: the query asks for comparison, evaluation, or reasoning.
- creative: the query asks for brainstorming, generation, or open-ended ideas.

Return ONLY the JSON object, nothing else.
"""


def classify_query(state: WorkflowState) -> dict[str, Any]:
    """Classify the user query and determine the routing decision.

    Returns partial state with ``classification`` and ``route`` keys.
    """
    query: str = state["query"]
    logger.info("[classify_query] Classifying: %.120s", query)

    llm = _get_llm()
    messages = [
        SystemMessage(content=_CLASSIFICATION_PROMPT),
        HumanMessage(content=query),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Parse JSON from the LLM response, handling potential markdown fences.
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        classification = parsed.get("classification", "factual")
        route = parsed.get("route", "direct")
    except (json.JSONDecodeError, AttributeError):
        logger.warning(
            "[classify_query] Failed to parse LLM JSON; falling back to defaults. Raw: %s",
            raw,
        )
        classification = "factual"
        route = "direct"

    # Normalise to allowed values.
    if classification not in {"factual", "analytical", "creative"}:
        classification = "factual"
    if route not in {"rag", "direct"}:
        route = "direct"

    logger.info(
        "[classify_query] classification=%s route=%s",
        classification,
        route,
    )

    return {"classification": classification, "route": route}


# ---------------------------------------------------------------------------
# Node: retrieve_context
# ---------------------------------------------------------------------------


def _retrieve_from_qdrant(query: str) -> list[str]:
    """Attempt to retrieve context from the Qdrant vector store.

    If the Qdrant instance is unreachable or the collection does not
    exist, the function returns an empty list and logs a warning so the
    pipeline can continue gracefully.
    """
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    collection = os.getenv("QDRANT_COLLECTION", "documents")

    try:
        # Optional import -- if qdrant-client is not installed, fall back.
        from qdrant_client import QdrantClient  # type: ignore[import-untyped]

        client = QdrantClient(url=qdrant_url, timeout=10)

        # Quick health check
        collections = [c.name for c in client.get_collections().collections]
        if collection not in collections:
            logger.warning(
                "[retrieve_context] Collection '%s' not found in Qdrant. "
                "Available: %s",
                collection,
                collections,
            )
            return []

        # Use scroll to get a small set of documents (simple keyword-like approach).
        # In production you would encode the query via an embedding model and use
        # client.search() with a vector.
        results = client.scroll(
            collection_name=collection,
            limit=5,
        )
        points, _ = results
        return [
            point.payload.get("text", str(point.payload))
            for point in points
            if point.payload
        ]

    except ImportError:
        logger.warning(
            "[retrieve_context] qdrant-client not installed; skipping Qdrant retrieval."
        )
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[retrieve_context] Qdrant retrieval failed (%s); continuing without context.",
            exc,
        )
        return []


def _simulate_context(query: str) -> list[str]:
    """Return simulated context for demonstration purposes.

    Used when Qdrant is unavailable or returns no results.
    """
    return [
        (
            f"Simulated context document 1: Background information relevant "
            f"to the query '{query[:80]}'. This would normally come from a "
            f"vector store such as Qdrant."
        ),
        (
            f"Simulated context document 2: Additional reference material "
            f"related to the topic. In production, documents are embedded and "
            f"retrieved via semantic similarity search."
        ),
        (
            f"Simulated context document 3: Supporting data and examples "
            f"that provide grounding for the LLM response."
        ),
    ]


def retrieve_context(state: WorkflowState) -> dict[str, Any]:
    """Retrieve supporting context for the user query.

    Tries Qdrant first; falls back to simulated context if unavailable.

    Returns partial state with the ``context`` key.
    """
    query: str = state["query"]
    logger.info("[retrieve_context] Retrieving context for: %.120s", query)

    context = _retrieve_from_qdrant(query)

    if not context:
        logger.info("[retrieve_context] Using simulated context (Qdrant unavailable or empty).")
        context = _simulate_context(query)

    logger.info("[retrieve_context] Retrieved %d context chunks.", len(context))
    return {"context": context}


# ---------------------------------------------------------------------------
# Node: generate_response
# ---------------------------------------------------------------------------

_RESPONSE_SYSTEM_PROMPT = """\
You are a helpful, accurate, and thorough AI assistant. Answer the user's
question clearly and completely. If context documents are provided, use them
to ground your answer. Always be honest about uncertainty.
"""


def generate_response(state: WorkflowState) -> dict[str, Any]:
    """Generate the final response, optionally using retrieved context.

    Returns partial state with the ``response`` key.
    """
    query: str = state["query"]
    context: list[str] = state.get("context", [])
    classification: str = state.get("classification", "factual")

    logger.info(
        "[generate_response] Generating response (classification=%s, context_chunks=%d)",
        classification,
        len(context),
    )

    llm = _get_llm()

    # Build user message with optional context.
    user_parts: list[str] = []
    if context:
        formatted_ctx = "\n\n---\n\n".join(context)
        user_parts.append(
            f"Use the following context to help answer the question:\n\n{formatted_ctx}\n\n---\n"
        )
    user_parts.append(f"Question: {query}")

    messages = [
        SystemMessage(content=_RESPONSE_SYSTEM_PROMPT),
        HumanMessage(content="\n".join(user_parts)),
    ]

    response = llm.invoke(messages)
    answer = response.content.strip()

    logger.info("[generate_response] Generated response (%d chars).", len(answer))
    return {"response": answer}


# ---------------------------------------------------------------------------
# Node: check_quality
# ---------------------------------------------------------------------------

_QUALITY_PROMPT = """\
You are a strict quality evaluator. Given a question and an answer, rate the
answer quality on a scale of 0-100.  Consider:
  - Relevance to the question
  - Accuracy (no hallucinated facts)
  - Completeness
  - Clarity

Respond with ONLY a JSON object: {"score": <int>, "passed": <bool>}
where passed is true if score >= 60.
"""


def check_quality(state: WorkflowState) -> dict[str, Any]:
    """Validate the quality of the generated response.

    Returns partial state with ``quality_score`` and ``quality_passed``.
    """
    query: str = state["query"]
    response: str = state.get("response", "")

    logger.info("[check_quality] Evaluating response quality ...")

    llm = _get_llm()
    messages = [
        SystemMessage(content=_QUALITY_PROMPT),
        HumanMessage(content=f"Question: {query}\n\nAnswer: {response}"),
    ]

    raw_output = llm.invoke(messages).content.strip()

    # Parse quality score.
    if raw_output.startswith("```"):
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]
        raw_output = raw_output.strip()

    try:
        parsed = json.loads(raw_output)
        score = int(parsed.get("score", 70))
        passed = bool(parsed.get("passed", score >= 60))
    except (json.JSONDecodeError, ValueError, AttributeError):
        logger.warning(
            "[check_quality] Could not parse quality JSON; assigning default score. Raw: %s",
            raw_output,
        )
        score = 70
        passed = True

    logger.info("[check_quality] score=%d passed=%s", score, passed)
    return {"quality_score": score, "quality_passed": passed}
