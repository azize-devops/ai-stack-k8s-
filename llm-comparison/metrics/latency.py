"""
Latency measurement utilities for LLM API calls.

Provides functions to measure total request time, time-to-first-token
(streaming), and estimated token throughput.  All timings use
:func:`time.perf_counter` for high-resolution monotonic measurements.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, asdict
from typing import Any, Generator, Optional

import openai

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LatencyMetrics:
    """Structured container for latency measurements."""

    total_seconds: float = 0.0
    first_token_seconds: Optional[float] = None
    tokens_per_second: Optional[float] = None
    estimated_tokens: Optional[int] = None
    response_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Round floats for readability.
        for key in ("total_seconds", "first_token_seconds", "tokens_per_second"):
            if d[key] is not None:
                d[key] = round(d[key], 4)
        return d


# ---------------------------------------------------------------------------
# Measurement functions
# ---------------------------------------------------------------------------

def time_request(
    client: openai.OpenAI,
    model: str,
    messages: list[dict[str, str]],
    **kwargs: Any,
) -> LatencyMetrics:
    """Send a non-streaming chat completion and measure the total round-trip time.

    Parameters
    ----------
    client:
        An ``openai.OpenAI`` client configured with the target base URL.
    model:
        Model identifier (e.g. ``"mistral-7b"``).
    messages:
        Chat messages in OpenAI format.
    **kwargs:
        Additional keyword arguments forwarded to ``client.chat.completions.create``.

    Returns
    -------
    LatencyMetrics
        Populated with ``total_seconds``, ``estimated_tokens``,
        ``tokens_per_second``, and ``response_text``.
    """
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            **kwargs,
        )
        elapsed = time.perf_counter() - start

        text = response.choices[0].message.content or ""
        est_tokens = _estimate_token_count(text)
        tps = est_tokens / elapsed if elapsed > 0 else 0.0

        return LatencyMetrics(
            total_seconds=elapsed,
            first_token_seconds=None,
            tokens_per_second=tps,
            estimated_tokens=est_tokens,
            response_text=text,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.error("Request failed after %.2fs: %s", elapsed, exc)
        return LatencyMetrics(
            total_seconds=elapsed,
            response_text=f"[ERROR] {exc}",
        )


def time_first_token(
    client: openai.OpenAI,
    model: str,
    messages: list[dict[str, str]],
    **kwargs: Any,
) -> LatencyMetrics:
    """Send a streaming chat completion and measure time-to-first-token.

    Consumes the full stream so that ``total_seconds`` and
    ``tokens_per_second`` are also populated.

    Parameters
    ----------
    client:
        An ``openai.OpenAI`` client configured with the target base URL.
    model:
        Model identifier.
    messages:
        Chat messages in OpenAI format.
    **kwargs:
        Additional keyword arguments forwarded to the API call.

    Returns
    -------
    LatencyMetrics
        Populated with all latency fields including ``first_token_seconds``.
    """
    chunks: list[str] = []
    first_token_time: Optional[float] = None

    start = time.perf_counter()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start
                chunks.append(delta.content)

        elapsed = time.perf_counter() - start
        text = "".join(chunks)
        est_tokens = _estimate_token_count(text)
        tps = est_tokens / elapsed if elapsed > 0 else 0.0

        return LatencyMetrics(
            total_seconds=elapsed,
            first_token_seconds=first_token_time,
            tokens_per_second=tps,
            estimated_tokens=est_tokens,
            response_text=text,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.error("Streaming request failed after %.2fs: %s", elapsed, exc)
        return LatencyMetrics(
            total_seconds=elapsed,
            first_token_seconds=first_token_time,
            response_text=f"[ERROR] {exc}",
        )


def calculate_tokens_per_second(
    text: str,
    elapsed_seconds: float,
) -> float:
    """Estimate throughput in tokens/second for a completed response.

    Uses a simple whitespace heuristic (1 token ~ 0.75 words) to
    approximate token count without requiring a tokenizer library.

    Parameters
    ----------
    text:
        The full response text.
    elapsed_seconds:
        Wall-clock time the response took.

    Returns
    -------
    float
        Estimated tokens per second.  Returns ``0.0`` if elapsed is zero.
    """
    if elapsed_seconds <= 0:
        return 0.0
    return _estimate_token_count(text) / elapsed_seconds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_token_count(text: str) -> int:
    """Rough token-count estimate: ~1.33 tokens per whitespace-delimited word.

    This approximation is widely used when a tokenizer is unavailable and
    is accurate to within ~15% for English prose.  It slightly underestimates
    for agglutinative languages like Turkish, but remains serviceable for
    benchmarking purposes.
    """
    word_count = len(text.split())
    return max(int(word_count * 1.33), 1) if word_count > 0 else 0
