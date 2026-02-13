"""
Metrics package for LLM comparison benchmarks.

Provides modules for measuring response quality, latency, and memory usage
when evaluating language models through OpenAI-compatible APIs.
"""

from metrics.response_quality import (
    evaluate_response,
    score_relevance,
    score_completeness,
    score_coherence,
    score_turkish_quality,
)
from metrics.latency import (
    time_request,
    time_first_token,
    calculate_tokens_per_second,
    LatencyMetrics,
)
from metrics.memory_usage import (
    get_system_memory,
    get_pod_memory,
    measure_memory_delta,
    MemoryMetrics,
)

__all__ = [
    # Response quality
    "evaluate_response",
    "score_relevance",
    "score_completeness",
    "score_coherence",
    "score_turkish_quality",
    # Latency
    "time_request",
    "time_first_token",
    "calculate_tokens_per_second",
    "LatencyMetrics",
    # Memory
    "get_system_memory",
    "get_pod_memory",
    "measure_memory_delta",
    "MemoryMetrics",
]
