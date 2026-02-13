#!/usr/bin/env python3
"""
LLM Comparison Benchmark
=========================

Systematically benchmarks multiple LLMs through OpenAI-compatible APIs,
measuring latency, throughput, response quality, and memory usage.

Designed to work with:
- LocalAI-hosted models (Mistral 7B, LLaMA 3 8B, Phi-3) in Kubernetes
- External APIs (OpenAI GPT-4o-mini) for baseline comparison

Usage
-----
    # Run against all configured models
    python compare_models.py

    # Run against specific models
    python compare_models.py --models mistral-7b llama-3-8b

    # Use a custom prompts file
    python compare_models.py --prompts-file my_prompts.json

    # Save results to a specific path
    python compare_models.py --output results/run_2024.json

Environment variables
---------------------
    LOCALAI_BASE_URL   Base URL for LocalAI  (default: http://localhost:8080/v1)
    OPENAI_API_KEY     OpenAI API key         (optional, enables GPT-4o-mini)
    OPENAI_BASE_URL    OpenAI base URL        (default: https://api.openai.com/v1)
    LOG_LEVEL          Logging verbosity      (default: INFO)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import openai

# Ensure the project root is on sys.path so `metrics` is importable as a package.
_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from metrics.latency import time_request, time_first_token, LatencyMetrics
from metrics.response_quality import evaluate_response, QualityScores
from metrics.memory_usage import get_system_memory, MemoryMetrics

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("llm-benchmark")


# ---------------------------------------------------------------------------
# Default test prompts
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS: list[dict[str, Any]] = [
    # --- English prompts ---
    {
        "id": "en_explain_rag",
        "language": "en",
        "prompt": "Explain Retrieval-Augmented Generation (RAG) and why it is useful for enterprise applications.",
        "keywords": ["retrieval", "generation", "knowledge", "hallucination", "context"],
    },
    {
        "id": "en_k8s_scaling",
        "language": "en",
        "prompt": "What are the key strategies for auto-scaling applications in Kubernetes? Provide specific examples.",
        "keywords": ["HPA", "VPA", "cluster autoscaler", "metrics", "pods"],
    },
    {
        "id": "en_python_async",
        "language": "en",
        "prompt": "Write a Python async function that fetches data from three URLs concurrently and returns the results as a list.",
        "keywords": ["async", "await", "asyncio", "gather", "aiohttp"],
    },
    {
        "id": "en_vector_db",
        "language": "en",
        "prompt": "Compare vector databases (Qdrant, Pinecone, Weaviate) for production RAG pipelines. What are the trade-offs?",
        "keywords": ["vector", "embedding", "similarity", "performance", "scalability"],
    },
    {
        "id": "en_creative",
        "language": "en",
        "prompt": "Write a short story (100-200 words) about a DevOps engineer who discovers their Kubernetes cluster has become sentient.",
        "keywords": ["cluster", "engineer", "sentient", "pod"],
    },
    # --- Turkish prompts ---
    {
        "id": "tr_ai_aciklama",
        "language": "tr",
        "prompt": "Yapay zeka ve makine \u00f6\u011frenimi aras\u0131ndaki fark\u0131 a\u00e7\u0131klay\u0131n. Ger\u00e7ek hayattan \u00f6rnekler verin.",
        "keywords": ["yapay zeka", "makine \u00f6\u011frenimi", "derin \u00f6\u011frenme", "veri", "model"],
    },
    {
        "id": "tr_bulut_bilisim",
        "language": "tr",
        "prompt": "Bulut bili\u015fimin (cloud computing) avantajlar\u0131 ve dezavantajlar\u0131 nelerdir? \u015eirketler i\u00e7in \u00f6neriniz nedir?",
        "keywords": ["bulut", "maliyet", "g\u00fcvenlik", "\u00f6l\u00e7eklenebilirlik", "sunucu"],
    },
    {
        "id": "tr_kubernetes",
        "language": "tr",
        "prompt": "Kubernetes nedir ve neden mikro servis mimarilerinde tercih edilir? Container orkestrasyon kavram\u0131n\u0131 a\u00e7\u0131klay\u0131n.",
        "keywords": ["kubernetes", "container", "orkestrasyon", "mikro servis", "pod"],
    },
]


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for a single model to benchmark."""

    name: str
    model_id: str
    base_url: str
    api_key: str = "sk-no-key-required"
    is_external: bool = False
    max_tokens: int = 512
    temperature: float = 0.7


def _get_default_models() -> list[ModelConfig]:
    """Build the default set of models from environment variables."""
    localai_url = os.environ.get("LOCALAI_BASE_URL", "http://localhost:8080/v1")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    openai_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    models = [
        ModelConfig(
            name="Mistral 7B",
            model_id="mistral-7b",
            base_url=localai_url,
        ),
        ModelConfig(
            name="LLaMA 3 8B",
            model_id="llama-3-8b",
            base_url=localai_url,
        ),
        ModelConfig(
            name="Phi-3 Mini",
            model_id="phi-3",
            base_url=localai_url,
        ),
    ]

    if openai_key:
        models.append(
            ModelConfig(
                name="GPT-4o-mini",
                model_id="gpt-4o-mini",
                base_url=openai_url,
                api_key=openai_key,
                is_external=True,
            )
        )
    else:
        logger.info(
            "OPENAI_API_KEY not set -- skipping GPT-4o-mini. "
            "Set the variable to include it in the benchmark."
        )

    return models


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class PromptResult:
    """Results for a single prompt against a single model."""

    prompt_id: str
    language: str
    prompt_text: str
    response_text: str
    latency: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResult:
    """Aggregated results for a single model across all prompts."""

    model_name: str
    model_id: str
    base_url: str
    is_external: bool
    prompt_results: list[dict[str, Any]] = field(default_factory=list)
    avg_total_seconds: float = 0.0
    avg_first_token_seconds: Optional[float] = None
    avg_tokens_per_second: float = 0.0
    avg_quality_overall: float = 0.0
    avg_quality_en: float = 0.0
    avg_quality_tr: float = 0.0
    memory_before: dict[str, Any] = field(default_factory=dict)
    memory_after: dict[str, Any] = field(default_factory=dict)
    memory_delta_used_mb: float = 0.0
    error_count: int = 0


def run_benchmark(
    models: list[ModelConfig],
    prompts: list[dict[str, Any]],
    use_streaming: bool = True,
) -> list[ModelResult]:
    """Execute the full benchmark suite.

    For each model, sends every prompt and records latency and quality
    metrics.  Memory usage is sampled before and after the model's
    prompt batch.

    Parameters
    ----------
    models:
        List of model configurations to benchmark.
    prompts:
        List of prompt dicts with ``id``, ``language``, ``prompt``, and
        optional ``keywords``.
    use_streaming:
        If ``True``, uses streaming requests to measure first-token latency.

    Returns
    -------
    list[ModelResult]
        One entry per model with per-prompt and aggregate metrics.
    """
    results: list[ModelResult] = []

    for model_cfg in models:
        logger.info("=" * 60)
        logger.info("Benchmarking: %s (%s)", model_cfg.name, model_cfg.model_id)
        logger.info("Endpoint:     %s", model_cfg.base_url)
        logger.info("=" * 60)

        client = openai.OpenAI(
            base_url=model_cfg.base_url,
            api_key=model_cfg.api_key,
            timeout=120.0,
        )

        mem_before = get_system_memory()
        prompt_results: list[dict[str, Any]] = []
        total_times: list[float] = []
        first_token_times: list[float] = []
        tps_values: list[float] = []
        quality_scores_en: list[float] = []
        quality_scores_tr: list[float] = []
        quality_scores_all: list[float] = []
        error_count = 0

        for i, prompt_spec in enumerate(prompts, 1):
            prompt_id = prompt_spec["id"]
            language = prompt_spec["language"]
            prompt_text = prompt_spec["prompt"]
            keywords = prompt_spec.get("keywords", [])
            is_turkish = language == "tr"

            logger.info(
                "  [%d/%d] %s (%s)", i, len(prompts), prompt_id, language,
            )

            messages = [
                {"role": "system", "content": _system_prompt(language)},
                {"role": "user", "content": prompt_text},
            ]

            # --- Latency measurement ---
            if use_streaming:
                latency = time_first_token(
                    client, model_cfg.model_id, messages,
                    max_tokens=model_cfg.max_tokens,
                    temperature=model_cfg.temperature,
                )
            else:
                latency = time_request(
                    client, model_cfg.model_id, messages,
                    max_tokens=model_cfg.max_tokens,
                    temperature=model_cfg.temperature,
                )

            response_text = latency.response_text

            if response_text.startswith("[ERROR]"):
                logger.warning("    -> Error: %s", response_text)
                error_count += 1
                # Record the failed prompt with zeroed scores.
                prompt_results.append({
                    "prompt_id": prompt_id,
                    "language": language,
                    "prompt": prompt_text,
                    "response": response_text,
                    "latency": latency.to_dict(),
                    "quality": {"overall": 0.0, "error": True},
                })
                continue

            # --- Quality evaluation ---
            quality = evaluate_response(
                response=response_text,
                prompt=prompt_text,
                keywords=keywords,
                is_turkish=is_turkish,
            )

            logger.info(
                "    -> %.2fs | quality=%.3f | %d tokens",
                latency.total_seconds,
                quality.overall,
                latency.estimated_tokens or 0,
            )

            # Accumulate stats.
            total_times.append(latency.total_seconds)
            if latency.first_token_seconds is not None:
                first_token_times.append(latency.first_token_seconds)
            if latency.tokens_per_second is not None and latency.tokens_per_second > 0:
                tps_values.append(latency.tokens_per_second)
            quality_scores_all.append(quality.overall)
            if is_turkish:
                quality_scores_tr.append(quality.overall)
            else:
                quality_scores_en.append(quality.overall)

            prompt_results.append({
                "prompt_id": prompt_id,
                "language": language,
                "prompt": prompt_text,
                "response": response_text[:500] + ("..." if len(response_text) > 500 else ""),
                "latency": latency.to_dict(),
                "quality": quality.to_dict(),
            })

            # Brief pause between prompts to avoid rate-limiting.
            if model_cfg.is_external:
                time.sleep(1.0)

        mem_after = get_system_memory()

        model_result = ModelResult(
            model_name=model_cfg.name,
            model_id=model_cfg.model_id,
            base_url=model_cfg.base_url,
            is_external=model_cfg.is_external,
            prompt_results=prompt_results,
            avg_total_seconds=_safe_mean(total_times),
            avg_first_token_seconds=_safe_mean(first_token_times) if first_token_times else None,
            avg_tokens_per_second=_safe_mean(tps_values),
            avg_quality_overall=_safe_mean(quality_scores_all),
            avg_quality_en=_safe_mean(quality_scores_en),
            avg_quality_tr=_safe_mean(quality_scores_tr),
            memory_before=mem_before.to_dict(),
            memory_after=mem_after.to_dict(),
            memory_delta_used_mb=round(mem_after.used_mb - mem_before.used_mb, 2),
            error_count=error_count,
        )
        results.append(model_result)

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_summary_table(results: list[ModelResult]) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 100)
    print("  LLM BENCHMARK RESULTS SUMMARY")
    print("=" * 100)

    header = (
        f"{'Model':<18} {'Avg Time':>10} {'1st Token':>10} {'Tok/s':>8} "
        f"{'Quality':>9} {'EN Qual':>9} {'TR Qual':>9} "
        f"{'Errors':>7} {'Mem \u0394 MB':>10}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        ft = f"{r.avg_first_token_seconds:.3f}s" if r.avg_first_token_seconds else "N/A"
        print(
            f"{r.model_name:<18} "
            f"{r.avg_total_seconds:>9.3f}s "
            f"{ft:>10} "
            f"{r.avg_tokens_per_second:>7.1f} "
            f"{r.avg_quality_overall:>8.3f} "
            f"{r.avg_quality_en:>8.3f} "
            f"{r.avg_quality_tr:>8.3f} "
            f"{r.error_count:>7} "
            f"{r.memory_delta_used_mb:>9.1f}"
        )

    print("=" * 100)
    print()


def save_results(
    results: list[ModelResult],
    output_path: Path,
) -> None:
    """Serialize benchmark results to JSON."""
    payload = {
        "benchmark_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "tool": "ai-stack-k8s/llm-comparison",
        },
        "models": [asdict(r) for r in results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)

    logger.info("Results saved to %s", output_path)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(path: Optional[Path]) -> list[dict[str, Any]]:
    """Load prompts from a JSON file or return the built-in defaults.

    The JSON file should be a list of objects, each with at least
    ``id``, ``language``, and ``prompt`` keys.
    """
    if path is None:
        logger.info("Using %d built-in test prompts", len(DEFAULT_PROMPTS))
        return DEFAULT_PROMPTS

    logger.info("Loading prompts from %s", path)
    with open(path, encoding="utf-8") as fh:
        prompts = json.load(fh)

    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"Prompts file must contain a non-empty JSON array: {path}")

    for p in prompts:
        if "id" not in p or "prompt" not in p:
            raise ValueError(f"Each prompt must have 'id' and 'prompt' keys: {p}")
        p.setdefault("language", "en")
        p.setdefault("keywords", [])

    return prompts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark and compare LLMs through OpenAI-compatible APIs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL_ID",
        help=(
            "Model IDs to benchmark (e.g. mistral-7b llama-3-8b). "
            "Defaults to all configured models."
        ),
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Path to a JSON file with custom test prompts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_DIR / "benchmark_results.json",
        help="Path to save JSON results (default: benchmark_results.json).",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming (skip first-token measurement).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max_tokens for all models.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override temperature for all models.",
    )
    parser.add_argument(
        "--localai-url",
        type=str,
        default=None,
        help="Override LOCALAI_BASE_URL for this run.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for the benchmark CLI."""
    args = parse_args(argv)

    # Override env vars from CLI flags.
    if args.localai_url:
        os.environ["LOCALAI_BASE_URL"] = args.localai_url

    # Build model list.
    all_models = _get_default_models()
    if args.models:
        selected_ids = set(args.models)
        models = [m for m in all_models if m.model_id in selected_ids]
        unknown = selected_ids - {m.model_id for m in models}
        if unknown:
            logger.warning("Unknown model IDs (skipped): %s", ", ".join(unknown))
        if not models:
            logger.error("No valid models selected. Available: %s",
                         ", ".join(m.model_id for m in all_models))
            sys.exit(1)
    else:
        models = all_models

    # Apply global overrides.
    if args.max_tokens is not None:
        for m in models:
            m.max_tokens = args.max_tokens
    if args.temperature is not None:
        for m in models:
            m.temperature = args.temperature

    # Load prompts.
    prompts = load_prompts(args.prompts_file)

    # Run benchmark.
    logger.info("Starting benchmark: %d models x %d prompts", len(models), len(prompts))
    start_time = time.perf_counter()

    results = run_benchmark(
        models=models,
        prompts=prompts,
        use_streaming=not args.no_streaming,
    )

    elapsed = time.perf_counter() - start_time
    logger.info("Benchmark completed in %.1f seconds", elapsed)

    # Output results.
    print_summary_table(results)
    save_results(results, args.output)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _system_prompt(language: str) -> str:
    """Return an appropriate system prompt for the given language."""
    if language == "tr":
        return (
            "Sen yard\u0131mc\u0131 bir yapay zeka asistan\u0131s\u0131n. "
            "T\u00fcrk\u00e7e olarak detayl\u0131 ve do\u011fru cevaplar ver. "
            "Cevaplar\u0131n\u0131 madde madde a\u00e7\u0131kla."
        )
    return (
        "You are a helpful AI assistant. Provide detailed, accurate, "
        "and well-structured answers."
    )


def _safe_mean(values: list[float]) -> float:
    """Return the mean of a list, or 0.0 if empty."""
    return round(sum(values) / len(values), 4) if values else 0.0


if __name__ == "__main__":
    main()
