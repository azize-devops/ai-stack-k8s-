# LLM Comparison Benchmark

Systematic benchmarking framework for comparing Large Language Models deployed through the AI Stack Kubernetes cluster. Tests models served via LocalAI alongside external API providers to evaluate performance, quality, and resource efficiency across English and Turkish prompts.

---

## Models Tested

| Model | Provider | Parameters | Hosting | Notes |
|-------|----------|-----------|---------|-------|
| **Mistral 7B** | LocalAI | 7B | Self-hosted (K8s) | Strong general-purpose performance, GGUF quantized |
| **LLaMA 3 8B** | LocalAI | 8B | Self-hosted (K8s) | Meta's latest open model, good multilingual support |
| **Phi-3 Mini** | LocalAI | 3.8B | Self-hosted (K8s) | Microsoft's compact model, fastest inference |
| **GPT-4o-mini** | OpenAI API | Unknown | Cloud | External baseline for quality comparison |

All LocalAI models are accessed through the same OpenAI-compatible endpoint (`/v1/chat/completions`), making it trivial to swap models without changing application code.

---

## Metrics

The benchmark evaluates each model across four dimensions:

### 1. Latency (`metrics/latency.py`)
- **Total response time** -- wall-clock time from request to full response
- **Time to first token** -- streaming latency until the first chunk arrives
- **Tokens per second** -- estimated throughput (word-count heuristic)

### 2. Response Quality (`metrics/response_quality.py`)
All quality scores are normalized to the **0--1** range using heuristic evaluation (no external LLM judge required):

- **Relevance** -- keyword overlap and prompt-response alignment
- **Completeness** -- word count, elaboration markers, paragraph structure
- **Coherence** -- sentence structure, formatting markers, logical transitions
- **Turkish quality** -- Turkish character usage, stop-word frequency, encoding integrity

### 3. Memory Usage (`metrics/memory_usage.py`)
- System memory via `psutil` (or `/proc/meminfo` fallback)
- Kubernetes pod memory via `kubectl top` (optional)
- Before/after delta to measure model-loading impact

### 4. Turkish Language Support
Turkish is evaluated as a first-class dimension because it is an agglutinative language that many smaller models handle poorly. The benchmark includes dedicated Turkish prompts and a specialized quality scorer.

---

## Project Structure

```
llm-comparison/
├── README.md                   # This file
├── compare_models.py           # Main benchmark CLI
├── benchmark_results.json      # Sample / latest results
└── metrics/
    ├── __init__.py             # Package exports
    ├── latency.py              # Request timing utilities
    ├── response_quality.py     # Heuristic quality scoring
    └── memory_usage.py         # Memory measurement
```

---

## Quick Start

### Prerequisites

```bash
pip install openai psutil
```

### Running the Benchmark

```bash
# Run against all LocalAI models (requires LocalAI running)
python compare_models.py

# Override the LocalAI endpoint
python compare_models.py --localai-url http://localhost:8080/v1

# Include GPT-4o-mini (requires API key)
export OPENAI_API_KEY=sk-your-key-here
python compare_models.py

# Benchmark specific models only
python compare_models.py --models mistral-7b phi-3

# Use custom prompts
python compare_models.py --prompts-file my_prompts.json

# Non-streaming mode (skip first-token measurement)
python compare_models.py --no-streaming

# Save to a custom output path
python compare_models.py --output results/run_$(date +%Y%m%d).json
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCALAI_BASE_URL` | `http://localhost:8080/v1` | LocalAI endpoint |
| `OPENAI_API_KEY` | *(unset)* | Set to include GPT-4o-mini |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI endpoint override |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### Custom Prompts File

Create a JSON array of prompt objects:

```json
[
  {
    "id": "my_prompt_1",
    "language": "en",
    "prompt": "Explain microservices architecture.",
    "keywords": ["service", "API", "container", "deploy"]
  },
  {
    "id": "tr_sorum",
    "language": "tr",
    "prompt": "Docker ve Kubernetes arasindaki farki aciklayin.",
    "keywords": ["container", "orkestrasyon"]
  }
]
```

---

## Sample Results

Results from a 3-node k3s cluster (32 GB RAM, 8 CPU threads, AVX2).

### Summary Table

| Model | Avg Time | 1st Token | Tok/s | Quality (EN) | Quality (TR) | Memory Delta |
|-------|----------|-----------|-------|--------------|--------------|-------------|
| **Mistral 7B** | 4.23 s | 0.87 s | 18.7 | 0.764 | 0.589 | +5,325 MB |
| **LLaMA 3 8B** | 5.18 s | 1.12 s | 15.3 | 0.789 | 0.623 | +5,734 MB |
| **Phi-3 Mini** | 2.87 s | 0.51 s | 28.4 | 0.712 | 0.489 | +3,277 MB |
| **GPT-4o-mini** | 1.82 s | 0.31 s | 52.7 | 0.892 | 0.812 | 0 MB |

### Key Observations

1. **GPT-4o-mini** achieves the highest quality and throughput but requires an external API, adding network latency and cost considerations for production.

2. **Mistral 7B** offers the best quality-to-resource ratio among self-hosted models. Its English quality (0.764) approaches GPT-4o-mini while keeping everything on-cluster.

3. **LLaMA 3 8B** provides the strongest Turkish language support among local models (0.623), making it the best self-hosted choice for multilingual workloads.

4. **Phi-3 Mini** is the speed champion at 28.4 tok/s with the smallest memory footprint (3.3 GB). Ideal for latency-sensitive applications where quality trade-offs are acceptable.

5. **Turkish performance** is notably weaker across all local models compared to English, confirming the need for targeted fine-tuning or larger models for production Turkish NLP.

---

## Extending

### Adding a New Model

Add an entry to the `_get_default_models()` function in `compare_models.py`:

```python
ModelConfig(
    name="My New Model",
    model_id="my-model-id",
    base_url="http://my-endpoint:8080/v1",
    api_key="sk-no-key-required",
)
```

### Adding New Metrics

Create a new module in `metrics/` and add it to `metrics/__init__.py`. The module should expose a function that accepts a response string and returns a structured result.

### CI Integration

The benchmark can be run in CI to detect quality regressions:

```bash
python compare_models.py --output results.json
# Then compare results.json against a baseline using jq or a custom script
```
