# Fine-Tuning Module: QLoRA Experiments

QLoRA (Quantized Low-Rank Adaptation) fine-tuning experiments targeting
consumer-grade hardware. This module demonstrates how to adapt a small
language model to a specific domain (Kubernetes / DevOps) using parameter-efficient
fine-tuning techniques that fit within 6 GB of VRAM.

---

## Purpose

Traditional full fine-tuning of even a 1B-parameter model requires far more
memory than a consumer GPU can offer. QLoRA solves this by:

1. **Quantizing** the base model to 4-bit precision (NF4), slashing memory by ~4x.
2. **Freezing** all original weights.
3. **Training** a small set of low-rank adapter matrices (LoRA) in fp16/bf16.

The result is a lightweight adapter (typically 10-50 MB) that can be merged
with the base model at inference time or loaded on top of it.

---

## Target Model

| Property | Value |
|----------|-------|
| Default model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Parameters | 1.1 B |
| Architecture | LlamaForCausalLM |
| Context length | 2 048 tokens |
| Quantization | NF4 (4-bit) via bitsandbytes |

TinyLlama is the recommended starting point because it comfortably fits in
6 GB VRAM when quantized to 4-bit. Phi-2 (2.7 B) is an alternative if you
have additional headroom or use aggressive gradient checkpointing.

---

## Dataset Format and Preparation

### Raw input

The preparation script (`prepare_dataset.py`) accepts either **CSV** or **JSON**
files. Expected columns / keys:

| Column | Required | Description |
|--------|----------|-------------|
| `instruction` | Yes | The task or question |
| `input` | No | Additional context (empty string if absent) |
| `output` | Yes | The desired response |

### Instruction-tuning format

Each record is converted to a single JSONL line:

```json
{"instruction": "Explain what a Kubernetes Pod is.", "input": "", "output": "A Pod is the smallest deployable unit in Kubernetes..."}
```

### Running the preparation script

```bash
python prepare_dataset.py \
    --input  data/raw_qa.csv \
    --output data/prepared \
    --val-split 0.1 \
    --seed 42
```

This produces `train.jsonl` and `val.jsonl` inside `data/prepared/`.

A sample dataset with 25 Kubernetes/DevOps Q&A pairs is included at
`data/sample_dataset.jsonl` for immediate experimentation.

---

## Training Process and Hyperparameters

### Quick start

```bash
python lora_finetune.py \
    --model  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dataset data/sample_dataset.jsonl \
    --output  output/tinyllama-k8s-qlora \
    --epochs  3
```

### Default hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| LoRA rank (`r`) | 16 | Trade-off between capacity and memory |
| LoRA alpha | 32 | Scaling factor (alpha / r = 2) |
| LoRA dropout | 0.05 | Light regularisation |
| Target modules | `q_proj`, `v_proj` | Attention query and value projections |
| Learning rate | 2e-4 | With cosine schedule and warmup |
| Warmup ratio | 0.03 | ~3 % of total steps |
| Batch size | 4 | Per-device |
| Gradient accumulation | 4 | Effective batch size = 16 |
| Epochs | 3 | Sufficient for small datasets |
| Max sequence length | 512 | Keeps memory predictable |
| FP16 | True | Mixed precision training |
| Gradient checkpointing | True | Saves ~30 % VRAM at slight speed cost |

### What happens during training

1. The base model is loaded in **4-bit NF4** quantisation.
2. LoRA adapter layers are injected into `q_proj` and `v_proj`.
3. Only the adapter parameters (~2-5 M) are updated; the base model is frozen.
4. Training loss, learning rate, and epoch progress are logged to the console
   and to a `training_log.jsonl` file in the output directory.
5. The final adapter weights and tokenizer are saved to the output directory.

---

## Evaluation (Before / After)

```bash
python evaluate.py \
    --model   TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --adapter output/tinyllama-k8s-qlora \
    --output  output/eval_results.json
```

The evaluation script:

1. Loads the **base model** (4-bit quantised) and generates answers to a set
   of built-in test prompts.
2. Loads the **fine-tuned model** (base + LoRA adapter) and generates answers
   to the same prompts.
3. Prints a side-by-side comparison in the terminal.
4. Computes **perplexity** on the validation set (if provided via `--val-data`).
5. Saves all results to a JSON file for later analysis.

You can supply your own test prompts via `--prompts path/to/prompts.json`
(a JSON list of strings).

---

## Hardware Requirements

### Minimum (tested)

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GTX 1660 SUPER 6 GB |
| System RAM | 16 GB (8 GB minimum, swap recommended) |
| Disk | ~10 GB free (model cache + output) |
| CUDA | 11.8 or later |
| Python | 3.10+ |

### Memory budget (approximate, TinyLlama 1.1B)

| Component | VRAM |
|-----------|------|
| Base model (NF4) | ~0.7 GB |
| LoRA adapters (fp16) | ~0.02 GB |
| Optimizer states | ~0.1 GB |
| Activations (batch 4, seq 512, grad ckpt) | ~2.5 GB |
| CUDA overhead / fragmentation | ~1.0 GB |
| **Total** | **~4.3 GB** |

This leaves roughly 1.5 GB of headroom on a 6 GB card, enough for occasional
spikes during long-sequence batches.

### Software dependencies

```
torch>=2.1.0
transformers>=4.36.0
peft>=0.7.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
datasets>=2.16.0
scipy
```

Install everything with:

```bash
pip install torch transformers peft accelerate bitsandbytes datasets scipy
```

---

## Directory Structure

```
fine-tuning/
  README.md                 # This file
  prepare_dataset.py        # Data preparation script
  lora_finetune.py          # QLoRA training script
  evaluate.py               # Before/after evaluation
  data/
    sample_dataset.jsonl     # 25 example K8s/DevOps Q&A pairs
```

---

## References

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [PEFT library documentation](https://huggingface.co/docs/peft)
- [TinyLlama on Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
