#!/usr/bin/env python3
"""Evaluate a QLoRA fine-tuned model against its base model.

Loads the base model (4-bit quantised) and optionally the LoRA adapter,
runs inference on a set of test prompts, prints a side-by-side comparison,
optionally computes perplexity, and saves all results to JSON.

Typical invocation:

    python evaluate.py \
        --model   TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --adapter output/tinyllama-k8s-qlora \
        --output  output/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import textwrap
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default test prompts (Kubernetes / DevOps themed)
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS: list[str] = [
    "Explain the difference between a Kubernetes Deployment and a StatefulSet.",
    "How do you troubleshoot a pod stuck in CrashLoopBackOff?",
    "What is a Kubernetes Service of type ClusterIP and when would you use it?",
    "Describe how horizontal pod autoscaling works in Kubernetes.",
    "What are init containers and why are they useful?",
    "How does a ConfigMap differ from a Secret in Kubernetes?",
    "Explain the purpose of a PersistentVolumeClaim.",
    "What steps would you take to perform a zero-downtime rolling update?",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_model(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the base model in 4-bit quantisation and its tokenizer."""
    logger.info("Loading base model: %s", model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model(
    base_model: PreTrainedModel,
    adapter_path: str,
) -> PreTrainedModel:
    """Load LoRA adapter weights on top of an already-loaded base model."""
    logger.info("Loading LoRA adapter from: %s", adapter_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

INSTRUCTION_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n\n\n"
    "### Response:\n"
)


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a single response from the model."""
    formatted = INSTRUCTION_TEMPLATE.format(instruction=prompt)
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (skip the prompt).
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


def run_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    label: str,
    max_new_tokens: int = 256,
) -> list[dict[str, Any]]:
    """Run inference on a list of prompts and collect results."""
    results: list[dict[str, Any]] = []
    for i, prompt in enumerate(prompts, start=1):
        logger.info("[%s] Generating %d/%d: %s", label, i, len(prompts), prompt[:60])
        start = time.time()
        response = generate_response(model, tokenizer, prompt, max_new_tokens)
        elapsed = time.time() - start
        results.append({
            "prompt": prompt,
            "response": response,
            "generation_time_s": round(elapsed, 2),
        })
    return results


# ---------------------------------------------------------------------------
# Perplexity calculation
# ---------------------------------------------------------------------------

def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int = 512,
) -> float | None:
    """Compute perplexity over a list of texts.

    Returns None if computation fails (e.g. empty input).
    """
    if not texts:
        return None

    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        input_ids = encodings["input_ids"]
        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

        if loss is not None:
            num_tokens = input_ids.shape[1] - 1
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return None

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return round(perplexity, 4)


def load_validation_texts(path: Path) -> list[str]:
    """Load validation texts from a JSONL file for perplexity computation."""
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Reconstruct the full prompt+response as a single text.
            text = (
                f"### Instruction:\n{record.get('instruction', '')}\n\n"
                f"### Input:\n{record.get('input', '')}\n\n"
                f"### Response:\n{record.get('output', '')}"
            )
            texts.append(text)
    return texts


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_comparison(
    base_results: list[dict[str, Any]],
    ft_results: list[dict[str, Any]],
) -> None:
    """Print a side-by-side comparison of base vs fine-tuned outputs."""
    width = 78
    separator = "=" * width

    print(f"\n{separator}")
    print("  BASE MODEL vs FINE-TUNED MODEL -- Side-by-Side Comparison")
    print(separator)

    for base_r, ft_r in zip(base_results, ft_results):
        print(f"\n{'- ' * (width // 2)}")
        print(f"PROMPT: {base_r['prompt']}")
        print()

        print("[Base Model]")
        for line in textwrap.wrap(base_r["response"], width=width - 2):
            print(f"  {line}")
        print(f"  (generated in {base_r['generation_time_s']}s)")
        print()

        print("[Fine-Tuned Model]")
        for line in textwrap.wrap(ft_r["response"], width=width - 2):
            print(f"  {line}")
        print(f"  (generated in {ft_r['generation_time_s']}s)")

    print(f"\n{separator}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(
    model_name: str,
    adapter_path: str | None,
    output_path: Path,
    prompts: list[str],
    val_data_path: Path | None = None,
    max_new_tokens: int = 256,
) -> None:
    """End-to-end evaluation pipeline."""
    results: dict[str, Any] = {
        "model": model_name,
        "adapter": adapter_path,
        "num_prompts": len(prompts),
        "prompts": prompts,
    }

    # ---- Base model inference --------------------------------------------
    base_model, tokenizer = load_base_model(model_name)
    base_results = run_inference(
        base_model, tokenizer, prompts, "BASE", max_new_tokens
    )
    results["base_model"] = base_results

    # ---- Perplexity (base) -----------------------------------------------
    if val_data_path and val_data_path.exists():
        logger.info("Computing base model perplexity on %s", val_data_path)
        val_texts = load_validation_texts(val_data_path)
        base_ppl = compute_perplexity(base_model, tokenizer, val_texts)
        results["base_perplexity"] = base_ppl
        logger.info("Base model perplexity: %s", base_ppl)
    else:
        results["base_perplexity"] = None

    # ---- Fine-tuned model inference --------------------------------------
    if adapter_path:
        ft_model = load_finetuned_model(base_model, adapter_path)
        ft_results = run_inference(
            ft_model, tokenizer, prompts, "FINETUNED", max_new_tokens
        )
        results["finetuned_model"] = ft_results

        # Perplexity (fine-tuned).
        if val_data_path and val_data_path.exists():
            logger.info("Computing fine-tuned model perplexity on %s", val_data_path)
            ft_ppl = compute_perplexity(ft_model, tokenizer, val_texts)
            results["finetuned_perplexity"] = ft_ppl
            logger.info("Fine-tuned model perplexity: %s", ft_ppl)
        else:
            results["finetuned_perplexity"] = None

        # Print comparison.
        print_comparison(base_results, ft_results)
    else:
        logger.info("No adapter path provided. Evaluating base model only.")
        results["finetuned_model"] = None
        results["finetuned_perplexity"] = None

    # ---- Save results ----------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Evaluation results saved to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a QLoRA fine-tuned model against its base model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name or local path (default: %(default)s).",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter weights directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/eval_results.json"),
        help="Path to save evaluation results JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Path to a JSON file containing a list of test prompt strings.",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=None,
        help="Path to validation JSONL file for perplexity computation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load custom prompts or fall back to defaults.
    if args.prompts and args.prompts.exists():
        with args.prompts.open("r", encoding="utf-8") as fh:
            prompts = json.load(fh)
        if not isinstance(prompts, list):
            raise ValueError("Prompts file must contain a JSON list of strings.")
        logger.info("Loaded %d custom test prompts.", len(prompts))
    else:
        prompts = DEFAULT_PROMPTS
        logger.info("Using %d default test prompts.", len(prompts))

    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available. Inference will be slow on CPU."
        )

    evaluate(
        model_name=args.model,
        adapter_path=args.adapter,
        output_path=args.output,
        prompts=prompts,
        val_data_path=args.val_data,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
