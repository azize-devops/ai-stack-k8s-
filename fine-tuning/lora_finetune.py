#!/usr/bin/env python3
"""QLoRA fine-tuning script for small language models.

Loads a causal-LM in 4-bit quantisation (NF4) via bitsandbytes, injects
LoRA adapter layers through PEFT, and trains on an instruction-tuning
dataset in JSONL format.  Optimised for NVIDIA GTX 1660 SUPER (6 GB VRAM).

Typical invocation:

    python lora_finetune.py \
        --model  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --dataset data/sample_dataset.jsonl \
        --output  output/tinyllama-k8s-qlora
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FinetuneConfig:
    """All tuneable knobs in one place."""

    # Model
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_seq_length: int = 512

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # Training
    epochs: int = 3
    learning_rate: float = 2e-4
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 5

    # Security
    trust_remote_code: bool = False

    # Paths
    dataset_path: str = "data/sample_dataset.jsonl"
    output_dir: str = "output/qlora-run"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


def load_jsonl(path: Path) -> list[dict[str, str]]:
    """Load an instruction-tuning JSONL file."""
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON on line %d: %s", lineno, exc)
                continue
            records.append(obj)
    return records


def format_prompt(record: dict[str, str]) -> str:
    """Convert a single record to the chat-style prompt string."""
    return PROMPT_TEMPLATE.format(
        instruction=record.get("instruction", ""),
        input=record.get("input", ""),
        output=record.get("output", ""),
    )


def tokenize_dataset(
    records: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    """Tokenize all records and return a HuggingFace Dataset."""
    prompts = [format_prompt(r) for r in records]

    def tokenize_fn(examples: dict[str, list[str]]) -> dict[str, Any]:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        # For causal LM training the labels are the same as the input ids.
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = Dataset.from_dict({"text": prompts})
    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    return dataset


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_quantized_model(
    model_name: str,
    gradient_checkpointing: bool,
) -> AutoModelForCausalLM:
    """Load a causal-LM in 4-bit NF4 quantisation."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
    )

    # Prepare the quantised model for LoRA training.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
    )
    return model


def apply_lora(
    model: AutoModelForCausalLM,
    config: FinetuneConfig,
) -> AutoModelForCausalLM:
    """Inject LoRA adapter layers and return the wrapped model."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "LoRA injected. Trainable parameters: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / total,
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class MetricsLogger:
    """Accumulate and persist training metrics to a JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate any previous log.
        self.path.write_text("", encoding="utf-8")

    def log(self, metrics: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics, default=str) + "\n")


class LoggingCallback(torch.nn.Module):
    """Transformers Trainer callback that writes metrics to a JSONL log."""

    # We inherit from nn.Module only to satisfy type-checkers; this is used
    # as a TrainerCallback via duck-typing (Trainer accepts any object with
    # the right method signatures).

    def __init__(self, metrics_logger: MetricsLogger) -> None:
        super().__init__()
        self._logger = metrics_logger

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs:
            entry = {"step": state.global_step, "epoch": state.epoch, **logs}
            self._logger.log(entry)


def build_training_args(config: FinetuneConfig) -> TrainingArguments:
    """Construct HuggingFace TrainingArguments from our config."""
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        optim="paged_adamw_8bit",
    )


def train(config: FinetuneConfig) -> None:
    """Full fine-tuning pipeline."""
    start = time.time()

    # ---- Tokenizer -------------------------------------------------------
    logger.info("Loading tokenizer for %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- Dataset ---------------------------------------------------------
    dataset_path = Path(config.dataset_path)
    logger.info("Loading dataset from %s", dataset_path)
    records = load_jsonl(dataset_path)
    if not records:
        raise RuntimeError(f"No valid records found in {dataset_path}")
    logger.info("Loaded %d training examples.", len(records))

    dataset = tokenize_dataset(records, tokenizer, config.max_seq_length)

    # ---- Model -----------------------------------------------------------
    logger.info("Loading model %s in 4-bit quantisation.", config.model_name)
    model = load_quantized_model(config.model_name, config.gradient_checkpointing)
    model = apply_lora(model, config)

    # ---- Trainer ---------------------------------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = build_training_args(config)
    metrics_logger = MetricsLogger(Path(config.output_dir) / "training_log.jsonl")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[LoggingCallback(metrics_logger)],
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete in %.1f minutes.", (time.time() - start) / 60.0)

    # ---- Save ------------------------------------------------------------
    output_path = Path(config.output_dir)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Adapter weights and tokenizer saved to %s", output_path)

    # Save a copy of the config for reproducibility.
    config_dict = {
        k: v for k, v in config.__dict__.items() if not k.startswith("_")
    }
    (output_path / "finetune_config.json").write_text(
        json.dumps(config_dict, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for small causal language models.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=FinetuneConfig.model_name,
        help="HuggingFace model name or local path (default: %(default)s).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=FinetuneConfig.dataset_path,
        help="Path to the instruction-tuning JSONL file (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=FinetuneConfig.output_dir,
        help="Directory to save adapter weights (default: %(default)s).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=FinetuneConfig.epochs,
        help="Number of training epochs (default: %(default)s).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=FinetuneConfig.learning_rate,
        help="Learning rate (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=FinetuneConfig.per_device_batch_size,
        help="Per-device training batch size (default: %(default)s).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=FinetuneConfig.gradient_accumulation_steps,
        help="Gradient accumulation steps (default: %(default)s).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=FinetuneConfig.max_seq_length,
        help="Maximum sequence length in tokens (default: %(default)s).",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=FinetuneConfig.lora_r,
        help="LoRA rank (default: %(default)s).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=FinetuneConfig.lora_alpha,
        help="LoRA alpha scaling factor (default: %(default)s).",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=FinetuneConfig.lora_dropout,
        help="LoRA dropout rate (default: %(default)s).",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (uses more VRAM but is faster).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Allow executing remote code from HuggingFace model repos (default: False).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = FinetuneConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        learning_rate=args.lr,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        trust_remote_code=args.trust_remote_code,
    )

    # Print a compact summary before starting.
    logger.info("=" * 60)
    logger.info("QLoRA Fine-Tuning Configuration")
    logger.info("=" * 60)
    for key, value in config.__dict__.items():
        logger.info("  %-30s %s", key, value)
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available. Training will be extremely slow on CPU. "
            "Consider running on a machine with a GPU."
        )

    train(config)


if __name__ == "__main__":
    main()
