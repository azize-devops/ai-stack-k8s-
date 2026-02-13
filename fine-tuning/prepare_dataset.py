#!/usr/bin/env python3
"""Prepare raw CSV/JSON data for instruction-tuning fine-tuning.

Reads a raw dataset (CSV or JSON), validates and cleans each record, converts
it to the standard instruction-tuning format, performs a train/validation split,
and writes the results as JSONL files.

Expected schema per record:
    instruction (str, required) - The task or question.
    input       (str, optional) - Additional context; defaults to "".
    output      (str, required) - The desired response.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict[str, Any]]:
    """Load records from a CSV file.

    The CSV must have a header row with at least ``instruction`` and
    ``output`` columns.
    """
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {path} appears to be empty or has no header.")
        missing = {"instruction", "output"} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV file {path} is missing required columns: {missing}"
            )
        for row in reader:
            records.append(dict(row))
    return records


def load_json(path: Path) -> list[dict[str, Any]]:
    """Load records from a JSON file.

    Accepts either a JSON array of objects or a JSONL file (one JSON object
    per line).
    """
    text = path.read_text(encoding="utf-8").strip()

    # Try JSON array first.
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}, got {type(data)}.")
        return data

    # Fall back to JSONL.
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {lineno} of {path}: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise ValueError(
                f"Expected a JSON object on line {lineno} of {path}, got {type(obj)}."
            )
        records.append(obj)
    return records


def load_raw_data(path: Path) -> list[dict[str, Any]]:
    """Dispatch to the correct loader based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path)
    if suffix in (".json", ".jsonl"):
        return load_json(path)
    raise ValueError(
        f"Unsupported file type '{suffix}'. Expected .csv, .json, or .jsonl."
    )


# ---------------------------------------------------------------------------
# Cleaning and validation
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace and strip surrounding quotes."""
    text = text.strip().strip('"').strip("'")
    # Collapse multiple whitespace characters into a single space.
    return " ".join(text.split())


def validate_record(record: dict[str, Any], index: int) -> dict[str, str] | None:
    """Validate and normalise a single record.

    Returns the cleaned record or ``None`` if the record should be skipped.
    """
    instruction = record.get("instruction")
    output = record.get("output")

    if not instruction or not str(instruction).strip():
        logger.warning("Record %d skipped: empty 'instruction' field.", index)
        return None
    if not output or not str(output).strip():
        logger.warning("Record %d skipped: empty 'output' field.", index)
        return None

    cleaned: dict[str, str] = {
        "instruction": clean_text(str(instruction)),
        "input": clean_text(str(record.get("input", ""))),
        "output": clean_text(str(output)),
    }

    # Sanity-check minimum length.
    if len(cleaned["instruction"]) < 5:
        logger.warning(
            "Record %d skipped: 'instruction' too short (%d chars).",
            index,
            len(cleaned["instruction"]),
        )
        return None
    if len(cleaned["output"]) < 5:
        logger.warning(
            "Record %d skipped: 'output' too short (%d chars).",
            index,
            len(cleaned["output"]),
        )
        return None

    return cleaned


# ---------------------------------------------------------------------------
# Splitting and writing
# ---------------------------------------------------------------------------

def split_dataset(
    records: list[dict[str, str]],
    val_split: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Shuffle and split records into train and validation sets."""
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1.0 - val_split)))
    return shuffled[:split_idx], shuffled[split_idx:]


def write_jsonl(records: list[dict[str, str]], path: Path) -> None:
    """Write a list of records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare(
    input_path: Path,
    output_dir: Path,
    val_split: float = 0.1,
    seed: int = 42,
) -> None:
    """End-to-end dataset preparation pipeline."""
    logger.info("Loading raw data from %s", input_path)
    raw_records = load_raw_data(input_path)
    logger.info("Loaded %d raw records.", len(raw_records))

    # Validate and clean.
    clean_records: list[dict[str, str]] = []
    for idx, record in enumerate(raw_records):
        cleaned = validate_record(record, idx)
        if cleaned is not None:
            clean_records.append(cleaned)

    skipped = len(raw_records) - len(clean_records)
    if skipped:
        logger.warning("Skipped %d invalid records.", skipped)
    logger.info("Retained %d valid records after cleaning.", len(clean_records))

    if not clean_records:
        logger.error("No valid records remain. Aborting.")
        sys.exit(1)

    # Deduplicate by instruction text.
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for record in clean_records:
        key = record["instruction"].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(record)
    if len(deduped) < len(clean_records):
        logger.info(
            "Removed %d duplicate instructions.", len(clean_records) - len(deduped)
        )
    clean_records = deduped

    # Split.
    train, val = split_dataset(clean_records, val_split, seed)
    logger.info("Split: %d train, %d validation.", len(train), len(val))

    # Write.
    write_jsonl(train, output_dir / "train.jsonl")
    write_jsonl(val, output_dir / "val.jsonl")

    # Also write a combined file for convenience.
    write_jsonl(clean_records, output_dir / "all.jsonl")

    logger.info("Dataset preparation complete. Output directory: %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a raw dataset for instruction-tuning fine-tuning.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw dataset file (.csv, .json, or .jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prepared"),
        help="Output directory for processed JSONL files (default: data/prepared).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data to reserve for validation (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(
        input_path=args.input,
        output_dir=args.output,
        val_split=args.val_split,
        seed=args.seed,
    )
