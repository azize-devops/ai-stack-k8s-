#!/usr/bin/env python3
"""Main entry point for the CrewAI multi-agent research pipeline.

Usage
-----
    # Research a specific topic
    python main.py "Kubernetes autoscaling strategies"

    # Use the default demonstration topic
    python main.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from crewai import Crew, Process
from dotenv import load_dotenv

# Ensure the package is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents import create_all_agents  # noqa: E402
from tasks import create_all_tasks  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOPIC = "The current state and future of AI agents in enterprise software"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """Set up structured logging based on the LOG_LEVEL env var."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CrewAI multi-agent research pipeline.",
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default=DEFAULT_TOPIC,
        help="Research topic (defaults to a built-in demo topic).",
    )
    parser.add_argument(
        "--process",
        choices=["sequential", "hierarchical"],
        default="sequential",
        help="CrewAI process type (default: sequential).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for result files.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def run_pipeline(topic: str, process_type: str, output_dir: Path) -> str:
    """Assemble and execute the research crew.

    Parameters
    ----------
    topic:
        The topic to research.
    process_type:
        CrewAI process type -- ``"sequential"`` or ``"hierarchical"``.
    output_dir:
        Directory where the results file will be saved.

    Returns
    -------
    str
        The final crew output as a string.
    """
    logger = logging.getLogger(__name__)

    # -- Agents -------------------------------------------------------------
    logger.info("Creating agents ...")
    researcher, writer, reviewer = create_all_agents()

    # -- Tasks --------------------------------------------------------------
    logger.info("Creating tasks for topic: %s", topic)
    research_task, writing_task, review_task = create_all_tasks(
        researcher=researcher,
        writer=writer,
        reviewer=reviewer,
        topic=topic,
    )

    # -- Crew ---------------------------------------------------------------
    process = Process.sequential if process_type == "sequential" else Process.hierarchical

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=process,
        verbose=True,
    )

    logger.info("Kicking off crew with process=%s ...", process_type)
    result = crew.kickoff()

    # -- Persist results ----------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_topic = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)[:80].strip()
    safe_topic = safe_topic.replace(" ", "_").lower()
    filename = f"{timestamp}_{safe_topic}.md"
    output_path = output_dir / filename

    result_text = str(result)
    report_content = (
        f"# Research Report: {topic}\n\n"
        f"**Generated:** {timestamp}\n"
        f"**Process:** {process_type}\n\n"
        f"---\n\n"
        f"{result_text}\n"
    )

    output_path.write_text(report_content, encoding="utf-8")
    logger.info("Results saved to %s", output_path)

    return result_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    load_dotenv()
    _configure_logging()

    logger = logging.getLogger(__name__)
    args = _parse_args()

    logger.info("=" * 72)
    logger.info("CrewAI Research Pipeline")
    logger.info("Topic : %s", args.topic)
    logger.info("Process: %s", args.process)
    logger.info("=" * 72)

    try:
        result = run_pipeline(
            topic=args.topic,
            process_type=args.process,
            output_dir=args.output_dir,
        )
        print("\n" + "=" * 72)
        print("FINAL OUTPUT")
        print("=" * 72)
        print(result)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Pipeline failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
