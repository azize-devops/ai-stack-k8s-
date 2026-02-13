#!/usr/bin/env python3
"""Main entry point for the LangGraph stateful query-routing workflow.

Usage
-----
    # Run built-in example queries
    python main.py

    # Run with a custom query
    python main.py "What are the trade-offs of microservices vs monoliths?"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Ensure the package is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph import build_workflow  # noqa: E402

# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES: list[str] = [
    "What is a vector database and how does it work?",
    "Compare Kubernetes StatefulSets and Deployments for running databases.",
    "Generate five creative names for an AI-powered code review tool.",
]

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
        description="Run the LangGraph query-routing workflow.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help=(
            "A single query to process. If omitted, the built-in example "
            "queries are run instead."
        ),
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Stream intermediate state updates instead of waiting for final output.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def _print_divider(char: str = "=", width: int = 72) -> None:
    print(char * width)


def _print_state_summary(state: dict[str, Any]) -> None:
    """Pretty-print the final workflow state."""
    _print_divider()
    print("WORKFLOW RESULT")
    _print_divider()
    print(f"  Query          : {state.get('query', 'N/A')}")
    print(f"  Classification : {state.get('classification', 'N/A')}")
    print(f"  Route          : {state.get('route', 'N/A')}")

    ctx = state.get("context", [])
    print(f"  Context chunks : {len(ctx)}")

    print(f"  Quality score  : {state.get('quality_score', 'N/A')}")
    print(f"  Quality passed : {state.get('quality_passed', 'N/A')}")
    _print_divider("-")
    print("RESPONSE:")
    print(state.get("response", "(no response generated)"))
    _print_divider()


def run_query(query: str, *, stream: bool = False) -> dict[str, Any]:
    """Build the graph and run a single query through it.

    Parameters
    ----------
    query:
        The user question to process.
    stream:
        If *True*, print intermediate node outputs as they complete.

    Returns
    -------
    dict
        The final workflow state.
    """
    logger = logging.getLogger(__name__)
    logger.info("Building workflow graph ...")
    workflow = build_workflow()

    initial_state = {"query": query}

    if stream:
        logger.info("Streaming workflow execution ...")
        final_state: dict[str, Any] = {}
        for step in workflow.stream(initial_state):
            # Each step is a dict mapping node_name -> partial state update.
            for node_name, node_output in step.items():
                print(f"\n>> Node '{node_name}' completed:")
                for key, value in node_output.items():
                    display = str(value)
                    if len(display) > 200:
                        display = display[:200] + " ..."
                    print(f"   {key}: {display}")
                final_state.update(node_output)
        # Ensure the original query is present.
        final_state.setdefault("query", query)
        return final_state
    else:
        logger.info("Invoking workflow ...")
        result = workflow.invoke(initial_state)
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    load_dotenv()
    _configure_logging()

    logger = logging.getLogger(__name__)
    args = _parse_args()

    queries = [args.query] if args.query else EXAMPLE_QUERIES

    for i, query in enumerate(queries, start=1):
        print(f"\n{'#' * 72}")
        print(f"# Query {i}/{len(queries)}")
        print(f"{'#' * 72}")
        print(f"  {query}\n")

        try:
            state = run_query(query, stream=args.stream)
            _print_state_summary(state)
        except KeyboardInterrupt:
            logger.warning("Interrupted by user.")
            sys.exit(130)
        except Exception:
            logger.exception("Workflow failed for query: %s", query)
            continue

    print("\nAll queries processed.")


if __name__ == "__main__":
    main()
