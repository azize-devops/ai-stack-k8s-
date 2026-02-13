"""CrewAI task definitions for the multi-agent research pipeline.

Each task maps to one of the three agents and defines the work to be
performed at that stage of the pipeline.  Task descriptions and expected
outputs are loaded from ``config.yaml`` so they can be tuned without
touching Python code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Task

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_task_config() -> dict[str, Any]:
    """Return the ``tasks`` section from config.yaml."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh).get("tasks", {})


# ---------------------------------------------------------------------------
# Task factories
# ---------------------------------------------------------------------------


def create_research_task(agent: Agent, topic: str) -> Task:
    """Create the research task assigned to the Researcher agent.

    Parameters
    ----------
    agent:
        The Researcher ``Agent`` instance that will execute this task.
    topic:
        The research topic to investigate.

    Returns
    -------
    Task
        A CrewAI ``Task`` for gathering research on *topic*.
    """
    config = _load_task_config()["research"]

    description = config["description"].replace("{topic}", topic)
    expected_output = config["expected_output"]

    logger.info("Created research task for topic: %s", topic)

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
    )


def create_writing_task(agent: Agent, topic: str, context: list[Task] | None = None) -> Task:
    """Create the writing task assigned to the Writer agent.

    Parameters
    ----------
    agent:
        The Writer ``Agent`` instance that will execute this task.
    topic:
        The report topic (used in the task description).
    context:
        Optional list of upstream ``Task`` objects whose outputs should
        be provided as context to this task.

    Returns
    -------
    Task
        A CrewAI ``Task`` for drafting a technical report.
    """
    config = _load_task_config()["writing"]

    description = config["description"].replace("{topic}", topic)
    expected_output = config["expected_output"]

    logger.info("Created writing task for topic: %s", topic)

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        context=context or [],
    )


def create_review_task(agent: Agent, topic: str, context: list[Task] | None = None) -> Task:
    """Create the review task assigned to the Reviewer agent.

    Parameters
    ----------
    agent:
        The Reviewer ``Agent`` instance that will execute this task.
    topic:
        The report topic (used in the task description).
    context:
        Optional list of upstream ``Task`` objects whose outputs should
        be provided as context to this task.

    Returns
    -------
    Task
        A CrewAI ``Task`` for quality-reviewing the report.
    """
    config = _load_task_config()["review"]

    description = config["description"].replace("{topic}", topic)
    expected_output = config["expected_output"]

    logger.info("Created review task for topic: %s", topic)

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        context=context or [],
    )


def create_all_tasks(
    researcher: Agent,
    writer: Agent,
    reviewer: Agent,
    topic: str,
) -> tuple[Task, Task, Task]:
    """Create the full task chain with proper context wiring.

    The writing task receives the research task output as context, and
    the review task receives both research and writing outputs.

    Parameters
    ----------
    researcher:
        The Researcher agent.
    writer:
        The Writer agent.
    reviewer:
        The Reviewer agent.
    topic:
        The topic string to research.

    Returns
    -------
    tuple[Task, Task, Task]
        ``(research_task, writing_task, review_task)``
    """
    research_task = create_research_task(researcher, topic)
    writing_task = create_writing_task(writer, topic, context=[research_task])
    review_task = create_review_task(reviewer, topic, context=[research_task, writing_task])

    return research_task, writing_task, review_task
