"""CrewAI agent definitions for the multi-agent research pipeline.

This module defines three specialized agents -- Researcher, Writer, and
Reviewer -- that collaborate to produce high-quality research reports.
All agents use an OpenAI-compatible API so they work interchangeably
with LocalAI, Ollama (via OpenAI proxy), or the OpenAI API itself.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from crewai import Agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment & configuration
# ---------------------------------------------------------------------------

load_dotenv()

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict[str, Any]:
    """Load agent and task configuration from config.yaml."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_llm() -> ChatOpenAI:
    """Construct a ChatOpenAI instance from environment / config.

    Priority: environment variables > config.yaml defaults.
    """
    config = _load_config().get("llm", {})

    base_url = os.getenv("LLM_BASE_URL", config.get("base_url", "http://localai:8080/v1"))
    api_key = os.getenv("LLM_API_KEY", config.get("api_key", "sk-no-key-required"))
    model = os.getenv("LLM_MODEL_NAME", config.get("model", "gpt-3.5-turbo"))
    temperature = float(os.getenv("LLM_TEMPERATURE", str(config.get("temperature", 0.7))))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", str(config.get("max_tokens", 4096))))
    request_timeout = int(os.getenv("LLM_TIMEOUT", str(config.get("request_timeout", 120))))

    logger.info(
        "Initialising LLM: model=%s base_url=%s temperature=%.2f",
        model,
        base_url,
        temperature,
    )

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
    )


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------


def create_researcher_agent(llm: ChatOpenAI | None = None) -> Agent:
    """Create the Senior Research Analyst agent.

    Parameters
    ----------
    llm:
        Optional pre-configured LLM instance. If *None*, one is created
        from environment / config.

    Returns
    -------
    Agent
        A CrewAI ``Agent`` configured for research tasks.
    """
    config = _load_config()["agents"]["researcher"]
    llm = llm or _build_llm()

    return Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        allow_delegation=config.get("allow_delegation", False),
        verbose=config.get("verbose", True),
        llm=llm,
    )


def create_writer_agent(llm: ChatOpenAI | None = None) -> Agent:
    """Create the Technical Writer agent.

    Parameters
    ----------
    llm:
        Optional pre-configured LLM instance.

    Returns
    -------
    Agent
        A CrewAI ``Agent`` configured for report-writing tasks.
    """
    config = _load_config()["agents"]["writer"]
    llm = llm or _build_llm()

    return Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        allow_delegation=config.get("allow_delegation", False),
        verbose=config.get("verbose", True),
        llm=llm,
    )


def create_reviewer_agent(llm: ChatOpenAI | None = None) -> Agent:
    """Create the Quality Reviewer agent.

    Parameters
    ----------
    llm:
        Optional pre-configured LLM instance.

    Returns
    -------
    Agent
        A CrewAI ``Agent`` configured for quality-review tasks.
    """
    config = _load_config()["agents"]["reviewer"]
    llm = llm or _build_llm()

    return Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        allow_delegation=config.get("allow_delegation", False),
        verbose=config.get("verbose", True),
        llm=llm,
    )


def create_all_agents(
    llm: ChatOpenAI | None = None,
) -> tuple[Agent, Agent, Agent]:
    """Convenience helper that returns all three agents sharing one LLM.

    Returns
    -------
    tuple[Agent, Agent, Agent]
        ``(researcher, writer, reviewer)``
    """
    llm = llm or _build_llm()
    return (
        create_researcher_agent(llm),
        create_writer_agent(llm),
        create_reviewer_agent(llm),
    )
