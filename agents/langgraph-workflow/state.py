"""State schema for the LangGraph query-routing workflow.

The ``WorkflowState`` TypedDict describes every field that flows through
the graph.  LangGraph uses this schema to validate node inputs/outputs
and to manage state checkpoints.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class WorkflowState(TypedDict, total=False):
    """Typed state that flows through the LangGraph workflow.

    Attributes
    ----------
    query:
        The original user question or prompt.
    classification:
        The category assigned to the query by the classification node.
        One of ``"factual"``, ``"analytical"``, or ``"creative"``.
    context:
        A list of context strings retrieved from a vector store (or
        simulated) when the RAG route is selected.
    response:
        The final generated answer.
    route:
        Routing decision made by the classifier -- ``"rag"`` when
        context retrieval is beneficial, ``"direct"`` otherwise.
    quality_score:
        A numeric score (0-100) assigned by the quality-check node.
    quality_passed:
        Boolean indicating whether the response passed the quality gate.
    messages:
        Accumulated LangChain message objects (used internally by nodes
        that interact with LLMs).
    """

    query: str
    classification: str
    context: list[str]
    response: str
    route: str
    quality_score: int
    quality_passed: bool
    messages: Annotated[list, add_messages]
