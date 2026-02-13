"""LangGraph StateGraph construction for the query-routing workflow.

Graph structure
---------------
::

    START
      |
      v
    classify_query
      |
      +-- route == "rag" -----> retrieve_context --+
      |                                            |
      +-- route == "direct" --------------------+  |
                                                |  |
                                                v  v
                                          generate_response
                                                |
                                                v
                                          check_quality
                                                |
                                                v
                                               END
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from nodes import (
    check_quality,
    classify_query,
    generate_response,
    retrieve_context,
)
from state import WorkflowState

logger = logging.getLogger(__name__)


def _route_decision(state: WorkflowState) -> str:
    """Conditional edge: choose the next node based on the routing decision.

    Returns
    -------
    str
        ``"retrieve_context"`` if the classifier chose RAG, otherwise
        ``"generate_response"`` for a direct answer.
    """
    route = state.get("route", "direct")
    logger.debug("[route_decision] route=%s", route)
    if route == "rag":
        return "retrieve_context"
    return "generate_response"


def build_workflow() -> StateGraph:
    """Construct and return the compiled LangGraph workflow.

    The returned object is a compiled ``StateGraph`` that can be invoked
    via ``.invoke()`` or streamed via ``.stream()``.
    """
    graph = StateGraph(WorkflowState)

    # -- Register nodes -----------------------------------------------------
    graph.add_node("classify_query", classify_query)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_response", generate_response)
    graph.add_node("check_quality", check_quality)

    # -- Edges --------------------------------------------------------------
    # START -> classify_query
    graph.add_edge(START, "classify_query")

    # classify_query -> conditional routing
    graph.add_conditional_edges(
        "classify_query",
        _route_decision,
        {
            "retrieve_context": "retrieve_context",
            "generate_response": "generate_response",
        },
    )

    # retrieve_context -> generate_response
    graph.add_edge("retrieve_context", "generate_response")

    # generate_response -> check_quality
    graph.add_edge("generate_response", "check_quality")

    # check_quality -> END
    graph.add_edge("check_quality", END)

    logger.info("Workflow graph built successfully.")
    return graph.compile()
