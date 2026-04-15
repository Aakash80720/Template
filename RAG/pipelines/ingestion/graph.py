"""LangGraph StateGraph for the knowledge-graph ingestion pipeline."""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from models.state import IngestionState
from pipelines.ingestion.nodes import (
    embed_node,
    extract_node,
    ingest_node,
    link_node,
    resolve_node,
    store_node,
)
from monitoring.logger import get_logger

logger = get_logger(__name__)


def build_ingestion_graph(checkpointer=None):
    """
    Compiles the 6-node ingestion StateGraph.

    Graph topology:
        ingest → extract → resolve → link → embed → store → END

    HITL interrupt is declared at `resolve` — LangGraph will pause there
    when `interrupt()` is called inside resolve_node and resume on
    graph.update_state() + graph.invoke(None, config).
    """
    graph = StateGraph(IngestionState)

    # Register nodes
    graph.add_node("ingest", ingest_node)
    graph.add_node("extract", extract_node)
    graph.add_node("resolve", resolve_node)
    graph.add_node("link", link_node)
    graph.add_node("embed", embed_node)
    graph.add_node("store", store_node)

    # Linear pipeline edges
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "extract")
    graph.add_edge("extract", "resolve")
    graph.add_edge("resolve", "link")
    graph.add_edge("link", "embed")
    graph.add_edge("embed", "store")
    graph.add_edge("store", END)

    checkpointer = checkpointer or MemorySaver()

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["resolve"],  # HITL pause point
    )

    logger.info("Ingestion graph compiled")
    return compiled


# ---------------------------------------------------------------------------
# Convenience run helpers
# ---------------------------------------------------------------------------

def run_ingestion(
    source_type: str,
    source_config: dict,
    thread_id: str = "default",
) -> dict:
    """
    Run the full ingestion pipeline for a single source.

    Args:
        source_type:   "filesystem" | "github" | "mcp"
        source_config: connector-specific config dict
        thread_id:     LangGraph thread ID (for checkpointing / HITL resume)

    Returns:
        Final state dict with ingestion_summary populated.
    """
    import json
    from langchain_core.messages import HumanMessage

    graph = build_ingestion_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = IngestionState(
        current_source=source_type,
        messages=[HumanMessage(content=json.dumps(source_config))],
    )

    result = graph.invoke(initial_state, config=config)

    # If interrupted for HITL, caller must:
    #   graph.update_state(config, {"messages": [HumanMessage(content="approved")]})
    #   result = graph.invoke(None, config=config)

    return result
