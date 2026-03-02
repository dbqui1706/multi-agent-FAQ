import logging
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send
from graph.state import GraphState
from graph.nodes import make_nodes

logger = logging.getLogger(__name__)


# Conditional routes
def route_generators(state: GraphState) -> list[Send]:
    """
    Route to generators in the same time
    """
    return [
        Send("faq_generator",   state),
        Send("adversarial_generator", state),
        Send("multichunk_generator",  state)
    ]
def route_after_review(state: GraphState) -> str:
    """
    After reviewer:
    - If there are items that failed the first time (not retried) → go to node retry
    - If there is nothing left to retry → go to coverage gate
    """
    reviewed = state.get("reviewed_faqs", [])
    need_retry = any(
        not r["is_approved"] and not r.get("is_retry")
        for r in reviewed
    )
    if need_retry:
        logger.info("Route: items need retry → node_retry")
        return "node_retry"
    logger.info("Route: no need retry → coverage_gate")
    return "coverage_gate"

def route_after_coverage(state: GraphState) -> str:
    """
    After coverage gate:
    - coverage < threshold and not yet done with gap-fill -> go to extractor (gap-fill)
    - coverage >= threshold OR gap_fill_done (already did gap-fill) -> go to evaluator
    """
    coverage      = state.get("coverage", 1.0)
    gap_fill_done = state.get("gap_fill_done", False)

    if coverage < state["COVERAGE_THRESHOLD"] and not gap_fill_done:
        logger.info(
            "Route: coverage %.1f%% < %.0f%% -> gap-fill",
            coverage * 100, state["COVERAGE_THRESHOLD"] * 100,
        )
        return "extractor"
    logger.info("Route: coverage %.1f%% OK -> evaluator", coverage * 100)
    return "evaluator"

# Build graph
def build_graph(client, model_name):
    workflow = StateGraph(GraphState)
    nodes = make_nodes(client, model_name)
    
    # nodes 
    workflow.add_node("chunker", nodes["chunker"])
    workflow.add_node("extractor", nodes["extractor"])
    workflow.add_node("faq_generator", nodes["faq_generator"])
    workflow.add_node("adversarial_generator", nodes["adversarial_generator"])
    workflow.add_node("multichunk_generator", nodes["multichunk_generator"])
    workflow.add_node("merge_dedup", nodes["merge_dedup"])
    workflow.add_node("naturalizer", nodes["naturalizer"])
    workflow.add_node("reviewer", nodes["reviewer"])
    workflow.add_node("node_retry", nodes["node_retry"])
    workflow.add_node("coverage_gate", nodes["coverage_gate"])
    workflow.add_node("evaluator", nodes["evaluator"])
    workflow.add_node("output", nodes["output"])    

    # edges
    workflow.add_edge(START, "chunker")
    workflow.add_edge("chunker", "extractor")

    # out -> parallel edges
    workflow.add_conditional_edges(
        "extractor",
        route_generators,
        ["faq_generator", "adversarial_generator", "multichunk_generator"],
    )
    
    # in <- parallel edges
    workflow.add_edge("faq_generator",   "merge_dedup")
    workflow.add_edge("adversarial_generator", "merge_dedup")
    workflow.add_edge("multichunk_generator",  "merge_dedup")

    workflow.add_edge("merge_dedup", "naturalizer")
    workflow.add_edge("naturalizer", "reviewer")

    # reviewer -> retry or coverage gate
    workflow.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"node_retry": "node_retry", "coverage_gate": "coverage_gate"},
    )

    # Retry → rerun reviewer (is_retry=True)
    workflow.add_edge("node_retry", "reviewer")
    
    # Coverage gate → evaluator hoặc extractor (gap-fill)
    workflow.add_conditional_edges(
        "coverage_gate",
        route_after_coverage,
        {"evaluator": "evaluator", "extractor": "extractor"},
    )

    workflow.add_edge("evaluator", "output")
    workflow.add_edge("output", END)
    return workflow.compile(checkpointer=MemorySaver())