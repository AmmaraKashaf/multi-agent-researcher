from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from backend.agents.fact_checker import fact_checker
from backend.agents.orchestrator import orchestrator
from backend.agents.report_writer import report_writer
from backend.agents.synthesizer import synthesizer
from backend.agents.web_searcher import web_searcher
from backend.graph.state import ResearchState


def fan_out_searches(state: ResearchState) -> list[Send]:
    """Launch one web_searcher per subtopic; LangGraph runs them in parallel."""
    return [
        Send("web_searcher", {"query": state["query"], "subtopic": subtopic, "mode": state.get("mode", "deep")})
        for subtopic in state["subtopics"]
    ]


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("orchestrator", orchestrator)
    graph.add_node("web_searcher", web_searcher)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("fact_checker", fact_checker)
    graph.add_node("report_writer", report_writer)

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges("orchestrator", fan_out_searches, ["web_searcher"])
    # synthesizer waits until every parallel web_searcher in the superstep has finished.
    graph.add_edge("web_searcher", "synthesizer")
    graph.add_edge("synthesizer", "fact_checker")
    graph.add_edge("fact_checker", "report_writer")
    graph.add_edge("report_writer", END)

    return graph.compile()


research_graph = build_graph()
