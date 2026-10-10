from backend.agents.llm import chat, format_search_results
from backend.graph.state import ResearchState

SYSTEM_PROMPT = (
    "You are a professional report writer. Write a clean Markdown research report with "
    "exactly this structure:\n"
    "# <Report title>\n"
    "## Executive Summary\n"
    "(one concise paragraph)\n"
    "## <Subtopic name>\n"
    "(one section per subtopic, in the order given)\n"
    "## Conclusion\n"
    "Base the content on the fact-checked synthesis and subtopic findings. The "
    "fact-checked text tags claims with [VERIFIED] or [UNVERIFIED]. Whenever you reuse a "
    "tagged claim you MUST copy its tag exactly, placed right after the claim; never drop, "
    "change, or invent tags. Do NOT include a Sources section; it is appended separately."
)


def report_writer(state: ResearchState) -> dict:
    subtopic_list = "\n".join(f"- {s}" for s in state["subtopics"])
    body = chat(
        SYSTEM_PROMPT,
        f"Research query: {state['query']}\n\n"
        f"Subtopics (one section each):\n{subtopic_list}\n\n"
        f"Fact-checked synthesis:\n\n{state['fact_checked']}\n\n"
        f"Subtopic findings:\n\n{format_search_results(state['search_results'])}",
        max_tokens=2500,
    )

    # Build the sources list in code so URLs are never hallucinated or dropped.
    unique_sources = list(dict.fromkeys(state.get("sources", [])))
    sources_md = "\n".join(f"{i}. <{url}>" for i, url in enumerate(unique_sources, 1)) or "_No sources found._"

    report = f"{body.rstrip()}\n\n## Sources\n\n{sources_md}\n"
    return {"report": report, "status": "Report complete"}
