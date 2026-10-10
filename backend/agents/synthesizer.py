from backend.agents.llm import chat, format_search_results
from backend.graph.state import ResearchState

SYSTEM_PROMPT = (
    "You are a research synthesizer. Merge the findings from several subtopic searches "
    "into 4-6 coherent paragraphs, each organized around a clear theme. Connect related "
    "points across subtopics, remove repetition, and note any contradictions between "
    "sources. Use only the information provided. Output plain paragraphs separated by "
    "blank lines, without headings."
)


def synthesizer(state: ResearchState) -> dict:
    findings = format_search_results(state["search_results"])
    synthesis = chat(
        SYSTEM_PROMPT,
        f"Research query: {state['query']}\n\nFindings:\n\n{findings}",
        max_tokens=1500,
    )
    return {"synthesis": synthesis, "status": "Synthesized findings"}
