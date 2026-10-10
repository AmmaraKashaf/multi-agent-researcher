from backend.agents.llm import chat, format_search_results
from backend.graph.state import ResearchState

SYSTEM_PROMPT = (
    "You are a meticulous fact checker. You are given a synthesis and the search findings "
    "it was based on. Rewrite the synthesis unchanged except for adding a tag immediately "
    "after each specific factual claim (figures, dates, names, causal statements):\n"
    "- [VERIFIED] if the search findings directly support the claim\n"
    "- [UNVERIFIED] if the findings do not support it or it goes beyond them\n"
    "Do not tag general framing sentences. Keep the paragraph structure. Output only the "
    "tagged text."
)


def fact_checker(state: ResearchState) -> dict:
    evidence = format_search_results(state["search_results"])
    fact_checked = chat(
        SYSTEM_PROMPT,
        f"Search findings (evidence):\n\n{evidence}\n\n---\n\nSynthesis to check:\n\n{state['synthesis']}",
        temperature=0.1,
        max_tokens=2048,
    )
    return {"fact_checked": fact_checked, "status": "Fact-checked synthesis"}
