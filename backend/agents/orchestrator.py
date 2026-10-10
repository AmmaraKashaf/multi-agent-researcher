from datetime import date

from backend.agents.llm import chat, parse_json_list
from backend.agents.modes import SUBTOPIC_COUNT, normalize_mode
from backend.graph.state import ResearchState

SYSTEM_PROMPT = (
    "You are a research orchestrator. Break the user's research query into exactly {count} "
    "distinct, non-overlapping subtopics that together cover the question well. "
    "Each subtopic should be a short, specific phrase suitable as a web search query. "
    "Today's date is {today}. Do not put dates, years, or version numbers in the subtopics "
    "unless the user's query mentions them; your own knowledge may be out of date. "
    "{focus}"
    'Respond with ONLY a JSON list of strings, e.g. ["subtopic one", "subtopic two"].'
)

FOCUS = {
    "deep": "Aim for thorough coverage: background, current state, evidence, challenges, and outlook. ",
    "quick": "Pick only the most essential angles needed for a fast overview. ",
    "news": (
        "Focus on recent developments, announcements, and events from the last few weeks or months. "
        "Keep subtopics generic (e.g. 'new model releases', not a specific product name you remember), "
        "so the news search surfaces what is actually new. "
    ),
}


def orchestrator(state: ResearchState) -> dict:
    query = state["query"]
    mode = normalize_mode(state.get("mode"))
    count = SUBTOPIC_COUNT[mode]

    prompt = SYSTEM_PROMPT.format(count=count, focus=FOCUS[mode], today=date.today().isoformat())
    response = chat(prompt, f"Research query: {query}", temperature=0.2, max_tokens=512)

    try:
        subtopics = [str(s).strip() for s in parse_json_list(response) if str(s).strip()]
    except ValueError:
        subtopics = []

    # Fall back to researching the query directly rather than failing the whole run.
    subtopics = subtopics[:count] or [query]

    return {"subtopics": subtopics, "status": f"Planned {len(subtopics)} subtopics"}
