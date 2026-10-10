import re

from backend.agents.llm import chat, parse_json_object, split_sources

SYSTEM_PROMPT = (
    "You are a research analyst. Read the research report and return ONLY a JSON object "
    "with exactly these keys:\n"
    '- "takeaways": exactly 5 key takeaways, one concise sentence each.\n'
    '- "statistics": up to 8 of the most important numbers, percentages, amounts, or dates '
    'that appear in the report, most important first. Each item is {"value": "<the figure '
    'with its unit, e.g. 9%, $4.5B, 131, 2022>", "label": "<what it measures, at most 8 '
    'words>"}. Prefer quantitative measurements (percentages, amounts, counts, growth rates) '
    "over bare years; include a year only when it is a key milestone, and never a "
    "publication date. Use ONLY figures stated in the report; return [] if there are none.\n"
    '- "follow_up_questions": exactly 5 specific research questions a reader might ask next, '
    "each answerable by a web search.\n"
    '- "mind_map": one entry per subtopic, in the order given, each {"subtopic": "<subtopic>", '
    '"facts": [2-3 short facts from the report, at most 12 words each]}.\n'
    "Do not include [VERIFIED] or [UNVERIFIED] tags. Output the JSON object only."
)

TAG_RE = re.compile(r"\s*\[(?:UN)?VERIFIED\]")


def _clean(value, limit: int) -> str:
    return TAG_RE.sub("", str(value)).strip()[:limit]


def _strings(items, count: int, limit: int = 300) -> list[str]:
    if not isinstance(items, list):
        return []
    return [s for s in (_clean(i, limit) for i in items) if s][:count]


def extract_insights(query: str, report: str, subtopics: list[str]) -> dict:
    body, _ = split_sources(report)
    subtopic_list = "\n".join(f"- {s}" for s in subtopics) or "- (none)"
    response = chat(
        SYSTEM_PROMPT,
        f"Research query: {query}\n\nSubtopics:\n{subtopic_list}\n\nReport:\n\n{body}",
        temperature=0.2,
        max_tokens=1800,
    )
    data = parse_json_object(response)

    statistics = []
    for item in data.get("statistics") or []:
        if isinstance(item, dict) and item.get("value"):
            statistics.append({"value": _clean(item["value"], 40), "label": _clean(item.get("label", ""), 90)})

    mind_map = []
    for item in data.get("mind_map") or []:
        if isinstance(item, dict) and item.get("subtopic"):
            mind_map.append({"subtopic": _clean(item["subtopic"], 120), "facts": _strings(item.get("facts"), 3, 140)})

    return {
        "takeaways": _strings(data.get("takeaways"), 5),
        "statistics": statistics[:8],
        "follow_up_questions": _strings(data.get("follow_up_questions"), 5, 200),
        "mind_map": mind_map,
    }
