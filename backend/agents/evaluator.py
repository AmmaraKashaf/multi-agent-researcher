from urllib.parse import urlparse

from backend.agents.llm import chat, parse_json_object, split_sources

# One call covers three post-report features (confidence score, source quality, related
# topics) to stay within the free tier's tokens-per-minute limit.
SYSTEM_PROMPT = (
    "You are a research quality reviewer. Return ONLY a JSON object with exactly these keys:\n"
    '- "confidence": {"score": <integer 0-100>, "reason": "<one or two sentences>"}: your '
    "overall confidence in the accuracy and completeness of the report. Consider source "
    "diversity, the fact-check results ([VERIFIED] vs [UNVERIFIED] tags), and topic coverage.\n"
    '- "sources": one entry for EVERY source listed, in the same order: {"url": "<exact URL '
    'as given>", "credibility": <integer 1-10>, "bias": "Low" | "Medium" | "High", "note": '
    '"<why, at most 8 words>"}. Judge from the publisher\'s general reputation and the kind of '
    "page (peer-reviewed journals and official agencies rate high; vendor marketing, "
    "content farms, and partisan outlets rate lower or more biased).\n"
    '- "related_topics": exactly 5 items {"topic": "<a research query, at most 8 words>", '
    '"reason": "<one short sentence>"}: adjacent angles worth researching next that the '
    "report does not already cover.\n"
    "Output the JSON object only."
)

BIAS_LEVELS = {"low": "Low", "medium": "Medium", "high": "High"}


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _clamp_int(value, low: int, high: int, default: int) -> int:
    try:
        return max(low, min(high, round(float(value))))
    except (TypeError, ValueError):
        return default


def evaluate_research(query: str, report: str, sources: list[dict]) -> dict:
    body, _ = split_sources(report)
    verified = body.count("[VERIFIED]")
    unverified = body.count("[UNVERIFIED]")
    source_list = "\n".join(
        f"{i}. {s['url']}" + (f" | {s['title']}" if s.get("title") else "") for i, s in enumerate(sources, 1)
    ) or "(no sources)"

    data = parse_json_object(chat(
        SYSTEM_PROMPT,
        f"Research query: {query}\n"
        f"Fact-check results: {verified} claims verified, {unverified} unverified.\n\n"
        f"Sources:\n{source_list}\n\nReport:\n\n{body}",
        temperature=0.2,
        max_tokens=2200,
    ))

    rated = {}
    for item in data.get("sources") or []:
        if isinstance(item, dict) and item.get("url"):
            rated[str(item["url"]).strip()] = item

    source_ratings = []
    for s in sources:
        item = rated.get(s["url"], {})
        source_ratings.append({
            "url": s["url"],
            "domain": _domain(s["url"]),
            "title": s.get("title", ""),
            "credibility": _clamp_int(item.get("credibility"), 1, 10, 5),
            "bias": BIAS_LEVELS.get(str(item.get("bias", "")).strip().lower(), "Medium"),
            "note": str(item.get("note", "")).strip()[:80],
            "rated": bool(item),
        })

    confidence = data.get("confidence") if isinstance(data.get("confidence"), dict) else {}
    related = [
        {"topic": str(t["topic"]).strip()[:100], "reason": str(t.get("reason", "")).strip()[:200]}
        for t in data.get("related_topics") or []
        if isinstance(t, dict) and t.get("topic")
    ]

    return {
        "confidence": {
            "score": _clamp_int(confidence.get("score"), 0, 100, 50),
            "reason": str(confidence.get("reason", "")).strip()[:400],
        },
        "sources": source_ratings,
        "related_topics": related[:5],
    }
