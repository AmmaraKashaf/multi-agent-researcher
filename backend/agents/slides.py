import re

from backend.agents.llm import chat, parse_json_list, split_sources

SYSTEM_PROMPT = (
    "Convert this research report into 5-6 presentation slides. Each slide: Title + 3 bullet "
    'points. Return ONLY a JSON array: [{"title": "...", "bullets": ["...", "...", "..."]}]. '
    "The first slide introduces the topic and the last one concludes. Keep titles under 8 "
    "words and bullets under 16 words. Use only information from the report and do not "
    "include [VERIFIED] or [UNVERIFIED] tags."
)

TAG_RE = re.compile(r"\s*\[(?:UN)?VERIFIED\]")


def generate_slides(report: str) -> list[dict]:
    body, _ = split_sources(report)
    data = parse_json_list(chat(SYSTEM_PROMPT, body, temperature=0.4, max_tokens=1500))

    slides = []
    for item in data:
        if not isinstance(item, dict) or not item.get("title"):
            continue
        bullets = [TAG_RE.sub("", str(b)).strip() for b in item.get("bullets") or [] if str(b).strip()]
        slides.append({"title": TAG_RE.sub("", str(item["title"])).strip(), "bullets": bullets[:4]})
    if not slides:
        raise ValueError("The model returned no usable slides")
    return slides[:6]
