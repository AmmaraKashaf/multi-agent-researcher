import random
import re
import time
from datetime import date

try:
    from ddgs import DDGS  # current package name
except ImportError:
    from duckduckgo_search import DDGS  # legacy package name

from backend.agents.llm import chat
from backend.agents.modes import normalize_mode
from backend.graph.state import WebSearcherInput

MAX_RESULTS = 4

SYSTEM_PROMPT = (
    "You are a research analyst. Using ONLY the search snippets provided, extract the "
    "key findings relevant to the subtopic in 3-5 clear, factual sentences. Include "
    "concrete facts, figures, and dates where available. Do not invent information. "
    "If the snippets contain little relevant information, say so briefly."
)

NEWS_SYSTEM_PROMPT = (
    "You are a news analyst. Today's date is {today}. Using ONLY the recent news snippets "
    "provided, summarize the latest developments relevant to the subtopic in 3-5 clear, "
    "factual sentences. Lead with the most recent events and mention their dates. Include "
    "concrete facts and figures. Trust the snippet dates over your own knowledge, which may "
    "be out of date. Do not invent information. If the snippets contain little relevant "
    "information, say so briefly."
)


def build_search_query(query: str, subtopic: str) -> str:
    return f"{query} {subtopic}"


def _clean_title(title: str) -> str:
    title = " ".join(str(title).split())
    # DuckDuckGo sometimes glues several result titles together, e.g.
    # "9 Side Effects - HealthlineIntermittent Fasting Benefits - Mayo Clinic...".
    if len(title) > 120:
        glued = re.match(r"^(.{15,}? - [^-]{2,40}?[a-z])(?=[A-Z][a-z])", title)
        if glued:
            title = glued.group(1)
    return title[:200]


def _normalize(result: dict) -> dict:
    # Text results use "href"; news results use "url" and include a publication date.
    return {
        "title": _clean_title(result.get("title", "")),
        "href": result.get("href") or result.get("url", ""),
        "body": result.get("body", ""),
        "date": (result.get("date") or "")[:10],
    }


def _run_search(query: str, mode: str, max_results: int) -> list[dict]:
    with DDGS() as ddgs:
        if mode != "news":
            return list(ddgs.text(query, max_results=max_results))
        results = list(ddgs.news(query, timelimit="m", max_results=max_results))
        # Fall back to recent web pages when there's no news coverage.
        return results or list(ddgs.text(query, timelimit="m", max_results=max_results))


def search(query: str, mode: str = "deep", max_results: int = MAX_RESULTS, attempts: int = 4) -> list[dict]:
    # DDGS fails intermittently under concurrent requests, so retry with jittered backoff.
    # If every attempt fails, return [] so one subtopic doesn't kill the whole run.
    for attempt in range(attempts):
        try:
            results = [_normalize(r) for r in _run_search(query, mode, max_results)]
            if results:
                return results
        except Exception:
            pass
        if attempt < attempts - 1:
            time.sleep(1.5 * (attempt + 1) + random.uniform(0, 1))
    return []


def web_searcher(state: WebSearcherInput) -> dict:
    query, subtopic = state["query"], state["subtopic"]
    mode = normalize_mode(state.get("mode"))
    results = search(build_search_query(query, subtopic), mode)

    if not results:
        result = {"subtopic": subtopic, "findings": "No search results were found.", "sources": [], "source_details": []}
        return {"search_results": [result], "sources": []}

    snippets = "\n\n".join(
        f"[{i}] {r['title']}\nURL: {r['href']}\n" + (f"Published: {r['date']}\n" if r["date"] else "") + r["body"]
        for i, r in enumerate(results, 1)
    )
    findings = chat(
        NEWS_SYSTEM_PROMPT.format(today=date.today().isoformat()) if mode == "news" else SYSTEM_PROMPT,
        f"Overall research query: {query}\nSubtopic: {subtopic}\n\nSearch snippets:\n{snippets}",
        max_tokens=512,
    )
    with_url = [r for r in results if r["href"]]
    urls = [r["href"] for r in with_url]
    # Titles and dates let the UI build real citations instead of guessing from the URL.
    details = [{"url": r["href"], "title": r["title"], "date": r["date"]} for r in with_url]

    result = {"subtopic": subtopic, "findings": findings, "sources": urls, "source_details": details}
    return {"search_results": [result], "sources": urls}
