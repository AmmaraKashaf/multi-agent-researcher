import operator
from typing import Annotated, TypedDict


class SourceDetail(TypedDict):
    url: str
    title: str
    date: str  # ISO date for news results, "" when unknown


class SearchResult(TypedDict):
    subtopic: str
    findings: str
    sources: list[str]  # URLs
    source_details: list[SourceDetail]


class ResearchState(TypedDict):
    query: str
    mode: str  # "deep", "quick" or "news"; see backend/agents/modes.py
    subtopics: list[str]
    # Parallel web searchers append to these lists instead of overwriting them.
    search_results: Annotated[list[SearchResult], operator.add]
    synthesis: str
    fact_checked: str
    report: str
    sources: Annotated[list[str], operator.add]
    status: str


class WebSearcherInput(TypedDict):
    query: str
    subtopic: str
    mode: str
