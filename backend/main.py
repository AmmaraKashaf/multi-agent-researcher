import json
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import iterate_in_threadpool
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.agents.evaluator import evaluate_research
from backend.agents.formatter import Style, reformat_report
from backend.agents.insights import extract_insights
from backend.agents.llm import FALLBACK_MODELS, MODEL
from backend.agents.modes import DEFAULT_MODE, SUBTOPIC_COUNT, Mode
from backend.agents.report_chat import chat_about_report
from backend.agents.slides import generate_slides
from backend.agents.web_searcher import build_search_query
from backend.graph.pipeline import research_graph

logger = logging.getLogger(__name__)

INDEX_HTML = Path(__file__).resolve().parent.parent / "index.html"

app = FastAPI(title="Multi-Agent Researcher")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    mode: Mode = DEFAULT_MODE


class InsightsRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    report: str = Field(..., min_length=1, max_length=60_000)
    subtopics: list[str] = Field(default_factory=list, max_length=10)


class ReformatRequest(BaseModel):
    report: str = Field(..., min_length=1, max_length=60_000)
    style: Style


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=8_000)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    report: str = Field(..., min_length=1, max_length=60_000)
    question: str = Field(..., min_length=1, max_length=1_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=40)


class SlidesRequest(BaseModel):
    report: str = Field(..., min_length=1, max_length=60_000)


class SourceInfo(BaseModel):
    url: str = Field(..., max_length=2_000)
    title: str = Field("", max_length=500)


class EvaluateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    report: str = Field(..., min_length=1, max_length=60_000)
    sources: list[SourceInfo] = Field(default_factory=list, max_length=40)


STEP_MESSAGES = {
    "orchestrator": "Planned research subtopics",
    "synthesizer": "Synthesized findings into themes",
    "fact_checker": "Fact-checked claims against sources",
    "report_writer": "Report written",
}


def sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data)}


def status_event(node: str, update: dict, query: str) -> dict:
    payload = {"step": node, "message": STEP_MESSAGES.get(node, node)}

    if node == "orchestrator":
        payload["subtopics"] = update["subtopics"]
        # The exact search queries the parallel web searchers are about to run.
        payload["searches"] = [build_search_query(query, s) for s in update["subtopics"]]
        payload["message"] = f"Planned {len(update['subtopics'])} subtopics"
    elif node == "web_searcher":
        result = update["search_results"][0]
        payload["subtopic"] = result["subtopic"]
        payload["sources"] = result["sources"]
        payload["findings"] = result["findings"]
        payload["source_details"] = result.get("source_details", [])
        payload["message"] = f"Searched \"{result['subtopic']}\" ({len(result['sources'])} sources)"
    elif node == "fact_checker":
        text = update.get("fact_checked", "")
        payload["verified"] = text.count("[VERIFIED]")
        payload["unverified"] = text.count("[UNVERIFIED]")

    return sse("status", payload)


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(INDEX_HTML)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/info")
async def info():
    """Stack details for the UI, so it always shows the models actually configured."""
    return {
        "model": MODEL,
        "fallback_models": [m for m in FALLBACK_MODELS if m != MODEL],
        "search": "DuckDuckGo",
        "framework": "LangGraph",
        "modes": SUBTOPIC_COUNT,
    }


@app.post("/api/research")
async def research(request: ResearchRequest):
    query = request.query.strip()
    mode = request.mode

    async def event_stream():
        yield sse("status", {"step": "start", "message": f"Researching: {query}", "mode": mode})
        sources: list[str] = []
        source_details: dict[str, dict] = {}
        try:
            async for chunk in research_graph.astream({"query": query, "mode": mode}, stream_mode="updates"):
                for node, update in chunk.items():
                    sources.extend(update.get("sources", []))
                    for result in update.get("search_results", []):
                        for detail in result.get("source_details", []):
                            source_details.setdefault(detail["url"], detail)
                    yield status_event(node, update, query)
                    if node == "report_writer":
                        yield sse("report", {
                            "report": update["report"],
                            "sources": list(dict.fromkeys(sources)),
                            "source_details": list(source_details.values()),
                        })
        except Exception as exc:
            logger.exception("Research run failed")
            yield sse("error", {"message": str(exc)})

    return EventSourceResponse(event_stream())


# Plain `def` endpoints: FastAPI runs them in a thread pool, so blocking Groq calls
# (including rate-limit waits) don't stall the event loop.
@app.post("/api/insights")
def insights(request: InsightsRequest):
    try:
        return extract_insights(request.query, request.report, request.subtopics)
    except Exception as exc:
        logger.exception("Insight extraction failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/reformat")
def reformat(request: ReformatRequest):
    try:
        return {"report": reformat_report(request.report, request.style), "style": request.style}
    except Exception as exc:
        logger.exception("Reformat failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/chat")
async def chat_with_report(request: ChatRequest):
    history = [m.model_dump() for m in request.history]

    async def event_stream():
        try:
            tokens = chat_about_report(request.query, request.report, request.question.strip(), history)
            async for token in iterate_in_threadpool(tokens):
                yield sse("token", {"text": token})
            yield sse("done", {})
        except Exception as exc:
            logger.exception("Report chat failed")
            yield sse("error", {"message": str(exc)})

    return EventSourceResponse(event_stream())


@app.post("/api/slides")
def slides(request: SlidesRequest):
    try:
        return {"slides": generate_slides(request.report)}
    except Exception as exc:
        logger.exception("Slide generation failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/evaluate")
def evaluate(request: EvaluateRequest):
    try:
        return evaluate_research(request.query, request.report, [s.model_dump() for s in request.sources])
    except Exception as exc:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
