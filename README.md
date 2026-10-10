# Multi-Agent Researcher

Ask a research question and a team of AI agents answers it with a fact-checked Markdown report:

1. **Orchestrator** splits the question into 4–5 subtopics
2. **Web searchers** search DuckDuckGo for each subtopic in parallel and extract key findings
3. **Synthesizer** merges the findings into themed paragraphs
4. **Fact checker** tags claims as `[VERIFIED]` or `[UNVERIFIED]` against the search results
5. **Report writer** produces the final report: Executive Summary, one section per subtopic, Conclusion, and Sources

The agents are built with [LangGraph](https://github.com/langchain-ai/langgraph) and run on [Groq](https://groq.com). A FastAPI backend streams each agent's progress to the browser using Server-Sent Events.

## Setup

Requires Python 3.10+.

### 1. Get a free Groq API key

1. Sign up at [console.groq.com](https://console.groq.com) (no credit card needed).
2. Go to [API Keys](https://console.groq.com/keys) and click **Create API Key**.
3. Copy the key (it starts with `gsk_`). It is only shown once.

The free tier has per-minute and per-day rate limits; see [console.groq.com/settings/limits](https://console.groq.com/settings/limits).

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your API key

```bash
cp .env.example .env             # Windows: copy .env.example .env
```

Open `.env` and replace `your_key_here` with your key:

```
GROQ_API_KEY=gsk_...
```

Put the key in `.env`, not `.env.example`. The app only reads `.env`, and `.env` is git-ignored so the key isn't committed.

### 4. Start the server

Run from the project root:

```bash
uvicorn backend.main:app --reload --reload-dir backend
```

Wait for `Application startup complete.` Keep this terminal open; closing it stops the server.

`--reload-dir backend` restarts the server only when your code changes. Plain `--reload` also watches `.venv`, which can cause unexpected restarts.

### 5. Open the UI

Go to **http://localhost:8000/** in your browser, or open `index.html` directly. Either way, the server from step 4 must be running.

A research run takes about 1–2 minutes.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Returns `{"status": "ok"}` |
| `GET` | `/api/info` | Configured model, fallback models, and subtopics per research mode |
| `POST` | `/api/research` | Body `{"query": "...", "mode": "deep"}`; streams Server-Sent Events |
| `POST` | `/api/insights` | Body `{"query", "report", "subtopics"}`; returns takeaways, statistics, follow-up questions, and mind-map facts as JSON |
| `POST` | `/api/reformat` | Body `{"report", "style"}` with style `academic`, `executive`, or `news`; returns the rewritten report |
| `POST` | `/api/evaluate` | Body `{"query", "report", "sources": [{"url", "title"}]}`; returns a confidence score with reason, per-source credibility (1-10) and bias ratings, and 5 related topics |
| `POST` | `/api/slides` | Body `{"report"}`; returns 5-6 slides as `[{"title", "bullets"}]` |
| `POST` | `/api/chat` | Body `{"query", "report", "question", "history"}`; streams the answer as SSE `token` events, then `done` |
| `GET` | `/docs` | Interactive API docs |

Research modes (`mode` in `/api/research`):

| Mode | Subtopics | Search |
|---|---|---|
| `deep` (default) | 5 | DuckDuckGo web search |
| `quick` | 3 | DuckDuckGo web search |
| `news` | 4 | DuckDuckGo News from the past month, falling back to recent web pages |

### Test with curl

Open a second terminal while the server is running:

```bash
curl -X POST http://localhost:8000/api/research \
  -H "Content-Type: application/json" \
  -d '{"query": "future of AI in healthcare"}' --no-buffer
```

On Windows PowerShell, use `curl.exe` and escape the inner quotes:

```powershell
curl.exe -X POST http://localhost:8000/api/research -H "Content-Type: application/json" -d '{\"query\": \"future of AI in healthcare\"}' --no-buffer
```

Events arrive as each agent finishes:

```
event: status
data: {"step": "start", "message": "Researching: future of AI in healthcare"}

event: status
data: {"step": "orchestrator", "message": "Planned 5 subtopics", "subtopics": [...]}

event: status
data: {"step": "web_searcher", "message": "Searched \"...\" (4 sources)", "subtopic": "...", "sources": [...]}

...

event: report
data: {"report": "# ...\n\n## Executive Summary\n...", "sources": [...]}
```

Lines starting with `: ping` are keep-alive comments sent every 15 seconds; clients ignore them.

| Event | Data |
|---|---|
| `status` | `step` (`start`, `orchestrator`, `web_searcher`, `synthesizer`, `fact_checker`, `report_writer`) and `message`. Orchestrator events also include `subtopics`; web searcher events include `subtopic` and `sources`. |
| `report` | Final event: `report` (Markdown) and `sources` (unique URLs) |
| `error` | `message` if the run fails |

## Configuration

Settings go in `.env`:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | required | Your Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Main Groq model used by every agent. Groq retires models regularly; see [console.groq.com/docs/models](https://console.groq.com/docs/models) for current ones. |
| `GROQ_FALLBACK_MODELS` | `openai/gpt-oss-20b,qwen/qwen3.8-27b` | Comma-separated models to switch to when the main model hits its daily token limit or is retired. Each model has its own daily quota. |

## Project structure

```
backend/
  agents/
    llm.py            Shared Groq client and helpers
    orchestrator.py   Splits the query into subtopics
    web_searcher.py   Searches DuckDuckGo for one subtopic, extracts findings
    synthesizer.py    Merges findings into themes
    fact_checker.py   Tags claims [VERIFIED] / [UNVERIFIED]
    report_writer.py  Writes the final Markdown report
    modes.py          Research modes (deep / quick / news) and subtopic counts
    insights.py       Extracts takeaways, statistics, follow-ups, mind-map facts
    formatter.py      Rewrites a report as academic / executive / news style
    evaluator.py      Confidence score, source credibility/bias, related topics
    slides.py         Turns a report into 5-6 presentation slides
    report_chat.py    Streams answers to questions about a report
  graph/
    state.py          Shared LangGraph state (ResearchState)
    pipeline.py       StateGraph wiring; exports research_graph
  main.py             FastAPI app: SSE endpoint, health check, serves index.html
index.html            Single-page UI (plain HTML/CSS/JS)
```

## Troubleshooting

- **`GROQ_API_KEY is not set`**: the key must be in a file named `.env` in the project root.
- **`model_not_found`**: Groq has retired the model. Set `GROQ_MODEL` in `.env` to a current one.
- **A subtopic shows "no results"**: DuckDuckGo sometimes rate-limits parallel searches. The searcher retries automatically; the rest of the report is still produced.
- **The UI says it can't reach the server**: check that uvicorn is still running on port 8000.
- **Code changes don't take effect**: on Windows, `--reload` can get stuck mid-restart and keep serving the old code. Two servers started on port 8000 can also both run at once. Stop all of them (Ctrl+C, or end the `python.exe` processes in Task Manager), then start a single server.
- **`Connection error` / `getaddrinfo failed`**: your computer briefly couldn't reach Groq (internet or DNS dropout). The agents retry for about a minute; if the outage lasts longer, run the query again.
- **Slow runs / `429 rate_limit_exceeded`**: Groq's free tier allows 8,000 tokens per minute, and one run uses more than that. The agents automatically wait and retry when they hit the limit, so a run takes about 1–2 minutes (longer if you start runs back to back). There is also a daily limit of 200,000 tokens per model (roughly a dozen reports). When the main model reaches it, the agents switch to the fallback models automatically. Only if every model is used up does the run fail, with a message saying when to try again; or upgrade at [console.groq.com/settings/billing](https://console.groq.com/settings/billing).
