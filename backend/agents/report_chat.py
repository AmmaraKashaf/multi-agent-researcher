from collections.abc import Iterator

from backend.agents.llm import split_sources, stream_chat

SYSTEM_PROMPT = (
    "You answer questions about the research report below. Use ONLY the report: if it "
    "doesn't cover the question, say so plainly and suggest researching it as a new query. "
    "Be concise (under 150 words unless the user asks for more) and use Markdown bullets "
    "where they help. Claims in the report are tagged [VERIFIED] or [UNVERIFIED]; when an "
    "answer relies on an unverified claim, say that it is unverified.\n\n"
    "Research query: {query}\n\nREPORT:\n{report}"
)

MAX_HISTORY_MESSAGES = 8


def chat_about_report(query: str, report: str, question: str, history: list[dict]) -> Iterator[str]:
    body, _ = split_sources(report)
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(query=query, report=body)}]
    for message in history[-MAX_HISTORY_MESSAGES:]:
        messages.append({"role": message["role"], "content": message["content"][:2000]})
    messages.append({"role": "user", "content": question})
    yield from stream_chat(messages, temperature=0.3, max_tokens=900)
