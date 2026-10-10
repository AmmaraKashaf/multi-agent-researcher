import json
import os
import random
import re
import time
from collections.abc import Iterator
from functools import lru_cache

import groq
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Groq has retired its Llama 70B chat models; gpt-oss-120b is the strongest general model available.
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Each Groq model has its own daily token quota, so fall back to these when one runs out.
FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GROQ_FALLBACK_MODELS", "openai/gpt-oss-20b,qwen/qwen3.8-27b").split(",")
    if m.strip()
]
MODELS = list(dict.fromkeys([MODEL, *FALLBACK_MODELS]))

MAX_ATTEMPTS = 6
# Longer waits mean a daily quota is exhausted; switch models instead of hanging the request.
MAX_RATE_LIMIT_WAIT = 90

# model -> time.time() when it can be used again (daily quota spent or model retired)
_unavailable_until: dict[str, float] = {}


@lru_cache(maxsize=1)
def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
    # Retries are handled in chat() so rate-limit waits can follow Groq's suggested delay.
    return Groq(api_key=api_key, max_retries=0)


def _retry_delay(error: groq.APIStatusError | groq.APIConnectionError, attempt: int) -> float:
    """Seconds to wait before retrying, preferring the delay Groq asks for."""
    response = getattr(error, "response", None)
    if response is not None:
        header = response.headers.get("retry-after")
        if header:
            try:
                return float(header)
            except ValueError:
                pass
        # e.g. "Please try again in 18.93s" or "in 1m2.5s"
        match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(error))
        if match:
            return int(match.group(1) or 0) * 60 + float(match.group(2))
    return 2.0 * 2**attempt


def _model_options(model: str) -> dict:
    # Hidden reasoning tokens count against the free tier's token limits, so keep them small.
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    if model.startswith("qwen/"):
        return {"reasoning_format": "hidden"}
    return {}


def _chat_with_retries(model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = get_client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **_model_options(model),
            )
            content = response.choices[0].message.content or ""
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            # gpt-oss sometimes leaves browsing-style citation markers such as 【1†L1-L2】.
            return re.sub(r"【[^】]{0,40}】", "", content).strip()
        except (groq.RateLimitError, groq.InternalServerError, groq.APIConnectionError) as error:
            delay = _retry_delay(error, attempt)
            if attempt == MAX_ATTEMPTS - 1 or delay > MAX_RATE_LIMIT_WAIT:
                raise
            time.sleep(delay + random.uniform(0.5, 2.0))
    raise RuntimeError("unreachable")


def _available_models() -> list[str]:
    now = time.time()
    return [m for m in MODELS if _unavailable_until.get(m, 0) <= now]


def _all_models_unavailable() -> RuntimeError:
    soonest = min(_unavailable_until.get(m, 0) for m in MODELS) - time.time()
    if soonest == float("inf"):
        return RuntimeError(f"None of the configured Groq models are available: {', '.join(MODELS)}")
    minutes, seconds = divmod(max(int(soonest), 0), 60)
    return RuntimeError(
        "All Groq models have hit their rate limits (free tier: 200,000 tokens per day per model). "
        f"Try again in {minutes}m {seconds}s, or upgrade at https://console.groq.com/settings/billing"
    )


def chat_messages(messages: list[dict], temperature: float = 0.3, max_tokens: int = 2048) -> str:
    for model in _available_models():
        try:
            return _chat_with_retries(model, messages, temperature, max_tokens)
        except groq.RateLimitError as error:
            _unavailable_until[model] = time.time() + _retry_delay(error, 0)
        except groq.NotFoundError:
            _unavailable_until[model] = float("inf")  # retired or not available to this key
    raise _all_models_unavailable()


def chat(system: str, user: str, temperature: float = 0.3, max_tokens: int = 2048) -> str:
    return chat_messages(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature,
        max_tokens,
    )


def stream_chat(messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024) -> Iterator[str]:
    """Yield response text as it streams. Retries and model fallback only happen before
    the first token; once text has been sent, an error is raised to the caller."""
    for model in _available_models():
        for attempt in range(MAX_ATTEMPTS):
            started = False
            try:
                stream = get_client().chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    **_model_options(model),
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        started = True
                        yield delta
                return
            except groq.NotFoundError:
                _unavailable_until[model] = float("inf")
                break
            except (groq.RateLimitError, groq.InternalServerError, groq.APIConnectionError) as error:
                if started:
                    raise
                delay = _retry_delay(error, attempt)
                if isinstance(error, groq.RateLimitError) and (delay > MAX_RATE_LIMIT_WAIT or attempt == MAX_ATTEMPTS - 1):
                    _unavailable_until[model] = time.time() + delay
                    break  # try the next model
                if attempt == MAX_ATTEMPTS - 1:
                    raise
                time.sleep(delay + random.uniform(0.5, 2.0))
    raise _all_models_unavailable()


def parse_json_list(text: str) -> list:
    """Extract the first JSON array from an LLM response, tolerating code fences and prose."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON list found in response: {text[:200]}")
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("Parsed JSON is not a list")
    return data


def parse_json_object(text: str) -> dict:
    """Extract the outermost JSON object from an LLM response, tolerating code fences and prose."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not an object")
    return data


def split_sources(report: str) -> tuple[str, str]:
    """Split a report into its body and the trailing '## Sources' section (built in code, not by the LLM)."""
    match = re.search(r"^## Sources\b", report, re.MULTILINE)
    if not match:
        return report.rstrip(), ""
    return report[: match.start()].rstrip(), report[match.start():].strip()


def format_search_results(search_results: list[dict]) -> str:
    blocks = []
    for result in search_results:
        sources = "\n".join(f"- {url}" for url in result.get("sources", [])) or "- (none)"
        blocks.append(
            f"### Subtopic: {result['subtopic']}\n"
            f"Findings: {result['findings']}\n"
            f"Sources:\n{sources}"
        )
    return "\n\n".join(blocks)
