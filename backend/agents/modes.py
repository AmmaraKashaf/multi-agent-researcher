from typing import Literal

Mode = Literal["deep", "quick", "news"]

DEFAULT_MODE: Mode = "deep"

# Number of subtopics the orchestrator plans (and so the number of parallel searches).
SUBTOPIC_COUNT: dict[str, int] = {
    "deep": 5,
    "quick": 3,
    "news": 4,
}


def normalize_mode(mode: str | None) -> str:
    return mode if mode in SUBTOPIC_COUNT else DEFAULT_MODE
