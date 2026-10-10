from typing import Literal

from backend.agents.llm import chat, split_sources

Style = Literal["academic", "executive", "news"]

BASE_PROMPT = (
    "You rewrite research reports into a different format. Use ONLY the information in the "
    "report; never add facts, figures, names, or quotes. Whenever you reuse a claim tagged "
    "[VERIFIED] or [UNVERIFIED], copy its tag exactly, right after the claim. Output clean "
    "Markdown that starts with '# <title>'. Do NOT include a Sources section.\n\n"
)

STYLE_PROMPTS = {
    "academic": (
        "Format: an academic paper. Sections: Abstract (one paragraph), 1. Introduction, "
        "numbered body sections for the main themes, Discussion (including limitations), and "
        "Conclusion. Formal, objective, third-person tone."
    ),
    "executive": (
        "Format: an executive brief for busy decision-makers, at most about 400 words. "
        "Sections: Bottom Line (2-3 sentences), Key Findings (5-7 bullets), Implications, and "
        "Recommended Next Steps (bullets). Direct and concise."
    ),
    "news": (
        "Format: a news feature article. A compelling headline as the title, a one-sentence "
        "italic standfirst, a strong lede paragraph, body sections with short subheadings, and "
        "a closing paragraph on what to watch next. Engaging, journalistic tone; do not invent "
        "quotes or sources."
    ),
}


def reformat_report(report: str, style: str) -> str:
    body, sources = split_sources(report)
    rewritten = chat(
        BASE_PROMPT + STYLE_PROMPTS[style],
        f"Report to rewrite:\n\n{body}",
        temperature=0.4,
        max_tokens=2500,
    )
    # Keep the code-built Sources list so URLs are never altered by the model.
    return f"{rewritten.rstrip()}\n\n{sources}\n" if sources else rewritten
