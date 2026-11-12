"""Week 10: rule-based checks for things a text classifier tends to miss.

These flag HOW text is hidden or dressed up, not what it says:

  invisible_chars   3+ zero-width / invisible characters
  hidden_html       HTML styled to be invisible (display:none, font-size:0, white-on-white, ...)
  encoded_text      a base64 run that decodes to readable text
  mixed_script      2+ words mixing Latin letters with Cyrillic/Greek lookalikes
  role_markers      fake chat-format markers that pretend to be the system or
                    assistant (<|im_start|>, [INST], "### system", ...)

The idea (week 10 plan): rules should ADD recall on top of the model for
disguised attacks without adding false alarms on normal text. Each rule is
reported separately so you can see which ones pull their weight.

Honesty note for the paper: `role_markers` was written knowing that the
held-out AgentDojo `system_message` attack uses a "###(system_message)" header,
so its results on that slice are optimistic. The other four rules target
disguises that do not appear in the original test set at all.
"""

from __future__ import annotations

import re

from pidetect.normalize import INVISIBLE, LOOKALIKE_TO_LATIN, decode_base64_runs

HIDDEN_STYLE = re.compile(
    r"style\s*=\s*[\"'][^\"']*(display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(px|pt|em)?\b|"
    r"opacity\s*:\s*0(\.0+)?\b|color\s*:\s*(#fff(fff)?|white)\b[^\"']*background(-color)?\s*:\s*(#fff(fff)?|white))",
    re.I,
)
HIDDEN_ATTR = re.compile(r"<[a-z]+[^>]*\s(hidden|aria-hidden\s*=\s*[\"']true[\"'])[\s>]", re.I)
ROLE_MARKERS = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<\|system\|>|\[/?INST\]|<<SYS>>|"
    r"(^|\n)\s*#{2,}\s*\(?\s*(system|assistant)(_message)?\s*\)?|"
    r"(^|\n)\s*(system|assistant)\s*:\s",
    re.I,
)

RULES = ["invisible_chars", "hidden_html", "encoded_text", "mixed_script", "role_markers"]


def _mixed_words(text: str) -> int:
    n = 0
    for w in text.split():
        has_lookalike = any(ch in LOOKALIKE_TO_LATIN for ch in w)
        has_latin = any("a" <= ch.lower() <= "z" for ch in w)
        n += has_lookalike and has_latin
    return n


def fired_rules(text: str) -> list[str]:
    fired = []
    if len(INVISIBLE.findall(text)) >= 3:
        fired.append("invisible_chars")
    if HIDDEN_STYLE.search(text) or HIDDEN_ATTR.search(text):
        fired.append("hidden_html")
    if decode_base64_runs(text):
        fired.append("encoded_text")
    if _mixed_words(text) >= 2:
        fired.append("mixed_script")
    if ROLE_MARKERS.search(text):
        fired.append("role_markers")
    return fired


def rules_score(text: str) -> float:
    """1.0 if any rule fires, else 0.0."""
    return 1.0 if fired_rules(text) else 0.0
