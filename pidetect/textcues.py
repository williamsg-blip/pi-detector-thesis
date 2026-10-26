"""Heuristic: does a benign text contain instructions aimed at a person?

Used only to TAG benign samples as "instructional" so the false-positive rate
can be reported separately on them (the hardest benign cases). It is not part
of any detector, and it is a rough heuristic, not ground truth.
"""

import re

IMPERATIVE_VERBS = (
    "add|ask|avoid|bake|book|bring|call|cancel|change|check|click|confirm|contact|copy|create|"
    "delete|do|don't|download|enable|enter|follow|forward|go|install|let|make|open|pick|"
    "please|print|read|register|remember|remove|reply|reset|return|review|run|save|schedule|"
    "see|select|send|set|sign|start|stop|submit|take|try|turn|update|upgrade|use|visit|wash"
)
_IMPERATIVE_START = re.compile(
    rf"(?:^|[.!?]\s+|\n\s*(?:[-*>]|\d+[.)])?\s*)(?:{IMPERATIVE_VERBS})\b", re.I | re.M
)
_PHRASES = re.compile(
    r"\b(please|make sure|be sure to|don't forget|remember to|you (?:should|must|need to|can|will need))\b", re.I
)


def looks_instructional(text: str) -> bool:
    return bool(_IMPERATIVE_START.search(text) or _PHRASES.search(text))
