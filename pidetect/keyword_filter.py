"""Baseline: a hand-written keyword/regex filter.

This is the kind of filter a developer might write in an afternoon: phrases
that commonly show up in prompt injections. It is deliberately generic (it does
not include the exact wording of any benchmark's attack template) so it is a
fair stand-in for "the simple approach". The patterns were written using only
the training split.

Score = number of distinct patterns that match. Flag if score >= 1.
"""

from __future__ import annotations

import re

PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|preceding|original)\b.{0,20}\b(instructions?|prompts?|directions|rules|task)",
    r"\bnew (instructions?|task)\s*:",
    r"\bimportant\s*!{2,}",
    r"\bimportant (message|instructions?|notice) (from|for)\b",
    r"\bbefore you (can )?(solve|complete|do|answer)\b.{0,30}\btask\b",
    r"\b(instead of|rather than)\b.{0,30}\b(your|the|original|current) task\b",
    r"</?(information|instructions?|system)>",
    r"(^|\n|\s)#*\s*\(?(system|assistant)( message)?\)?\s*:",
    r"\byou are now\b",
    r"\bfrom now on\b.{0,40}\b(you|reply|respond|answer)\b",
    r"\bdo not (tell|inform|mention|reveal)\b.{0,20}\b(the user|anyone)\b",
    r"\b(as an ai|ai assistant|language model)\b.{0,40}\b(must|should|need to)\b",
]
_COMPILED = [re.compile(p, re.I | re.S) for p in PATTERNS]


def keyword_score(text: str) -> float:
    return float(sum(1 for p in _COMPILED if p.search(text)))
