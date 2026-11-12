"""Undo common disguises BEFORE the classifier reads the text.

Idea: instead of only flagging disguised text (pidetect/rules.py), clean it
up so the model sees what the agent would actually read:

  - remove invisible characters (zero-width spaces, soft hyphens, BOM, ...)
  - map Cyrillic/Greek lookalike letters back to Latin when they sit inside
    a mostly-Latin word
  - strip HTML tags (the hidden text inside stays and gets scored)
  - decode base64 runs that decode to readable text, and add the decoded text

The cleaned text is what gets scored; the original text is never changed for
the agent. Normalization should leave ordinary text almost untouched, and
week 10 checks that it doesn't raise false alarms.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata

INVISIBLE = re.compile("[­᠎​-‏⁠-⁤﻿]")
HTML_TAG = re.compile(r"</?[a-zA-Z][^<>]{0,300}>")
B64_RUN = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")

# Cyrillic/Greek letters that look like Latin ones
LOOKALIKE_TO_LATIN = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "і": "i", "ј": "j", "ѕ": "s", "һ": "h", "А": "A", "В": "B", "Е": "E",
    "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "ο": "o", "α": "a", "Α": "A", "Β": "B", "Ε": "E", "Ο": "O",
}


def decode_base64_runs(text: str) -> list[str]:
    """Readable texts hidden in base64 runs (binary blobs are ignored)."""
    found = []
    for m in B64_RUN.finditer(text):
        s = m.group(0)
        try:
            raw = base64.b64decode(s + "=" * (-len(s) % 4), validate=True)
            decoded = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        printable = sum(ch.isprintable() or ch in "\n\t" for ch in decoded) / max(1, len(decoded))
        letters = sum(ch.isalpha() for ch in decoded) / max(1, len(decoded))
        if len(decoded) >= 12 and printable > 0.97 and letters > 0.3 and " " in decoded:
            found.append(decoded)
    return found


def _fix_lookalikes(word: str) -> str:
    if not any(ch in LOOKALIKE_TO_LATIN for ch in word):
        return word
    latin = sum(1 for ch in word if "a" <= ch.lower() <= "z")
    other_script = sum(1 for ch in word if ch.isalpha() and not ("a" <= ch.lower() <= "z") and ch not in LOOKALIKE_TO_LATIN)
    # Fix words that mix in real Latin letters, or whose letters are ALL lookalikes.
    # Genuine Russian/Greek words contain other letters of their alphabet and are left alone.
    if other_script or (latin == 0 and len(word) > 3):
        return word
    return "".join(LOOKALIKE_TO_LATIN.get(ch, ch) for ch in word)


def normalize(text: str) -> str:
    text = INVISIBLE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(_fix_lookalikes(w) for w in text.split(" "))
    text = HTML_TAG.sub(" ", text)
    decoded = decode_base64_runs(text)
    if decoded:
        text = text + "\n" + "\n".join(decoded)
    return text
