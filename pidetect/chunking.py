"""Split long inputs into overlapping word windows.

Why: AgentDojo tool outputs can be thousands of words (a whole inbox), but
transformer classifiers read at most ~512 tokens. An injection buried in the
middle of a long email would be cut off. So every input is split into
overlapping windows, each window is scored, and the document's score is the
HIGHEST window score ("max pooling"). One suspicious window is enough to flag
the whole input.

All models in this project (keyword filter, TF-IDF, transformers) use the same
chunking so they are compared fairly.

Window size is in words, not tokens. 200 words is roughly 260-320 DeBERTa
tokens, which leaves headroom under the 512-token limit even for YAML/code
text that tokenizes into more pieces.
"""

from __future__ import annotations

WINDOW_WORDS = 200
STRIDE_WORDS = 100  # 50% overlap: an injection near a boundary appears whole in some window


def word_spans(n_words: int, window: int = WINDOW_WORDS, stride: int = STRIDE_WORDS) -> list[tuple[int, int]]:
    """Return [start, end) word index pairs covering n_words with overlap."""
    if n_words <= window:
        return [(0, n_words)]
    spans = []
    start = 0
    while True:
        end = min(start + window, n_words)
        spans.append((start, end))
        if end == n_words:
            break
        start += stride
    return spans


def chunk_text(text: str, window: int = WINDOW_WORDS, stride: int = STRIDE_WORDS) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    return [" ".join(words[s:e]) for s, e in word_spans(len(words), window, stride)]


def label_chunks(
    text: str,
    label: int,
    inj_spans: list[list[int]] | None,
    window: int = WINDOW_WORDS,
    stride: int = STRIDE_WORDS,
    min_overlap: float = 0.5,
) -> list[tuple[str, int]]:
    """Chunk a document and give each chunk its own label (used for training).

    A benign document gives only benign chunks. For an injected document, only
    the chunks that actually contain an injection are labeled 1; the other
    chunks of the same document are ordinary benign text and are labeled 0.
    Without this, the model would learn that "long inbox listing" means
    "injection", which is wrong.

    `inj_spans` is a list of [start, end) word ranges, because one tool output
    can contain several injected spots (e.g., two poisoned emails in one inbox).
    A chunk counts as injected if it contains at least `min_overlap` of some
    injection's words, or if injection text fills at least `min_overlap` of the
    chunk (for injections longer than one window).
    """
    words = text.split()
    spans = word_spans(len(words), window, stride) if words else [(0, 0)]
    out = []
    for s, e in spans:
        chunk = " ".join(words[s:e])
        if label == 0 or not inj_spans:
            out.append((chunk, label))
            continue
        is_inj = False
        for a, b in inj_spans:
            overlap = max(0, min(e, b) - max(s, a))
            if overlap / max(1, b - a) >= min_overlap or overlap >= (e - s) * min_overlap:
                is_inj = True
                break
        out.append((chunk, int(is_inj)))
    return out
