"""Week 10: disguised versions of the val and test sets.

Writes data/test_obfuscated.jsonl (for reporting results) and
data/val_obfuscated.jsonl (for CHOOSING whether to use normalization and
rules, so that choice never looks at test data).

For each disguise in pidetect/obfuscate.py (zero_width, hidden_html, base64,
homoglyph):
  - 150 INJECTED test docs (50 each from AgentDojo, InjecAgent, BIPIA) with
    only the injected words disguised. The agent could still read and follow
    them; the question is whether the detector still catches them.
  - 150 BENIGN test docs with a random 8-20 word stretch disguised the same
    way. Normal text can contain these too (hidden menu text on web pages,
    invisible characters from copy-paste, encoded attachments), so a detector
    that flags the disguise itself, not the content, gets false alarms here.

Each file is built only from its own split, so the test file inherits all
of the test set's leakage guarantees. Fixed seeds: identical on every run.

Usage:
    python scripts/make_obfuscated_test.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pidetect import obfuscate  # noqa: E402

SEEDS = {"test": 31, "val": 37}
PER_SOURCE_POS = 50
N_NEG = 150


def disguise_spans(text: str, spans: list[list[int]], transform: str, rng: random.Random):
    """Disguise each [start, end) word span; return the new text and new spans."""
    words = text.split()
    new_spans = []
    shift = 0
    out: list[str] = []
    cursor = 0
    for a, b in sorted(spans):
        out += words[cursor:a]
        disguised = obfuscate.apply(transform, " ".join(words[a:b]), rng).split(" ")
        start = len(out)
        out += disguised
        new_spans.append([start, len(out)])
        cursor = b
        shift += len(disguised) - (b - a)
    out += words[cursor:]
    return " ".join(out), new_spans


def build(split: str) -> list[dict]:
    rng = random.Random(SEEDS[split])
    docs = [json.loads(line) for line in Path(f"data/{split}.jsonl").open()]
    pos_by_src = {s: [r for r in docs if r["label"] == 1 and r["source"] == s] for s in ["agentdojo", "injecagent", "bipia"]}
    neg = [r for r in docs if r["label"] == 0 and r["n_words"] >= 12]
    out = []
    for transform in obfuscate.TRANSFORMS:
        for src, pool in pos_by_src.items():
            for r in rng.sample(pool, min(PER_SOURCE_POS, len(pool))):
                text, spans = disguise_spans(r["text"], r["inj_spans"], transform, rng)
                out.append({**r, "text": text, "inj_spans": spans, "transform": transform, "original_id": r["id"]})
        for r in rng.sample(neg, N_NEG):
            n = r["n_words"]
            length = rng.randint(8, min(20, n))
            a = rng.randint(0, n - length)
            text, spans = disguise_spans(r["text"], [[a, a + length]], transform, rng)
            # benign: the spans are where the disguise is, not an injection
            out.append({**r, "text": text, "inj_spans": [], "disguised_spans": spans, "transform": transform,
                        "original_id": r["id"]})
    for r in out:
        r["id"] = hashlib.sha1((r["transform"] + r["text"]).encode()).hexdigest()[:16]
        r["n_words"] = len(r["text"].split())
    return out


def main() -> int:
    for split in ["test", "val"]:
        out = build(split)
        path = Path(f"data/{split}_obfuscated.jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in out))
        print(f"Wrote {len(out)} docs to {path}")
        for t in obfuscate.TRANSFORMS:
            print(f"  {t:<12} injected {sum(r['transform'] == t and r['label'] == 1 for r in out)}, "
                  f"benign {sum(r['transform'] == t and r['label'] == 0 for r in out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
