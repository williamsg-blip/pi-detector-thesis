"""Week 8: split data/dataset.jsonl into train / val / test.

The rules live in pidetect/splits.py (read that file for the reasoning).
This script applies them, checks that no text or attacker goal leaks from
test into train, and prints a summary table.

Usage:
    python scripts/split_dataset.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pidetect.splits import assign_split  # noqa: E402

DATA = Path("data")


def goal_key(r: dict) -> str | None:
    """The attacker goal behind a positive sample (to check goals don't leak)."""
    if r["label"] != 1:
        return None
    if r["source"] == "bipia":
        # BIPIA groups are per context, so compare the inserted attack text itself
        words = r["text"].split()
        return "bipia-attack:" + " ".join(" ".join(words[a:b]) for a, b in r["inj_spans"])
    return r["group"]


def main() -> int:
    records = [json.loads(line) for line in (DATA / "dataset.jsonl").open()]
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    dropped = 0
    for r in records:
        s = assign_split(r)
        if s == "drop":
            dropped += 1
            continue
        r["split"] = s
        splits[s].append(r)

    # In val/test, keep at most 2 matched benign copies of the same AgentDojo
    # tool output (same user task + tool, different benign filler). Otherwise a
    # few near-identical documents can decide the threshold on their own.
    # Training keeps every copy.
    capped = 0
    for s in ["val", "test"]:
        seen: Counter = Counter()
        kept = []
        for r in sorted(splits[s], key=lambda r: r["id"]):
            if r["source"] == "agentdojo" and r["label"] == 0 and "filler_idx" in r:
                key = (r["subset"], r["user_task"], r["tool"])
                seen[key] += 1
                if seen[key] > 2:
                    capped += 1
                    continue
            kept.append(r)
        splits[s] = kept

    # ---- leakage checks
    ids = {s: {r["id"] for r in rs} for s, rs in splits.items()}
    assert not ids["train"] & ids["test"] and not ids["val"] & ids["test"], "same text in train and test"
    goals = {s: {goal_key(r) for r in rs} - {None} for s, rs in splits.items()}
    shared_goals = (goals["train"] | goals["val"]) & goals["test"]
    assert not shared_goals, f"attacker goals leaked into test: {sorted(shared_goals)[:5]}"
    fillers = {s: {r["filler_idx"] for r in rs if "filler_idx" in r} for s, rs in splits.items()}
    assert not (fillers["train"] | fillers["val"]) & fillers["test"], "benign filler text leaked into test"
    pkgs = {s: {r["group"] for r in rs if r["source"] == "pkgdocs"} for s, rs in splits.items()}
    assert not (pkgs["train"] | pkgs["val"]) & pkgs["test"], "README package split across train and test"

    for s, rs in splits.items():
        with (DATA / f"{s}.jsonl").open("w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")

    print(f"Dropped {dropped} 'system_message' samples from train/val (held-out attack style, test only)")
    print(f"Dropped {capped} near-duplicate matched benign copies from val/test (max 2 per tool output)\n")
    print(f"{'split':<7}{'source':<12}{'injected':>10}{'benign':>9}{'instructional benign':>23}")
    for s, rs in splits.items():
        c = Counter((r["source"], r["label"]) for r in rs)
        for src in ["agentdojo", "injecagent", "bipia", "pkgdocs"]:
            instr = sum(1 for r in rs if r["source"] == src and r["instructional"])
            print(f"{s:<7}{src:<12}{c[(src, 1)]:>10}{c[(src, 0)]:>9}{instr:>23}")
        print(f"{s:<7}{'TOTAL':<12}{sum(r['label'] for r in rs):>10}{sum(1 - r['label'] for r in rs):>9}\n")
    print("Leakage checks passed: no shared texts, attacker goals, or benign fillers between train and test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
