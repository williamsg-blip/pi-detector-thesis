"""Evaluate any detector the same way and write the results.

Every detector (keyword filter, TF-IDF, Prompt Guard, our fine-tuned model)
produces a document score: the max over its chunks. This module:
  1. picks the decision threshold on the VALIDATION set, so that at most
     `target_fpr` of benign validation docs get flagged,
  2. applies that threshold unchanged to the TEST set,
  3. reports overall metrics and the slices the thesis cares about.

Fixed-threshold detectors (the keyword filter) skip step 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pidetect.metrics import detection_metrics, threshold_for_fpr
from pidetect.splits import HELDOUT_STYLE

TARGET_FPR = 0.02  # same 2% false-alarm budget as the AgentDojo community benchmark (#211)
LONG_DOC_WORDS = 200  # longer than one chunk


def slices(records: list[dict]) -> dict[str, list[int]]:
    """Named subsets of the test set (lists of indexes into `records`)."""
    idx = lambda f: [i for i, r in enumerate(records) if f(r)]  # noqa: E731
    out = {"all": list(range(len(records)))}
    for src in ["agentdojo", "injecagent", "bipia", "pkgdocs"]:
        out[f"source={src}"] = idx(lambda r, s=src: r["source"] == s)
    out["benign, instructional"] = idx(lambda r: r["label"] == 0 and r["instructional"])
    # Long docs, split by source: an unsplit "long" slice is mostly BIPIA and
    # would mix up "long" with "hard attack type".
    for src in ["agentdojo", "bipia"]:
        out[f"{src}, injected, long (>{LONG_DOC_WORDS} words)"] = idx(
            lambda r, s=src: r["source"] == s and r["label"] == 1 and r["n_words"] > LONG_DOC_WORDS
        )
    out[f"agentdojo, unseen style ({HELDOUT_STYLE})"] = idx(
        lambda r: r["source"] == "agentdojo" and r["attack_style"] == HELDOUT_STYLE
    )
    for style in ["direct", "ignore_previous", "injecagent", "important_instructions", "tool_knowledge"]:
        out[f"agentdojo, style={style}"] = idx(lambda r, s=style: r["source"] == "agentdojo" and r["attack_style"] == s)
    out["injecagent, base"] = idx(lambda r: r["attack_style"] == "injecagent_base")
    out["injecagent, enhanced"] = idx(lambda r: r["attack_style"] == "injecagent_enhanced")
    return {k: v for k, v in out.items() if v}


def evaluate(
    name: str,
    val_records: list[dict],
    val_scores: np.ndarray,
    test_records: list[dict],
    test_scores: np.ndarray,
    fixed_threshold: float | None = None,
    target_fpr: float = TARGET_FPR,
) -> dict:
    val_scores, test_scores = np.asarray(val_scores, float), np.asarray(test_scores, float)
    if fixed_threshold is None:
        benign_val = val_scores[[r["label"] == 0 for r in val_records]]
        threshold = threshold_for_fpr(benign_val, target_fpr)
    else:
        threshold = fixed_threshold
    y_val = np.array([r["label"] for r in val_records])
    y_test = np.array([r["label"] for r in test_records])
    result = {
        "detector": name,
        "threshold": threshold,
        "threshold_rule": "fixed" if fixed_threshold is not None else f"val FPR <= {target_fpr:.0%}",
        "val": detection_metrics(y_val, val_scores, threshold),
        "test": {},
    }
    for slice_name, ix in slices(test_records).items():
        result["test"][slice_name] = detection_metrics(y_test[ix], test_scores[ix], threshold)
    return result


def fmt_pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def markdown_summary(results: list[dict]) -> str:
    """Two tables: headline test metrics, then per-slice recall / FPR."""
    lines = ["## Test set: overall", "", "| Detector | Threshold rule | Precision | Recall | F1 | FPR | AUROC |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        m = r["test"]["all"]
        auroc = f"{m['auroc']:.3f}" if "auroc" in m else "-"
        lines.append(f"| {r['detector']} | {r['threshold_rule']} | {fmt_pct(m['precision'])} | {fmt_pct(m['recall'])} "
                     f"| {fmt_pct(m['f1'])} | {fmt_pct(m['fpr'])} | {auroc} |")
    lines += ["", "## Test set: AUROC by source", "",
              "AUROC measures how well a detector ranks injected above benign, independent of the threshold "
              "(1.0 = perfect, 0.5 = random). Useful when one threshold can't fit every source.", "",
              ]
    src_cols = [s for s in results[0]["test"]
                if s.startswith("source=") and any("auroc" in r["test"][s] for r in results)]
    lines.append("| Detector | " + " | ".join(s.split("=")[1] for s in src_cols) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in src_cols) + " |")
    for r in results:
        lines.append(f"| {r['detector']} | " + " | ".join(
            f"{r['test'][s]['auroc']:.3f}" if "auroc" in r["test"][s] else "0.500 (no signal)" for s in src_cols) + " |")
    slice_names = [s for s in results[0]["test"] if s != "all"]
    lines += ["", "## Test set: by slice", "",
              "Recall = % of injected docs caught. FPR = % of benign docs wrongly flagged. n = injected/benign docs.", "",
              "| Slice | n | " + " | ".join(r["detector"] for r in results) + " |",
              "| --- | --- | " + " | ".join("---" for _ in results) + " |"]
    for s in slice_names:
        m0 = results[0]["test"][s]
        n = f"{m0['n_pos']}/{m0['n'] - m0['n_pos']}"
        cells = []
        for r in results:
            m = r["test"][s]
            parts = []
            if m["n_pos"]:
                parts.append(f"recall {fmt_pct(m['recall'])}")
            if m["n"] - m["n_pos"]:
                parts.append(f"FPR {fmt_pct(m['fpr'])}")
            cells.append(", ".join(parts))
        lines.append(f"| {s} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save(results: list[dict], out_json: Path, out_md: Path, title: str) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    out_md.write_text(f"# {title}\n\n" + markdown_summary(results))


def load_split(name: str, data_dir: Path = Path("data")) -> list[dict]:
    return [json.loads(line) for line in (data_dir / f"{name}.jsonl").open()]
