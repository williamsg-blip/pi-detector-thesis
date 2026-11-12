"""Week 10: threshold tuning, rule checks, and the two new test sets.

Reads the score files from scripts/score_sets.py and answers four questions:

  1. Threshold sweep. How do catches and false alarms change as the cutoff
     moves (picked on validation at 0.5%, 1%, 2%, 5% false alarms)?
  2. Sensitive-topic benign text. Does the detector flag normal bank
     messages, password resets, and door-lock logs? (Tests the week 8 finding
     that TF-IDF learns the topic, not the attack.)
  3. Disguised attacks. Does it still catch injections hidden with invisible
     characters, hidden HTML, base64, or lookalike letters, and do
     normalization and the rules help?
  4. Rules on their own. Which rules catch disguises, and which cause false alarms?

It also picks an operating point per detector for week 11 using VALIDATION
data only (val + val_obfuscated; rule below) and saves it to
results/week10_operating_points.json.

Usage:
    python scripts/analyze_week10.py                     # every detector in results/scores/
    python scripts/analyze_week10.py --detectors tfidf deberta-v3-small
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pidetect.metrics import threshold_for_fpr  # noqa: E402
from pidetect.normalize import normalize  # noqa: E402
from pidetect.obfuscate import TRANSFORMS  # noqa: E402
from pidetect.rules import RULES, fired_rules  # noqa: E402

TARGETS = [0.005, 0.01, 0.02, 0.05]
MAIN_TARGET = 0.02  # the week 8-9 comparison point
MIN_VAL_RECALL = 0.95
SETS = ["val", "test", "sensitive_benign", "val_obfuscated", "test_obfuscated"]
FIXED_THRESHOLD = {"keyword": 1.0}  # detectors with a built-in cutoff


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def rate(flags, mask) -> float | None:
    mask = np.asarray(mask, bool)
    return float(np.asarray(flags)[mask].mean()) if mask.any() else None


def load_docs() -> dict[str, list[dict]]:
    return {s: [json.loads(line) for line in (Path("data") / f"{s}.jsonl").open()] for s in SETS}


def load_scores(det: str, docs: dict) -> dict[tuple[str, str], np.ndarray]:
    out = {}
    for s in SETS:
        for v in ["raw", "normalized"]:
            path = Path("results/scores") / det / f"{s}__{v}.jsonl"
            rows = [json.loads(line) for line in path.open()]
            assert [r["id"] for r in rows] == [d["id"] for d in docs[s]], f"{path} doesn't match data/{s}.jsonl"
            out[(s, v)] = np.array([r["score"] for r in rows])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detectors", nargs="*", default=None)
    parser.add_argument("--out", default="results/week10_analysis.md")
    args = parser.parse_args()

    docs = load_docs()
    dets = args.detectors or sorted(p.name for p in Path("results/scores").iterdir() if p.is_dir())
    labels = {s: np.array([d["label"] for d in docs[s]]) for s in SETS}
    # rules on the raw text (what the agent sees); normalization would erase what they look for
    rules_fired = {s: [fired_rules(d["text"]) for d in docs[s]] for s in SETS}
    rule_any = {s: np.array([bool(f) for f in rules_fired[s]]) for s in SETS}
    transform = np.array([d.get("transform", "") for d in docs["test_obfuscated"]])
    sens_cat = np.array([d["subset"] for d in docs["sensitive_benign"]])

    md = ["# Week 10: threshold tuning, rules, and new test sets", "",
          "Every cutoff is chosen on **validation** data and then applied unchanged to the test sets. "
          "*Catches* = % of injected docs flagged (recall). *False alarms* = % of benign docs flagged (FPR).", ""]
    operating_points = {}
    sens_rows, disguise_blocks = [], []

    for det in dets:
        sc = load_scores(det, docs)
        benign_val = sc[("val", "raw")][labels["val"] == 0]

        def flags(s, v, th, with_rules=False):
            f = sc[(s, v)] >= th
            return f | rule_any[s] if with_rules else f

        # ---- 1. threshold sweep
        md += [f"## {det}", "", "### Threshold sweep", "",
               "| Val false-alarm target | Threshold | Val catches | Test catches | Test false alarms | "
               "Sensitive-topic false alarms | Disguised catches | Disguised-benign false alarms |",
               "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        targets = [None] if det in FIXED_THRESHOLD else TARGETS
        sweep = {}
        for tgt in targets:
            th = FIXED_THRESHOLD[det] if tgt is None else threshold_for_fpr(benign_val, tgt)
            vf, tf, sf, of = (flags("val", "raw", th), flags("test", "raw", th),
                              flags("sensitive_benign", "raw", th), flags("test_obfuscated", "raw", th))
            row = {"threshold": th, "val_recall": rate(vf, labels["val"] == 1), "val_fpr": rate(vf, labels["val"] == 0)}
            sweep[tgt] = row
            md.append(f"| {'fixed' if tgt is None else pct(tgt)} | {th:.4g} | {pct(row['val_recall'])} | "
                      f"{pct(rate(tf, labels['test'] == 1))} | {pct(rate(tf, labels['test'] == 0))} | "
                      f"{pct(rate(sf, labels['sensitive_benign'] == 0))} | "
                      f"{pct(rate(of, labels['test_obfuscated'] == 1))} | {pct(rate(of, labels['test_obfuscated'] == 0))} |")

        # ---- operating point, chosen on validation only
        if det in FIXED_THRESHOLD:
            tgt, th = None, FIXED_THRESHOLD[det]
        else:
            ok = [t for t in TARGETS if sweep[t]["val_recall"] >= MIN_VAL_RECALL]
            tgt = min(ok) if ok else MAIN_TARGET
            th = sweep[tgt]["threshold"]
        # Normalization / rules: pick the variant with the best (catches - false alarms) on
        # validation PLUS disguised validation, so the cost of flagging disguised-but-benign
        # text counts. Ties go to the simpler variant.
        variants = [("model", False, False), ("+ normalization", True, False),
                    ("+ rules", False, True), ("+ normalization + rules", True, True)]
        md += ["", "**Choosing normalization and rules (validation data only):**", "",
               "| Variant | Val + disguised-val catches | Val + disguised-val false alarms | Score (catches - false alarms) |",
               "| --- | --- | --- | --- |"]
        best = None
        for vname, norm_on, rules_on_ in variants:
            v = "normalized" if norm_on else "raw"
            f_all = np.concatenate([flags("val", v, th, rules_on_), flags("val_obfuscated", v, th, rules_on_)])
            y_all = np.concatenate([labels["val"], labels["val_obfuscated"]])
            rec, fpr = rate(f_all, y_all == 1), rate(f_all, y_all == 0)
            md.append(f"| {vname} | {pct(rec)} | {pct(fpr)} | {100 * (rec - fpr):.1f} |")
            if best is None or rec - fpr > best[0] + 1e-3:
                best = (rec - fpr, vname, norm_on, rules_on_)
        val_score, chosen_variant, normalize_on, rules_on = best
        operating_points[det] = {"threshold": th, "val_fpr_target": tgt, "normalize": normalize_on, "rules": rules_on,
                                 "variant": chosen_variant, "val_score": round(val_score, 4),
                                 "score_dir": f"results/scores/{det}",
                                 **json.loads((Path("results/scores") / det / "config.json").read_text())}
        md += ["", f"**Chosen for week 11:** cutoff {th:.4g} "
                   f"({'fixed' if tgt is None else 'val false alarms ≤ ' + pct(tgt)}), variant **{chosen_variant}**. "
                   f"The cutoff is the strictest target that still catches ≥ {MIN_VAL_RECALL:.0%} of validation "
                   "injections; the variant is the best row above. Test data plays no part in either choice.", ""]

        # ---- 2. sensitive-topic benign, by category, at the chosen cutoff
        sf = flags("sensitive_benign", "normalized" if normalize_on else "raw", th, rules_on)
        sens_rows.append((det, [rate(sf, sens_cat == c) for c in sorted(set(sens_cat))], rate(sf, sens_cat != "")))

        # ---- 3. disguised attacks: each defense variant at the chosen cutoff
        block = [f"### {det}", "", "| Variant | Test catches | Test false alarms | Sensitive-topic false alarms | "
                 + " | ".join(f"{t} catches" for t in TRANSFORMS) + " | Disguised-benign false alarms |",
                 "| --- | --- | --- | --- | " + " | ".join("---" for _ in TRANSFORMS) + " | --- |"]
        for name, v, wr in [("model", "raw", False), ("+ normalization", "normalized", False),
                            ("+ rules", "raw", True), ("+ normalization + rules", "normalized", True)]:
            tf, sf2, of = flags("test", v, th, wr), flags("sensitive_benign", v, th, wr), flags("test_obfuscated", v, th, wr)
            pos = labels["test_obfuscated"] == 1
            block.append(f"| {name} | {pct(rate(tf, labels['test'] == 1))} | {pct(rate(tf, labels['test'] == 0))} | "
                         f"{pct(rate(sf2, labels['sensitive_benign'] == 0))} | "
                         + " | ".join(pct(rate(of, pos & (transform == t))) for t in TRANSFORMS)
                         + f" | {pct(rate(of, ~pos))} |")
        disguise_blocks += block + [""]

    # ---- assemble the cross-detector sections
    cats = sorted(set(sens_cat))
    md += ["## Sensitive-topic benign text (175 docs, all benign)", "",
           "False alarms at each detector's week 11 operating point. A good detector stays near 0%: "
           "these texts are about money, passwords, and locks, but contain no attack.", "",
           "| Detector | " + " | ".join(c.replace("_", " ") for c in cats) + " | All |",
           "| --- | " + " | ".join("---" for _ in cats) + " | --- |"]
    md += [f"| {d} | " + " | ".join(pct(x) for x in r) + f" | {pct(a)} |" for d, r, a in sens_rows]
    md += ["", "## Disguised attacks (data/test_obfuscated.jsonl)", "",
           "150 injected + 150 benign docs per disguise, all from the test split. "
           "Benign docs have a random stretch disguised the same way, so a detector that flags the *disguise* "
           "rather than the *attack* shows up as false alarms in the last column.", ""] + disguise_blocks

    # ---- which detector for week 11: ranked on validation only
    md += ["## Which detector to use in week 11", "",
           "Ranked by the validation score above (catches minus false alarms on validation + disguised validation, "
           "at each detector's chosen cutoff and variant). Test results play no part.", "",
           "| Rank | Detector | Variant | Validation score |", "| --- | --- | --- | --- |"]
    ranked = sorted(operating_points.items(), key=lambda kv: -kv[1]["val_score"])
    md += [f"| {i} | {k} | {v['variant']} | {100 * v['val_score']:.1f} |" for i, (k, v) in enumerate(ranked, 1)]
    md += [""]

    # ---- 4. rules alone
    def fire_rate(s, rule, mask):
        f = np.array([rule in fr for fr in rules_fired[s]])
        return rate(f, mask)

    md += ["## Rules on their own", "",
           "How often each rule fires. Ideally high on disguised attacks and near 0% on everything benign.", "",
           "| Rule | Test injected | Test benign | Sensitive-topic benign | "
           + " | ".join(f"{t} injected" for t in TRANSFORMS) + " | Disguised benign |",
           "| --- | --- | --- | --- | " + " | ".join("---" for _ in TRANSFORMS) + " | --- |"]
    pos = labels["test_obfuscated"] == 1
    for rule in RULES + ["any"]:
        fr = (lambda s, m: rate(rule_any[s], m)) if rule == "any" else (lambda s, m, r=rule: fire_rate(s, r, m))
        md.append(f"| {rule} | {pct(fr('test', labels['test'] == 1))} | {pct(fr('test', labels['test'] == 0))} | "
                  f"{pct(fr('sensitive_benign', labels['sensitive_benign'] == 0))} | "
                  + " | ".join(pct(fr('test_obfuscated', pos & (transform == t))) for t in TRANSFORMS)
                  + f" | {pct(fr('test_obfuscated', ~pos))} |")
    md += ["", "Note: `role_markers` was written knowing the held-out `system_message` attack style, so its test "
               "catches are optimistic (see pidetect/rules.py)."]

    Path(args.out).write_text("\n".join(md) + "\n")
    # merge, so analyzing a few detectors doesn't erase the others' operating points
    op_path = Path("results/week10_operating_points.json")
    merged = json.loads(op_path.read_text()) if op_path.exists() else {}
    merged.update(operating_points)
    op_path.write_text(json.dumps(merged, indent=2))
    print(Path(args.out).read_text())
    print("Saved results/week10_operating_points.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
