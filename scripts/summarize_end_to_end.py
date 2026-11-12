"""Week 11: compare end-to-end AgentDojo runs side by side.

Each --run is LABEL=GLOB (several globs: separate with commas), pointing at the CSVs that run_baseline.py wrote
(one per suite is fine). For every run it reports benign utility, utility
under attack, attack success rate, and (for detector runs) false alarms,
split three ways:

  banking         the held-out suite the detector never trained on (the fairest number)
  seen suites     workspace + travel + slack (their environments were in the training data)
  all             everything

Usage:
    python scripts/summarize_end_to_end.py \\
        --run "No defense=results/nodefense_*.csv" \\
        --run "DeBERTa=results/defended_deberta_*.csv"
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

GROUPS = {
    "banking (held out)": {"banking"},
    "seen suites": {"workspace", "travel", "slack"},
    "all": {"banking", "workspace", "travel", "slack"},
}


def pct(xs) -> str:
    return f"{100 * sum(xs) / len(xs):.1f}%" if xs else "-"


def load(patterns: str) -> list[dict]:
    files = sorted({f for p in patterns.split(",") for f in glob.glob(p.strip())})
    if not files:
        raise SystemExit(f"No files match {patterns}")
    return [r for f in files for r in csv.DictReader(open(f))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="LABEL=GLOB of run_baseline.py CSVs")
    parser.add_argument("--out", default="results/week11_end_to_end.md")
    args = parser.parse_args()

    lines = ["# Week 11: end-to-end results in AgentDojo", "",
             "*Benign utility* = tasks completed with no attack. *Utility under attack* = tasks completed while "
             "attacked. *Attack success* = attacks where the attacker's goal happened (lower is better). "
             "*False alarms* = benign runs where the detector blocked at least one tool output.", "",
             "| Run | Attack | Suites | Benign utility | Utility under attack | Attack success | False alarms (benign runs) | n (benign / attacked) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for spec in args.run:
        label, _, pattern = spec.partition("=")
        rows = load(pattern)
        attacks = sorted({r["attack"] for r in rows if r["attack"] != "none"})
        for group, suites in GROUPS.items():
            rs = [r for r in rows if r["suite"] in suites]
            if not rs:
                continue
            benign = [r for r in rs if r["attack"] == "none"]
            attacked = [r for r in rs if r["attack"] != "none"]
            has_det = benign and benign[0].get("n_flagged", "") not in ("", None)
            fa = pct([int(r["n_flagged"]) > 0 for r in benign]) if has_det else "-"
            lines.append(f"| {label} | {', '.join(attacks)} | {group} | {pct([int(r['utility']) for r in benign])} | "
                         f"{pct([int(r['utility']) for r in attacked])} | {pct([int(r['attack_success']) for r in attacked])} | "
                         f"{fa} | {len(benign)} / {len(attacked)} |")
    lines += ["", "Compare rows with the same attack and suites. Runs that used a fake agent "
              "(`dryrun-gullible`) always do the user's task correctly, so their utility shows how often a blocked "
              "tool output *would* matter, not real task failures."]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(Path(args.out).read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
