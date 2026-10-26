"""Train / validation / test split rules, in one place.

The goal is an HONEST test set: nothing in test should be a near-copy of
something in training, or the scores would look better than they really are.

  AgentDojo   The whole `banking` suite is test only: its environment, user
              tasks and attacker goals never appear in training. Of the other
              three suites, one attacker goal per suite goes to validation.
              The `system_message` attack style is never trained on; it only
              appears in test, to measure how the detector handles an attack
              template it has never seen.
  InjecAgent  Split by attacker instruction (goal), so test goals are unseen.
  BIPIA       Uses BIPIA's own train/test files; 15% of train goes to validation.
  pkgdocs     Split by package, so no README is split across sets.
  Fillers     The author-written benign fillers are split by index. Test
              negatives only use test fillers, in every source.

Week 11 note: because three AgentDojo suites are used in training, the
end-to-end attack success numbers should be reported separately for
`banking` (fully held out) and the other suites.
"""

from __future__ import annotations

import hashlib

TEST_SUITE = "banking"
VAL_INJECTION_TASK = {"workspace": "injection_task_3", "travel": "injection_task_3", "slack": "injection_task_3"}
HELDOUT_STYLE = "system_message"


def bucket(key: str) -> float:
    """Deterministic number in [0, 1) from a string, so splits never change between runs."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0x100000000


def three_way(key: str, val: float = 0.15, test: float = 0.15) -> str:
    b = bucket(key)
    return "test" if b < test else "val" if b < test + val else "train"


def filler_split(k: int) -> str:
    """Fillers 0-47: every 6th goes to test, every 6th (offset 3) to val, the rest train."""
    return "test" if k % 6 == 0 else "val" if k % 6 == 3 else "train"


def assign_split(r: dict) -> str:
    src = r["source"]
    if "filler_idx" in r:
        return filler_split(r["filler_idx"])
    if src == "agentdojo":
        if r["subset"] == TEST_SUITE:
            return "test"
        if r["label"] == 1:
            if r["attack_style"] == HELDOUT_STYLE:
                return "drop"  # held-out style: only allowed in test
            return "val" if r["injection_task"] == VAL_INJECTION_TASK[r["subset"]] else "train"
        return "val" if bucket(r["group"]) < 0.15 else "train"
    if src == "injecagent":
        return three_way(r["group"])
    if src == "bipia":
        if r["bipia_split"] == "test":
            return "test"
        return "val" if bucket(r["group"]) < 0.15 else "train"
    if src == "pkgdocs":
        return three_way(r["group"])
    raise ValueError(f"unknown source {src}")
