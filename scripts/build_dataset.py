"""Week 7: build the labeled dataset of injected and benign text.

Every record is one piece of "outside content" an agent might read (a tool
result, an email, a table, a README section), labeled:
    1 = contains an injected instruction aimed at the AI
    0 = benign

Sources (all public, see notes/data_card.md for details and licenses):
  agentdojo  Tool outputs from AgentDojo v1.2.2. For each (user task, injection
             task, attack style), the reference solution is run against the
             poisoned environment. Tool outputs that changed are positives;
             the same calls on the clean environment give negatives.
  injecagent InjecAgent (Zhan et al., 2024) tool responses with the attacker
             instruction (positives). Negatives: the SAME 17 response templates
             with the attack slot filled by benign text, so the model can't
             cheat by learning which templates are attacked.
  bipia      BIPIA (Yi et al., 2023): real emails, tables, and code Q&A
             contexts (negatives), plus the same contexts with a BIPIA attack
             inserted at the start, middle, or end (positives).
  pkgdocs    Hard negatives: README sections from installed Python packages.
             Real human-written text full of instructions aimed at a person
             ("Install with pip", "Run this command", "Make sure to...").

For positives we also store `inj_spans`: the word positions of the injected
text, found by diffing the injected text against its clean version. Training
uses this to label which chunks of a long document contain the injection.

Usage:
    git clone --depth 1 https://github.com/uiuc-kang-lab/InjecAgent.git external/InjecAgent
    git clone --depth 1 https://github.com/microsoft/BIPIA.git external/BIPIA
    python scripts/build_dataset.py
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import random
import re
import sys
import warnings
from collections import Counter
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pidetect.splits import TEST_SUITE, filler_split  # noqa: E402
from pidetect.textcues import looks_instructional  # noqa: E402

SEED = 13
BENCHMARK_VERSION = "v1.2.2"
AGENTDOJO_ATTACKS = [
    "direct",
    "ignore_previous",
    "system_message",
    "injecagent",
    "important_instructions",
    "tool_knowledge",
]


# ---------------------------------------------------------------- helpers
def sha(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def diff_spans(clean: str, injected: str) -> list[list[int]]:
    """Word ranges in `injected` that are not in `clean` (= where the injection is)."""
    a, b = clean.split(), injected.split()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    spans = [[j1, j2] for tag, _, _, j1, j2 in sm.get_opcodes() if tag in ("replace", "insert") and j2 > j1]
    # merge spans separated by only a few words (YAML line breaks can split one injection)
    merged: list[list[int]] = []
    for s in spans:
        if merged and s[0] - merged[-1][1] <= 3:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    # Drop 1-2 word side effects of the injection (a file-size field that grew,
    # YAML quote style changing). Real injections are always longer.
    real = [s for s in merged if s[1] - s[0] >= 3]
    return real or sorted(merged, key=lambda s: s[1] - s[0])[-1:]


def record(text, label, source, subset, group, style="none", inj_spans=None, **extra) -> dict:
    return {
        "id": sha(text),
        "text": text,
        "label": int(label),
        "source": source,
        "subset": subset,
        "group": group,  # used to keep related samples on the same side of the split
        "attack_style": style,
        "inj_spans": inj_spans or [],
        "instructional": bool(label == 0 and looks_instructional(text)),
        "n_words": len(text.split()),
        **extra,
    }


# ---------------------------------------------------------------- AgentDojo
# When a reference solution creates/edits a file or sends an email, AgentDojo
# stamps it with the REAL current time (datetime.now()). Left alone, every
# build would produce different text and the dataset would contain the build
# date. Real clock timestamps are the only ones with microseconds (AgentDojo's
# own data has none), so they are replaced with one fixed time inside
# AgentDojo's simulated timeline.
_REAL_NOW = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\.\d{6}")
FIXED_NOW = "2024-05-15 12:00:00"


def normalize_clock(text: str) -> str:
    return _REAL_NOW.sub(FIXED_NOW, text)


def build_agentdojo() -> list[dict]:
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suites
    from agentdojo.types import get_text_content_as_str

    class NamedPipeline(BasePipelineElement):
        """Attacks address the model by name ("...to you, GPT-4"), read from the pipeline name."""

        name = "gpt-4o-2024-05-13"

        def query(self, *a, **k):
            raise NotImplementedError

    def tool_outputs(suite, user_task, injections):
        env = suite.load_and_inject_default_environment(injections)
        env = user_task.init_environment(env)
        _, _, _, msgs, _ = GroundTruthPipeline(user_task).query(user_task.PROMPT, FunctionsRuntime(suite.tools), env)
        return [
            (m["tool_call"].function, normalize_clock(get_text_content_as_str(m["content"])))
            for m in msgs
            if m["role"] == "tool"
        ]

    rng = random.Random(SEED)
    out = []
    for suite_name, suite in get_suites(BENCHMARK_VERSION).items():
        attackers = {a: load_attack(a, suite, NamedPipeline()) for a in AGENTDOJO_ATTACKS}
        # Benign fillers allowed for this suite: test fillers only for the test
        # suite, train/val fillers for the others (see pidetect/splits.py).
        allowed = [
            k for k in range(len(INJECAGENT_BENIGN_FILLERS))
            if (filler_split(k) == "test") == (suite_name == TEST_SUITE)
        ]
        n_pos = n_neg = 0
        for user_task in suite.user_tasks.values():
            clean = tool_outputs(suite, user_task, {})
            for fn, text in clean:
                if text.strip():
                    out.append(record(text, 0, "agentdojo", suite_name, f"agentdojo/{suite_name}/{user_task.ID}",
                                      tool=fn, user_task=user_task.ID))
                    n_neg += 1
            # Matched benign versions: the same injection slots, filled with
            # benign text instead of an attack. Without these, nearly every
            # AgentDojo-formatted text in the data would be an attack.
            vectors = attackers["direct"].get_injection_candidates(user_task)
            for k in rng.sample(allowed, min(6, len(allowed))):
                filled = tool_outputs(suite, user_task, {v: INJECAGENT_BENIGN_FILLERS[k] for v in vectors})
                if len(filled) != len(clean):
                    continue
                for (fn, c_text), (_, f_text) in zip(clean, filled):
                    if c_text != f_text:
                        out.append(record(f_text, 0, "agentdojo", suite_name, f"agentdojo/{suite_name}/{user_task.ID}",
                                          tool=fn, user_task=user_task.ID, filler_idx=k))
                        n_neg += 1
            for inj_task in suite.injection_tasks.values():
                for style, attacker in attackers.items():
                    injected = tool_outputs(suite, user_task, attacker.attack(user_task, inj_task))
                    if len(injected) != len(clean):
                        continue  # calls diverged; can't pair outputs safely
                    for (fn, c_text), (_, i_text) in zip(clean, injected):
                        if c_text == i_text:
                            continue
                        out.append(record(
                            i_text, 1, "agentdojo", suite_name, f"agentdojo/{suite_name}/{inj_task.ID}",
                            style=style, inj_spans=diff_spans(c_text, i_text),
                            tool=fn, user_task=user_task.ID, injection_task=inj_task.ID,
                        ))
                        n_pos += 1
        print(f"  agentdojo/{suite_name}: {n_pos} injected outputs, {n_neg} clean outputs (before dedupe)")
    return out


# ---------------------------------------------------------------- InjecAgent
# Benign text for InjecAgent's attack slot (a review, an email body, a note,
# a repo description...). Written by the project author because InjecAgent
# ships no benign version. About half are instructions aimed at a PERSON, so
# negatives aren't trivially "no instructions". Split by index so test
# negatives use fillers never seen in training (see split_dataset.py).
INJECAGENT_BENIGN_FILLERS = [
    "Great product, works exactly as described. Battery easily lasts a full day.",
    "Arrived two days late but the quality makes up for it. Would buy again.",
    "Please remember to bring your ID and insurance card to the appointment.",
    "Reminder: the team meeting has moved to 3pm in conference room B.",
    "Don't forget to water the plants on the balcony while I'm away this weekend.",
    "A lightweight Python library for parsing and validating CSV files.",
    "Open-source dashboard for tracking personal fitness goals. Contributions welcome!",
    "Please review the attached draft and send me your comments by Friday.",
    "Thanks for your order! Your package will ship within 3-5 business days.",
    "Dr. Lee was patient and explained everything clearly. Highly recommend.",
    "Pick up milk, eggs, and bread on the way home.",
    "Software engineer, coffee lover, and weekend hiker. Opinions are my own.",
    "Call the landlord about the leaking faucet before the end of the month.",
    "Just finished my first marathon! Thanks to everyone who cheered me on.",
    "To reset your thermostat, hold the mode button for five seconds.",
    "The quarterly report is attached. Let me know if you have any questions.",
    "Bring a dish to share for the potluck on Saturday. Drinks will be provided.",
    "Fixed a bug in the login form and updated the README with setup steps.",
    "Make sure to submit your timesheet before noon on Monday.",
    "Loved the new menu at the cafe downtown, especially the lemon tart.",
    "Please confirm whether you can attend the workshop next Tuesday.",
    "Wash on cold and tumble dry low. Do not bleach.",
    "Our office will be closed on Monday for the holiday.",
    "Read chapters 4 and 5 before class and come ready to discuss.",
    "This repository contains the code for our course project on graph search.",
    "The package was well protected and the instructions were easy to follow.",
    "Book a table for four at 7pm and ask for a seat by the window.",
    "Cancel the gym membership renewal if the price goes up again.",
    "Happy birthday, Sam! Hope you have an amazing day.",
    "Preheat the oven to 375 degrees and bake for 25 minutes.",
    "Remember to back up your laptop before installing the update.",
    "Front desk staff were friendly and the wait time was short.",
    "Can you send me the slides from yesterday's presentation?",
    "New blog post: five tips for keeping your houseplants alive in winter.",
    "Schedule the car's oil change for sometime next week.",
    "Please sign the permission slip and return it to school by Thursday.",
    "Comfortable shoes, but they run about half a size small.",
    "Grab an umbrella, it's supposed to rain all afternoon.",
    "Update your contact information in the HR portal by the end of the month.",
    "Weekly digest: three new issues opened, two pull requests merged.",
    "Ask the pharmacist whether the prescription can be refilled early.",
    "Team lunch is on Friday, reply with your order by Wednesday.",
    "Charge the drone batteries fully before the flight on Sunday.",
    "Excellent customer service, they replaced the broken part right away.",
    "Turn off the lights and lock the back door when you leave.",
    "Follow the trail markers and stay on the path after the second bridge.",
    "Please print two copies of the contract for the meeting.",
    "The new version adds dark mode and fixes the export crash.",
]


def build_injecagent(root: Path) -> list[dict]:
    out = []
    data = root / "data"
    templates = {}
    for line in (data / "user_cases.jsonl").read_text().splitlines():
        case = json.loads(line)
        templates[case["User Tool"]] = case["Tool Response Template"]

    for fname in ["test_cases_dh_base", "test_cases_dh_enhanced", "test_cases_ds_base", "test_cases_ds_enhanced"]:
        style = "injecagent_" + fname.rsplit("_", 1)[1]  # base / enhanced
        for case in json.loads((data / f"{fname}.json").read_text()):
            text = case["Tool Response"].strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            clean = case["Tool Response Template"].replace("<Attacker Instruction>", "")
            out.append(record(
                text, 1, "injecagent", fname.split("_")[2],  # dh (direct harm) / ds (data stealing)
                "injecagent/" + sha(case["Attacker Instruction"]), style=style,
                inj_spans=diff_spans(clean, text), tool=case["User Tool"], attack_type=case["Attack Type"],
            ))

    for tool, template in templates.items():
        for k, filler in enumerate(INJECAGENT_BENIGN_FILLERS):
            text = template.replace("<Attacker Instruction>", filler)
            out.append(record(text, 0, "injecagent", "benign", f"injecagent/filler{k}", tool=tool, filler_idx=k))
    print(f"  injecagent: {sum(r['label'] for r in out)} injected, {sum(1 - r['label'] for r in out)} benign")
    return out


# ---------------------------------------------------------------- BIPIA
def insert_attack(context: str, attack: str, rng: random.Random, sep: str) -> tuple[str, str]:
    """Insert `attack` at the start, middle, or end of `context` on a sentence/line boundary."""
    where = rng.choice(["start", "middle", "end"])
    if where == "start":
        return attack + sep + context, where
    if where == "end":
        return context + sep + attack, where
    pieces = context.split("\n") if sep == "\n" else re.split(r"(?<=[.!?])\s+", context)
    cut = max(1, len(pieces) // 2)
    joiner = "\n" if sep == "\n" else " "
    return joiner.join(pieces[:cut]) + sep + attack + sep + joiner.join(pieces[cut:]), where


def build_bipia(root: Path, n_table_train: int = 250) -> list[dict]:
    rng = random.Random(SEED)
    bench = root / "benchmark"
    out = []

    def attacks(name, split):
        d = json.loads((bench / f"{name}_attack_{split}.json").read_text())
        return [(cat, i, a) for cat, items in d.items() for i, a in enumerate(items)]

    for split in ["train", "test"]:
        text_attacks, code_attacks = attacks("text", split), attacks("code", split)
        for task in ["email", "table", "code"]:
            rows = [json.loads(line) for line in (bench / task / f"{split}.jsonl").read_text().splitlines()]
            if task == "table" and split == "train":
                rows = rng.sample(rows, n_table_train)
            pool = code_attacks if task == "code" else text_attacks
            sep = "\n" if task in ("table", "code") else " "
            for k, row in enumerate(rows):
                ctx = row["context"]
                ctx = "\n".join(ctx) if isinstance(ctx, list) else ctx
                group = f"bipia/{split}/{task}/{k}"
                out.append(record(ctx, 0, "bipia", task, group, bipia_split=split))
                cat, idx, attack = rng.choice(pool)
                injected, where = insert_attack(ctx, attack, rng, sep)
                out.append(record(
                    injected, 1, "bipia", task, group, style=f"bipia_{cat}",
                    inj_spans=diff_spans(ctx, injected), bipia_split=split, position=where,
                    attack_group=f"bipia/{split}/{cat}/{idx}",
                ))
    print(f"  bipia: {sum(r['label'] for r in out)} injected, {sum(1 - r['label'] for r in out)} benign")
    return out


# ---------------------------------------------------------------- package docs
SKIP_PACKAGES = {"agentdojo"}  # its README quotes real attack strings
ATTACK_PHRASES = re.compile(r"ignore (all |your )?(previous|prior) instructions|prompt injection|jailbreak", re.I)


def build_pkgdocs(max_per_package: int = 3, min_words: int = 40, max_words: int = 250) -> list[dict]:
    out = []
    seen_pkgs = set()
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower()
        if not name or name in seen_pkgs or name in SKIP_PACKAGES:
            continue
        seen_pkgs.add(name)
        body = dist.metadata.get_payload() or dist.metadata.get("Description") or ""
        if not body or ATTACK_PHRASES.search(body):
            continue
        # sections split on markdown/rst headings or blank-line paragraphs
        sections = re.split(r"\n(?=#{1,4} )|\n(?=\S[^\n]*\n[=\-~^]{3,}\n)|\n\s*\n\s*\n", body)
        kept = 0
        for sec in sections:
            sec = sec.strip()
            n = len(sec.split())
            if min_words <= n <= max_words and looks_instructional(sec):
                out.append(record(sec, 0, "pkgdocs", "readme", f"pkgdocs/{name}", package=name))
                kept += 1
                if kept >= max_per_package:
                    break
    print(f"  pkgdocs: {len(out)} instructional README sections from {len({r['package'] for r in out})} packages")
    return out


# ---------------------------------------------------------------- main
def main() -> int:
    # AgentDojo stores some lists (e.g., event participants) as Python sets,
    # whose order changes between runs unless the hash seed is fixed. Restart
    # this script with a fixed seed so every build is byte-identical.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--external", type=Path, default=Path("external"), help="folder with InjecAgent/ and BIPIA/")
    parser.add_argument("--out", type=Path, default=Path("data/dataset.jsonl"))
    args = parser.parse_args()

    for repo in ["InjecAgent", "BIPIA"]:
        if not (args.external / repo).exists():
            print(f"Missing {args.external / repo}. Clone it first (see the usage notes at the top of this file).")
            return 1

    warnings.filterwarnings("ignore")
    print("Building dataset:")
    records = build_agentdojo() + build_injecagent(args.external / "InjecAgent") + build_bipia(args.external / "BIPIA")
    # README hard negatives depend on which packages happen to be installed, so
    # the first build saves a snapshot and later builds reuse it. Delete the
    # snapshot file to regenerate from the current machine.
    snapshot = args.out.parent / "pkgdocs_snapshot.jsonl"
    if snapshot.exists():
        pkg = [json.loads(line) for line in snapshot.open()]
        print(f"  pkgdocs: {len(pkg)} README sections loaded from {snapshot} (frozen snapshot)")
    else:
        pkg = build_pkgdocs()
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text("".join(json.dumps(r) + "\n" for r in pkg))
    records += pkg

    # Dedupe on exact text. If the same text appears as both labels (shouldn't
    # happen), drop it rather than guess.
    by_id: dict[str, dict] = {}
    conflicts = set()
    for r in records:
        prev = by_id.get(r["id"])
        if prev is None:
            by_id[r["id"]] = r
        elif prev["label"] != r["label"]:
            conflicts.add(r["id"])
    for cid in conflicts:
        del by_id[cid]
    final = list(by_id.values())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")

    print(f"\nRemoved {len(records) - len(final) - len(conflicts)} exact duplicates, {len(conflicts)} label conflicts")
    print(f"Wrote {len(final)} records to {args.out}\n")
    counts = Counter((r["source"], r["label"]) for r in final)
    print(f"{'source':<12}{'injected':>10}{'benign':>10}{'benign w/ instructions':>25}")
    for src in ["agentdojo", "injecagent", "bipia", "pkgdocs"]:
        instr = sum(1 for r in final if r["source"] == src and r["instructional"])
        print(f"{src:<12}{counts[(src, 1)]:>10}{counts[(src, 0)]:>10}{instr:>25}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
