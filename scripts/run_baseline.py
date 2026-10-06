"""Week 6: run the no-defense baseline on AgentDojo.

Measures three numbers for an agent with NO defense:
  - Benign utility:        % of user tasks completed when there is no attack
  - Utility under attack:  % of user tasks still completed when an attack is present
  - Attack success rate:   % of (user task, injection task) pairs where the attacker's goal happened

These become the "before" numbers. In week 11 the same script runs again with
the detector plugged in (--defense), and the drop in attack success rate is the
main result of the thesis.

Every individual run is saved as a row in a CSV, so results can be re-analyzed
later without paying for the API calls again.

Examples:
    # Test the whole pipeline for free, no API key needed:
    python scripts/run_baseline.py --dry-run safe
    python scripts/run_baseline.py --dry-run obey

    # Small, cheap real run first (2 user tasks per suite):
    python scripts/run_baseline.py --limit 2

    # Full baseline (all 4 suites, 97 user tasks, 949 attack pairs):
    python scripts/run_baseline.py

    # Later baselines (week 8):
    python scripts/run_baseline.py --defense spotlighting_with_delimiting
    python scripts/run_baseline.py --defense transformers_pi_detector
"""

import argparse
import csv
import sys
import time
import warnings
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.types import ChatMessage

BENCHMARK_VERSION = "v1.2.2"
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_ATTACK = "important_instructions"  # the strongest built-in attack on the AgentDojo leaderboard


# --------------------------------------------------------------------------
# Dry-run agents: fake "agents" that replay AgentDojo's reference solutions.
# They need no API key and let us confirm the scoring code is correct before
# spending money:
#   safe: does only the user's task       -> expect ~100% utility, 0% attack success
#   obey: does the user's task, then the attacker's task -> expect attack success > 0
# --------------------------------------------------------------------------
class DryRunAgent(BasePipelineElement):
    def __init__(self, mode: str, name: str):
        self.mode = mode
        self.name = name  # attacks read the pipeline name to address "the model"
        self.user_task = None
        self.injection_task = None

    def set_tasks(self, user_task, injection_task) -> None:
        self.user_task = user_task
        self.injection_task = injection_task

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ):
        query, runtime, env, messages, extra_args = GroundTruthPipeline(self.user_task).query(
            query, runtime, env, messages, extra_args
        )
        if self.mode == "obey" and self.injection_task is not None:
            query, runtime, env, messages, extra_args = GroundTruthPipeline(self.injection_task).query(
                query, runtime, env, messages, extra_args
            )
        return query, runtime, env, messages, extra_args


def build_pipeline(args) -> BasePipelineElement:
    if args.dry_run:
        return DryRunAgent(args.dry_run, name=f"{args.model}-dryrun-{args.dry_run}")
    return AgentPipeline.from_config(
        PipelineConfig(
            llm=args.model,
            model_id=None,
            defense=args.defense,
            system_message_name=None,
            system_message=None,
        )
    )


def run(args) -> list[dict]:
    pipeline = build_pipeline(args)
    suites = get_suites(BENCHMARK_VERSION)
    selected = args.suites or list(suites)
    rows: list[dict] = []

    for suite_name in selected:
        suite = suites[suite_name]
        user_tasks = list(suite.user_tasks.values())[: args.limit] if args.limit else list(suite.user_tasks.values())
        injection_tasks = list(suite.injection_tasks.values())
        attacker = load_attack(args.attack, suite, pipeline)
        print(f"\n== {suite_name}: {len(user_tasks)} user tasks x {len(injection_tasks)} injection tasks ==")

        for i, user_task in enumerate(user_tasks, 1):
            # 1) No attack: can the agent do the job at all?
            if isinstance(pipeline, DryRunAgent):
                pipeline.set_tasks(user_task, None)
            utility, _ = run_one(suite, pipeline, user_task, None, {})
            rows.append(row(suite_name, user_task.ID, "none", "none", args, utility, None))

            # 2) With attack: pair this user task with every injection task
            successes = 0
            for injection_task in injection_tasks:
                injections = attacker.attack(user_task, injection_task)
                if isinstance(pipeline, DryRunAgent):
                    pipeline.set_tasks(user_task, injection_task)
                utility_atk, attack_success = run_one(suite, pipeline, user_task, injection_task, injections)
                successes += attack_success
                rows.append(
                    row(suite_name, user_task.ID, injection_task.ID, args.attack, args, utility_atk, attack_success)
                )
            print(
                f"  [{i}/{len(user_tasks)}] {user_task.ID:<14} benign utility={'Y' if utility else 'N'}  "
                f"attacks succeeded {successes}/{len(injection_tasks)}"
            )
    return rows


def run_one(suite, pipeline, user_task, injection_task, injections) -> tuple[bool, bool]:
    """Run one task, retrying on API errors (rate limits, timeouts)."""
    for attempt in range(3):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return suite.run_task_with_pipeline(pipeline, user_task, injection_task, injections)
        except Exception as e:  # noqa: BLE001 - API errors vary by provider
            wait = 10 * (attempt + 1)
            print(f"    error on {user_task.ID}/{getattr(injection_task, 'ID', 'none')}: {e!r}; retrying in {wait}s")
            time.sleep(wait)
    print("    giving up on this run; counting it as utility=False, attack_success=False")
    return False, False


def row(suite, user_task, injection_task, attack, args, utility, attack_success) -> dict:
    return {
        "suite": suite,
        "user_task": user_task,
        "injection_task": injection_task,
        "attack": attack,
        "model": args.model if not args.dry_run else f"dryrun-{args.dry_run}",
        "defense": args.defense or "none",
        "utility": int(bool(utility)),
        "attack_success": "" if attack_success is None else int(bool(attack_success)),
    }


def summarize(rows: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"{'suite':<11}{'benign utility':>16}{'utility under attack':>22}{'attack success rate':>22}")
    print("-" * 72)
    for suite in sorted({r["suite"] for r in rows}) + ["ALL"]:
        rs = rows if suite == "ALL" else [r for r in rows if r["suite"] == suite]
        benign = [r["utility"] for r in rs if r["attack"] == "none"]
        attacked = [r for r in rs if r["attack"] != "none"]
        pct = lambda xs: f"{100 * sum(xs) / len(xs):.1f}% (n={len(xs)})" if xs else "-"  # noqa: E731
        print(
            f"{suite:<11}{pct(benign):>16}{pct([r['utility'] for r in attacked]):>22}"
            f"{pct([r['attack_success'] for r in attacked]):>22}"
        )
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"AgentDojo model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--attack", default=DEFAULT_ATTACK, help=f"AgentDojo attack (default: {DEFAULT_ATTACK})")
    parser.add_argument("--defense", default=None, help="AgentDojo defense; leave empty for the no-defense baseline")
    parser.add_argument("--suites", nargs="*", help="Subset of suites: workspace travel banking slack")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N user tasks per suite (cheap test)")
    parser.add_argument("--dry-run", choices=["safe", "obey"], help="Fake agent, no API key needed")
    parser.add_argument("--out", type=Path, default=None, help="CSV path (default: results/<auto name>.csv)")
    args = parser.parse_args()

    if not args.dry_run and not load_dotenv(".env"):
        print("No .env file found. Copy .env.example to .env and add your API key, or use --dry-run.")
        return 1

    start = time.time()
    rows = run(args)

    name = f"dryrun_{args.dry_run}" if args.dry_run else f"baseline_{args.model}_{args.defense or 'nodefense'}"
    out = args.out or Path("results") / f"{name}{'_limit' + str(args.limit) if args.limit else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summarize(rows)
    print(f"\nSaved {len(rows)} rows to {out}  ({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
