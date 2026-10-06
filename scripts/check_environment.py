"""Week 5: confirm the environment is set up correctly.

Runs without an API key. It checks that:
  1. AgentDojo is installed and the benchmark suites load.
  2. Each suite's built-in self-check passes (the "correct" answer for every
     task really does succeed, and injections are reachable).
  3. Optional training libraries (torch, transformers) are importable, and
     whether a GPU is available.

Usage:
    python scripts/check_environment.py
"""

import importlib
import sys

BENCHMARK_VERSION = "v1.2.2"


def check_package(name: str) -> bool:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "installed")
        print(f"  [ok]   {name} {version}")
        return True
    except ImportError:
        print(f"  [miss] {name} not installed (pip install -r requirements.txt)")
        return False


def main() -> int:
    print(f"Python {sys.version.split()[0]}\n")

    print("Packages:")
    if not check_package("agentdojo"):
        return 1
    for pkg in ["pandas", "transformers", "torch", "sklearn"]:
        check_package(pkg)

    try:
        import torch

        gpu = torch.cuda.is_available()
        print(f"  GPU available: {gpu}" + ("" if gpu else " (fine for now; use Colab GPU for training in week 9)"))
    except ImportError:
        pass

    from agentdojo.attacks.base_attacks import FixedJailbreakAttack
    from agentdojo.task_suite.load_suites import get_suites

    print(f"\nAgentDojo suites ({BENCHMARK_VERSION}):")
    suites = get_suites(BENCHMARK_VERSION)
    total_user, total_pairs, all_ok = 0, 0, True
    for name, suite in suites.items():
        n_user = len(suite.user_tasks)
        n_inj = len(suite.injection_tasks)

        # Ground-truth check: the reference solution for every task must succeed.
        # check_injectable=False because AgentDojo 0.1.35's built-in injectable
        # check has a bug (it expects tool output as a plain string, but tool
        # output is now stored as content blocks), so it always reports
        # "not injectable". We check injectability ourselves below instead.
        _, (user_results, inj_results) = suite.check(check_injectable=False)
        # Some injection tasks (workspace 6-13, added in v1.2) ship with no
        # reference solution (ground_truth returns []), so they can never pass
        # this self-check. That's expected; real attacks are still scored by
        # each task's security() function. Only count tasks that have one.
        env = suite.load_and_inject_default_environment({})
        no_reference = {
            tid for tid, task in suite.injection_tasks.items() if not task.ground_truth(env.model_copy(deep=True))
        }
        gt_ok = all(r[0] for r in user_results.values()) and all(
            passed for tid, passed in inj_results.items() if tid not in no_reference
        )

        # Injectability check, using the same function the real attacks use.
        attack = FixedJailbreakAttack("{goal}", suite, target_pipeline=None)
        injectable = sum(1 for t in suite.user_tasks.values() if attack.get_injection_candidates(t))

        ok = gt_ok and injectable == n_user
        all_ok &= ok
        total_user += n_user
        total_pairs += n_user * n_inj
        status = "ok" if ok else "FAILED"
        print(
            f"  [{status}] {name:<10} {n_user:>3} user tasks, {n_inj:>3} injection tasks, "
            f"{injectable}/{n_user} injectable, ground truth {'passes' if gt_ok else 'FAILS'}"
            + (f" ({len(no_reference)} injection tasks have no reference solution, expected)" if no_reference else "")
        )

    print(f"\n  Total: {total_user} user tasks, {total_pairs} (user task, injection task) pairs")
    print("\nEnvironment check " + ("PASSED" if all_ok else "FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
