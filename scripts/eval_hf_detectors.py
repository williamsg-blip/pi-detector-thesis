"""Week 8: off-the-shelf detector baselines (run in Google Colab with a GPU).

  prompt-guard-2   meta-llama/Llama-Prompt-Guard-2-86M  (Meta, 2025)
  prompt-guard-1   meta-llama/Prompt-Guard-86M          (Meta, 2024)
  protectai        protectai/deberta-v3-base-prompt-injection-v2
                   (the model behind AgentDojo's built-in `transformers_pi_detector` defense)

Same data, same chunking, and same threshold rule as eval_baselines.py, so the
numbers are directly comparable. Results are written next to the week 8
baselines in results/.

Prompt Guard is a "gated" model. Before running:
  1. Make a free Hugging Face account and open each Prompt Guard model page.
  2. Accept Meta's license on the page (approval is usually quick).
  3. Create a READ token at https://huggingface.co/settings/tokens and log in:
        from huggingface_hub import login; login()        # in Colab
     or set the HF_TOKEN environment variable.
The ProtectAI model is not gated.

Usage:
    python scripts/eval_hf_detectors.py                      # all three
    python scripts/eval_hf_detectors.py --detectors protectai
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pidetect.hf_scoring import injection_prob_fn  # noqa: E402
from pidetect.report import evaluate, load_split, markdown_summary  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_baselines import chunk_max  # noqa: E402

DETECTORS = {
    "prompt-guard-2": ("Prompt Guard 2 (86M)", "meta-llama/Llama-Prompt-Guard-2-86M"),
    "prompt-guard-1": ("Prompt Guard 1 (86M)", "meta-llama/Prompt-Guard-86M"),
    "protectai": ("ProtectAI DeBERTa v2", "protectai/deberta-v3-base-prompt-injection-v2"),
}


def main() -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detectors", nargs="*", default=list(DETECTORS), choices=list(DETECTORS))
    parser.add_argument("--out", default="results/week8_hf_detectors")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: no GPU found. This will be slow; in Colab use Runtime > Change runtime type > T4 GPU.")
    val, test = load_split("val"), load_split("test")
    results = []
    for key in args.detectors:
        name, model_id = DETECTORS[key]
        t = time.time()
        print(f"Loading {model_id} ...")
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
        except OSError as e:
            print(f"  Could not load {model_id}. If it's a Prompt Guard model, accept the license on its "
                  f"Hugging Face page and log in with a token (see the top of this file).\n  {e}")
            continue
        score = injection_prob_fn(model, tok, device)
        results.append(evaluate(name, val, chunk_max(val, score), test, chunk_max(test, score)))
        print(f"  done in {time.time() - t:.0f}s: test recall {results[-1]['test']['all']['recall']:.1%}, "
              f"FPR {results[-1]['test']['all']['fpr']:.1%}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if not results:
        return 1
    # Merge with the week 8 local baselines if present, so one table has everything
    local = Path("results/week8_baselines.json")
    combined = (json.loads(local.read_text()) if local.exists() else []) + results
    Path(args.out + ".json").write_text(json.dumps(results, indent=2))
    Path("results/week8_all_baselines.md").write_text("# Week 8: all detection baselines\n\n" + markdown_summary(combined))
    print(Path("results/week8_all_baselines.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
