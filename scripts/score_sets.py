"""Week 10: score every evaluation set with one detector and save the scores.

Scoring (slow, may need a GPU) is kept separate from analysis (fast, any
laptop): this script writes one small file per set, and
scripts/analyze_week10.py reads them.

Sets:     val, test, sensitive_benign, val_obfuscated, test_obfuscated
Variants: raw         the text as the agent would see it
          normalized  after pidetect/normalize.py undoes common disguises

Detectors:
  keyword                    the week 8 keyword filter
  tfidf                      week 8 TF-IDF + LR (trained here, saved to models/tfidf.pkl)
  hf:<model dir or HF id>    any Hugging Face classifier, e.g.
                             hf:models/deberta-v3-small  (your fine-tuned model)
                             hf:meta-llama/Llama-Prompt-Guard-2-86M

Output: results/scores/<name>/<set>__<variant>.jsonl  ({id, label, score} per doc)

Usage:
    python scripts/score_sets.py keyword
    python scripts/score_sets.py tfidf
    python scripts/score_sets.py hf:models/deberta-v3-small --name deberta-v3-small
    python scripts/score_sets.py hf:models/deberta-v3-small-nochunk --name deberta-v3-small-nochunk --no-chunking
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_baselines import chunk_max, train_tfidf  # noqa: E402
from pidetect.keyword_filter import keyword_score  # noqa: E402
from pidetect.normalize import normalize  # noqa: E402
from pidetect.report import load_split  # noqa: E402

SETS = ["val", "test", "sensitive_benign", "val_obfuscated", "test_obfuscated"]
TFIDF_PATH = Path("models/tfidf.pkl")


def load_tfidf():
    if TFIDF_PATH.exists():
        return pickle.loads(TFIDF_PATH.read_bytes())
    print("Training TF-IDF on the train split (same settings as week 8) ...")
    model = train_tfidf(load_split("train"))
    TFIDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    TFIDF_PATH.write_bytes(pickle.dumps(model))
    print(f"Saved {TFIDF_PATH}")
    return model


def make_scorer(spec: str):
    """Return (default name, function: list[str] -> np.ndarray of scores per text)."""
    if spec == "keyword":
        return "keyword", lambda texts: np.array([keyword_score(t) for t in texts], float)
    if spec == "tfidf":
        model = load_tfidf()
        return "tfidf", lambda texts: model.predict_proba(texts)[:, 1]
    if spec.startswith("hf:"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from pidetect.hf_scoring import injection_prob_fn

        path = spec[3:]
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).float()
        print(f"Loaded {path} on {device}")
        return Path(path).name, injection_prob_fn(model, tok, device)
    raise SystemExit(f"Unknown detector {spec!r}: use keyword, tfidf, or hf:<path>")


def load_set(name: str) -> list[dict]:
    path = Path("data") / f"{name}.jsonl"
    if not path.exists():
        hint = {"sensitive_benign": "build_sensitive_benign.py", "test_obfuscated": "make_obfuscated_test.py",
                "val_obfuscated": "make_obfuscated_test.py"}
        raise SystemExit(f"Missing {path}. Run scripts/{hint.get(name, 'split_dataset.py')} first.")
    return [json.loads(line) for line in path.open()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("detector")
    parser.add_argument("--name", help="folder name under results/scores (default: from the detector)")
    parser.add_argument("--no-chunking", action="store_true", help="score whole docs (for the --no-chunking model)")
    parser.add_argument("--sets", nargs="*", default=SETS, choices=SETS)
    args = parser.parse_args()

    default_name, score_texts = make_scorer(args.detector)
    out_dir = Path("results/scores") / (args.name or default_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    for set_name in args.sets:
        docs = load_set(set_name)
        for variant in ["raw", "normalized"]:
            t = time.time()
            texts = [d["text"] if variant == "raw" else normalize(d["text"]) for d in docs]
            if args.no_chunking:
                scores = score_texts(texts)
            else:
                proxies = [{"text": x} for x in texts]
                scores = chunk_max(proxies, score_texts)
            path = out_dir / f"{set_name}__{variant}.jsonl"
            path.write_text("".join(
                json.dumps({"id": d["id"], "label": d["label"], "score": float(s)}) + "\n" for d, s in zip(docs, scores)))
            print(f"  {set_name:<17} {variant:<11} {len(docs):>5} docs  ({time.time() - t:.0f}s) -> {path}")
    (out_dir / "config.json").write_text(json.dumps({"detector": args.detector, "no_chunking": args.no_chunking}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
