"""Week 8: detection baselines that run anywhere (no GPU, no downloads).

  1. Keyword filter  - hand-written regex patterns (pidetect/keyword_filter.py)
  2. TF-IDF + logistic regression, trained on chunks, document score = max chunk
  3. Same TF-IDF model scoring the whole document at once (no chunking), to
     show whether chunking matters

Prompt Guard and the AgentDojo built-in detector need Hugging Face downloads
and a GPU, so they are in scripts/eval_hf_detectors.py (run in Colab).

Usage:
    python scripts/eval_baselines.py
    python scripts/eval_baselines.py --train-sources injecagent bipia   # cross-benchmark test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pidetect.chunking import chunk_text, label_chunks  # noqa: E402
from pidetect.keyword_filter import keyword_score  # noqa: E402
from pidetect.report import evaluate, load_split, save  # noqa: E402

SEED = 13


def chunk_max(records: list[dict], score_chunks) -> np.ndarray:
    """Score every chunk of every doc in one batch, then take the max per doc."""
    chunks, owner = [], []
    for i, r in enumerate(records):
        for c in chunk_text(r["text"]):
            chunks.append(c)
            owner.append(i)
    chunk_scores = np.asarray(score_chunks(chunks), float)
    doc = np.full(len(records), -np.inf)
    np.maximum.at(doc, np.asarray(owner), chunk_scores)
    return doc


def train_tfidf(train: list[dict]):
    texts, labels = [], []
    for r in train:
        for chunk, y in label_chunks(r["text"], r["label"], r["inj_spans"]):
            texts.append(chunk)
            labels.append(y)
    print(f"  TF-IDF training on {len(texts)} chunks ({sum(labels)} injected) from {len(train)} docs")
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100_000)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                  max_features=150_000)),
    ])
    model = make_pipeline(features, LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000,
                                                       random_state=SEED))
    model.fit(texts, labels)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-sources", nargs="*", default=None,
                        help="only train on these sources (default: all). Test always uses every source.")
    parser.add_argument("--out", default="results/week8_baselines")
    args = parser.parse_args()

    train, val, test = load_split("train"), load_split("val"), load_split("test")
    if args.train_sources:
        train = [r for r in train if r["source"] in args.train_sources]
        val = [r for r in val if r["source"] in args.train_sources]
    print(f"train {len(train)} docs, val {len(val)}, test {len(test)}")
    results = []

    # 1) keyword filter: fixed threshold, flag if any pattern matches
    t = time.time()
    results.append(evaluate("Keyword filter", val, chunk_max(val, lambda cs: [keyword_score(c) for c in cs]),
                            test, chunk_max(test, lambda cs: [keyword_score(c) for c in cs]), fixed_threshold=1.0))
    print(f"  keyword filter done ({time.time() - t:.0f}s)")

    # 2) TF-IDF + LR on chunks, max over chunks
    t = time.time()
    model = train_tfidf(train)
    prob = lambda texts: model.predict_proba(texts)[:, 1]  # noqa: E731
    results.append(evaluate("TF-IDF + LR (chunked)", val, chunk_max(val, prob), test, chunk_max(test, prob)))

    # 3) Same model, whole document as one input (no chunking)
    whole = lambda recs: prob([r["text"] for r in recs])  # noqa: E731
    results.append(evaluate("TF-IDF + LR (whole doc)", val, whole(val), test, whole(test)))
    print(f"  TF-IDF done ({time.time() - t:.0f}s)")

    title = "Week 8 detection baselines" + (f" (trained on {', '.join(args.train_sources)} only)"
                                            if args.train_sources else "")
    save(results, Path(args.out + ".json"), Path(args.out + ".md"), title)
    print(Path(args.out + ".md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
