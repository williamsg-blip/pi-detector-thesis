"""Detection metrics shared by every baseline and model.

All scores are document-level: a document's score is the max over its chunks.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_for_fpr(scores_benign: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose false-positive rate on benign docs is <= target.

    Chosen on the VALIDATION set, then applied unchanged to the test set.
    Picking the threshold on test data would overstate results.
    """
    s = np.sort(np.asarray(scores_benign))[::-1]
    k = int(np.floor(target_fpr * len(s)))  # number of benign docs allowed above threshold
    if k >= len(s):
        return float(s[-1])
    # threshold strictly above the (k+1)-th highest benign score
    return float(np.nextafter(s[k], np.inf))


def detection_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {
        "n": len(y),
        "n_pos": int(y.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
    # Threshold-free metrics only make sense with both classes and non-constant scores
    if 0 < y.sum() < len(y) and len(np.unique(scores)) > 1:
        out["auroc"] = float(roc_auc_score(y, scores))
        out["auprc"] = float(average_precision_score(y, scores))
    return out
