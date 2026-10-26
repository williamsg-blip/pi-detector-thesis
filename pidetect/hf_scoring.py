"""Score text chunks with a Hugging Face sequence classifier.

Shared by eval_hf_detectors.py (off-the-shelf detectors like Prompt Guard)
and train_classifier.py (our fine-tuned model), so both are scored the
same way: each 200-word chunk gets P(injection), document = max chunk.
"""

from __future__ import annotations

import numpy as np

BENIGN_LABEL_NAMES = {"benign", "safe", "label_0", "0", "negative", "legit"}


def injection_prob_fn(model, tokenizer, device: str, batch_size: int = 32, max_length: int = 512):
    """Return a function: list[str] -> np.ndarray of P(not benign) per text.

    Detectors name their labels differently (Prompt Guard 1: BENIGN/INJECTION/
    JAILBREAK; Prompt Guard 2: LABEL_0/LABEL_1; ProtectAI: SAFE/INJECTION), so
    the score is 1 - P(benign label), which works for all of them.
    """
    import torch

    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    benign_ids = [i for i, name in id2label.items() if name in BENIGN_LABEL_NAMES]
    if len(benign_ids) != 1:
        raise ValueError(f"Can't tell which label is benign: {id2label}")
    benign = benign_ids[0]
    model.eval().to(device)

    def score(texts: list[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=max_length,
                                padding=True, return_tensors="pt").to(device)
                probs = torch.softmax(model(**enc).logits.float(), dim=-1)
                out.append((1 - probs[:, benign]).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    return score
