"""Week 9: fine-tune a small transformer to detect injected instructions.

What it does:
  1. Splits every training document into 200-word chunks (pidetect/chunking.py).
     Each chunk is labeled 1 only if it actually contains injected text, using
     the injection positions saved in week 7.
  2. Fine-tunes DeBERTa-v3-small (44M parameters, plus a 98M-parameter
     embedding table; fast enough for a free Colab T4) to classify chunks.
  3. Scores documents as the max over their chunks, picks the threshold on
     validation (<= 2% false alarms), and reports test results in the same
     format as the week 8 baselines.

Usage (Colab, T4 GPU, about 10-20 minutes):
    python scripts/train_classifier.py
    python scripts/train_classifier.py --no-chunking      # ablation: whole doc, truncated at 512 tokens
    python scripts/train_classifier.py --model distilbert-base-uncased

Test that the code runs, on a laptop/CPU, no downloads (tiny random model, meaningless scores):
    python scripts/train_classifier.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pidetect.chunking import label_chunks  # noqa: E402
from pidetect.hf_scoring import injection_prob_fn  # noqa: E402
from pidetect.metrics import detection_metrics  # noqa: E402
from pidetect.report import evaluate, load_split, markdown_summary  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_baselines import chunk_max  # noqa: E402

LABELS = {0: "benign", 1: "injection"}


def make_examples(records: list[dict], chunked: bool) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    for r in records:
        if chunked:
            for chunk, y in label_chunks(r["text"], r["label"], r["inj_spans"]):
                texts.append(chunk)
                labels.append(y)
        else:
            texts.append(r["text"])  # the tokenizer truncates to max_length
            labels.append(r["label"])
    return texts, labels


def tiny_offline_model(train_texts: list[str], out_dir: Path):
    """For --smoke-test: a 2-layer BERT with a tokenizer trained on our own data. No internet."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(train_texts, trainers.WordPieceTrainer(
        vocab_size=4000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    from tokenizers.processors import TemplateProcessing
    tok.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[
        ("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    config = BertConfig(vocab_size=fast.vocab_size, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=128, max_position_embeddings=512, num_labels=2, id2label=LABELS,
                        label2id={v: k for k, v in LABELS.items()})
    out_dir.mkdir(parents=True, exist_ok=True)
    return BertForSequenceClassification(config), fast


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="microsoft/deberta-v3-small")
    parser.add_argument("--epochs", type=float, default=None, help="default 3 (1 for --smoke-test)")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-chunking", action="store_true", help="ablation: train/score whole docs (truncated)")
    parser.add_argument("--smoke-test", action="store_true", help="tiny offline model on a data subset")
    parser.add_argument("--smoke-sample", type=int, default=400,
                        help="training docs for --smoke-test (0 = all; val/test always full for 0)")
    parser.add_argument("--out", default=None, help="results prefix (default: results/week9_<name>)")
    args = parser.parse_args()

    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments, set_seed)

    set_seed(args.seed)
    random.seed(args.seed)
    chunked = not args.no_chunking
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    if args.smoke_test:
        if args.smoke_sample:
            rng = random.Random(0)
            train, val, test = rng.sample(train, args.smoke_sample), rng.sample(val, 150), rng.sample(test, 150)
        args.max_length = 256
    if args.epochs is None:
        args.epochs = 1 if args.smoke_test else 3

    train_texts, train_labels = make_examples(train, chunked)
    val_texts, val_labels = make_examples(val, chunked)
    print(f"Training examples: {len(train_texts)} ({sum(train_labels)} injected) "
          f"from {len(train)} docs; chunking={'on' if chunked else 'off'}; device={device}")

    name = "smoke-test" if args.smoke_test else args.model.split("/")[-1] + ("" if chunked else "-nochunk")
    model_dir = Path("models") / name
    if args.smoke_test:
        model, tokenizer = tiny_offline_model(train_texts, model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2, id2label=LABELS, label2id={v: k for k, v in LABELS.items()})

    class ChunkDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(texts, truncation=True, max_length=args.max_length)
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.labels[i]
            return item

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        p = torch.softmax(torch.tensor(logits).float(), -1)[:, 1].numpy()
        m = detection_metrics(labels, p, 0.5)
        return {k: m[k] for k in ("precision", "recall", "f1", "fpr") if k in m} | {"auroc": m.get("auroc", 0.0)}

    # class weights so the model isn't biased toward the more common label
    n_pos, n_neg = sum(train_labels), len(train_labels) - sum(train_labels)
    weights = torch.tensor([len(train_labels) / (2 * n_neg), len(train_labels) / (2 * n_pos)], dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    # 10% of training steps as learning-rate warmup. Given as a step count
    # because transformers 5 removed `warmup_ratio`; this works in 4.x and 5.x.
    steps_per_epoch = -(-len(train_texts) // args.batch_size)
    warmup_steps = max(1, int(0.1 * steps_per_epoch * args.epochs))
    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="auroc",
        fp16=device == "cuda",
        logging_steps=50,
        report_to=[],
        seed=args.seed,
    )
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ChunkDataset(train_texts, train_labels),
        eval_dataset=ChunkDataset(val_texts, val_labels),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    t = time.time()
    trainer.train()
    print(f"Training took {time.time() - t:.0f}s")
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    # Document-level evaluation, identical to the baselines
    score = injection_prob_fn(trainer.model, tokenizer, device, max_length=args.max_length)
    if chunked:
        val_scores, test_scores = chunk_max(val, score), chunk_max(test, score)
    else:
        val_scores, test_scores = score([r["text"] for r in val]), score([r["text"] for r in test])
    label = f"{args.model.split('/')[-1]} ({'chunked' if chunked else 'whole doc, truncated'})"
    result = evaluate("SMOKE TEST (meaningless)" if args.smoke_test else label, val, val_scores, test, test_scores)
    result["config"] = vars(args) | {"train_examples": len(train_texts), "train_seconds": round(time.time() - t)}

    out = Path(args.out or f"results/week9_{name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(json.dumps(result, indent=2))
    # Save per-document test scores, for error analysis and for week 10 threshold tuning
    Path(str(out) + "_test_scores.jsonl").write_text("".join(
        json.dumps({"id": r["id"], "label": r["label"], "source": r["source"], "score": float(s)}) + "\n"
        for r, s in zip(test, test_scores)))

    if not args.smoke_test:
        baselines = [json.loads(p.read_text()) for p in [Path("results/week8_baselines.json"),
                                                         Path("results/week8_hf_detectors.json")] if p.exists()]
        combined = [r for b in baselines for r in b] + [result]
        Path("results/week9_comparison.md").write_text(
            "# Week 9: fine-tuned classifier vs. baselines\n\n" + markdown_summary(combined))
        print(Path("results/week9_comparison.md").read_text())
    else:
        print(markdown_summary([result]))
        print("Smoke test finished: the training and evaluation code runs end to end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
