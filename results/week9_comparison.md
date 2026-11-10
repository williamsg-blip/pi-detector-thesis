# Week 9: fine-tuned classifier vs. baselines

## Test set: overall

| Detector | Threshold rule | Precision | Recall | F1 | FPR | AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| Keyword filter | fixed | 100.0% | 35.5% | 52.4% | 0.0% | 0.677 |
| TF-IDF + LR (chunked) | val FPR <= 2% | 97.9% | 87.8% | 92.6% | 3.5% | 0.972 |
| TF-IDF + LR (whole doc) | val FPR <= 2% | 97.7% | 87.2% | 92.2% | 3.7% | 0.969 |
| Prompt Guard 2 (86M) | val FPR <= 2% | 98.0% | 43.6% | 60.4% | 1.6% | 0.844 |
| Prompt Guard 1 (86M) | val FPR <= 2% | 96.2% | 47.2% | 63.4% | 3.5% | 0.857 |
| ProtectAI DeBERTa v2 | val FPR <= 2% | 94.2% | 30.7% | 46.3% | 3.5% | 0.692 |
| deberta-v3-small (chunked) | val FPR <= 2% | 98.0% | 98.7% | 98.3% | 3.7% | 0.997 |

## Test set: AUROC by source

AUROC measures how well a detector ranks injected above benign, independent of the threshold (1.0 = perfect, 0.5 = random). Useful when one threshold can't fit every source.

| Detector | agentdojo | injecagent | bipia |
| --- | --- | --- | --- |
| Keyword filter | 0.750 | 0.750 | 0.500 (no signal) |
| TF-IDF + LR (chunked) | 0.961 | 1.000 | 0.888 |
| TF-IDF + LR (whole doc) | 0.961 | 1.000 | 0.877 |
| Prompt Guard 2 (86M) | 0.947 | 0.881 | 0.699 |
| Prompt Guard 1 (86M) | 0.659 | 0.832 | 0.909 |
| ProtectAI DeBERTa v2 | 0.764 | 0.805 | 0.466 |
| deberta-v3-small (chunked) | 0.992 | 1.000 | 0.988 |

## Test set: by slice

Recall = % of injected docs caught. FPR = % of benign docs wrongly flagged. n = injected/benign docs.

| Slice | n | Keyword filter | TF-IDF + LR (chunked) | TF-IDF + LR (whole doc) | Prompt Guard 2 (86M) | Prompt Guard 1 (86M) | ProtectAI DeBERTa v2 | deberta-v3-small (chunked) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| source=agentdojo | 216/26 | recall 50.0%, FPR 0.0% | recall 94.9%, FPR 30.8% | recall 94.9%, FPR 30.8% | recall 65.3%, FPR 3.8% | recall 33.3%, FPR 0.0% | recall 32.4%, FPR 3.8% | recall 100.0%, FPR 3.8% |
| source=injecagent | 272/136 | recall 50.0%, FPR 0.0% | recall 100.0%, FPR 0.0% | recall 100.0%, FPR 0.0% | recall 55.1%, FPR 2.2% | recall 51.1%, FPR 2.9% | recall 46.3%, FPR 0.0% | recall 100.0%, FPR 0.7% |
| source=bipia | 200/167 | recall 0.0%, FPR 0.0% | recall 63.5%, FPR 3.0% | recall 61.5%, FPR 3.6% | recall 4.5%, FPR 0.6% | recall 57.0%, FPR 5.4% | recall 7.5%, FPR 7.2% | recall 95.5%, FPR 6.6% |
| source=pkgdocs | 0/47 | FPR 0.0% | FPR 0.0% | FPR 0.0% | FPR 2.1% | FPR 0.0% | FPR 0.0% | FPR 2.1% |
| benign, instructional | 0/143 | FPR 0.0% | FPR 4.2% | FPR 4.2% | FPR 2.1% | FPR 6.3% | FPR 5.6% | FPR 7.0% |
| agentdojo, injected, long (>200 words) | 6/0 | recall 100.0% | recall 100.0% | recall 100.0% | recall 100.0% | recall 0.0% | recall 0.0% | recall 100.0% |
| bipia, injected, long (>200 words) | 76/0 | recall 0.0% | recall 53.9% | recall 48.7% | recall 9.2% | recall 68.4% | recall 0.0% | recall 98.7% |
| agentdojo, unseen style (system_message) | 36/0 | recall 0.0% | recall 75.0% | recall 75.0% | recall 2.8% | recall 0.0% | recall 2.8% | recall 100.0% |
| agentdojo, style=direct | 36/0 | recall 0.0% | recall 94.4% | recall 94.4% | recall 2.8% | recall 0.0% | recall 0.0% | recall 100.0% |
| agentdojo, style=ignore_previous | 36/0 | recall 0.0% | recall 100.0% | recall 100.0% | recall 100.0% | recall 100.0% | recall 88.9% | recall 100.0% |
| agentdojo, style=injecagent | 36/0 | recall 100.0% | recall 100.0% | recall 100.0% | recall 100.0% | recall 100.0% | recall 94.4% | recall 100.0% |
| agentdojo, style=important_instructions | 36/0 | recall 100.0% | recall 100.0% | recall 100.0% | recall 100.0% | recall 0.0% | recall 8.3% | recall 100.0% |
| agentdojo, style=tool_knowledge | 36/0 | recall 100.0% | recall 100.0% | recall 100.0% | recall 86.1% | recall 0.0% | recall 0.0% | recall 100.0% |
| injecagent, base | 136/0 | recall 0.0% | recall 100.0% | recall 100.0% | recall 10.3% | recall 2.2% | recall 0.0% | recall 100.0% |
| injecagent, enhanced | 136/0 | recall 100.0% | recall 100.0% | recall 100.0% | recall 100.0% | recall 100.0% | recall 92.6% | recall 100.0% |
