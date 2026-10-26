# Week 8 detection baselines

## Test set: overall

| Detector | Threshold rule | Precision | Recall | F1 | FPR | AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| Keyword filter | fixed | 100.0% | 35.5% | 52.4% | 0.0% | 0.677 |
| TF-IDF + LR (chunked) | val FPR <= 2% | 97.9% | 87.8% | 92.6% | 3.5% | 0.972 |
| TF-IDF + LR (whole doc) | val FPR <= 2% | 97.7% | 87.2% | 92.2% | 3.7% | 0.969 |

## Test set: AUROC by source

AUROC measures how well a detector ranks injected above benign, independent of the threshold (1.0 = perfect, 0.5 = random). Useful when one threshold can't fit every source.

| Detector | agentdojo | injecagent | bipia |
| --- | --- | --- | --- |
| Keyword filter | 0.750 | 0.750 | 0.500 (no signal) |
| TF-IDF + LR (chunked) | 0.961 | 1.000 | 0.888 |
| TF-IDF + LR (whole doc) | 0.961 | 1.000 | 0.877 |

## Test set: by slice

Recall = % of injected docs caught. FPR = % of benign docs wrongly flagged. n = injected/benign docs.

| Slice | n | Keyword filter | TF-IDF + LR (chunked) | TF-IDF + LR (whole doc) |
| --- | --- | --- | --- | --- |
| source=agentdojo | 216/26 | recall 50.0%, FPR 0.0% | recall 94.9%, FPR 30.8% | recall 94.9%, FPR 30.8% |
| source=injecagent | 272/136 | recall 50.0%, FPR 0.0% | recall 100.0%, FPR 0.0% | recall 100.0%, FPR 0.0% |
| source=bipia | 200/167 | recall 0.0%, FPR 0.0% | recall 63.5%, FPR 3.0% | recall 61.5%, FPR 3.6% |
| source=pkgdocs | 0/47 | FPR 0.0% | FPR 0.0% | FPR 0.0% |
| benign, instructional | 0/143 | FPR 0.0% | FPR 4.2% | FPR 4.2% |
| agentdojo, injected, long (>200 words) | 6/0 | recall 100.0% | recall 100.0% | recall 100.0% |
| bipia, injected, long (>200 words) | 76/0 | recall 0.0% | recall 53.9% | recall 48.7% |
| agentdojo, unseen style (system_message) | 36/0 | recall 0.0% | recall 75.0% | recall 75.0% |
| agentdojo, style=direct | 36/0 | recall 0.0% | recall 94.4% | recall 94.4% |
| agentdojo, style=ignore_previous | 36/0 | recall 0.0% | recall 100.0% | recall 100.0% |
| agentdojo, style=injecagent | 36/0 | recall 100.0% | recall 100.0% | recall 100.0% |
| agentdojo, style=important_instructions | 36/0 | recall 100.0% | recall 100.0% | recall 100.0% |
| agentdojo, style=tool_knowledge | 36/0 | recall 100.0% | recall 100.0% | recall 100.0% |
| injecagent, base | 136/0 | recall 0.0% | recall 100.0% | recall 100.0% |
| injecagent, enhanced | 136/0 | recall 100.0% | recall 100.0% | recall 100.0% |
