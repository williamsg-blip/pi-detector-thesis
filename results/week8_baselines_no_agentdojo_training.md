# Week 8 detection baselines (trained on injecagent, bipia, pkgdocs only)

## Test set: overall

| Detector | Threshold rule | Precision | Recall | F1 | FPR | AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| Keyword filter | fixed | 100.0% | 35.5% | 52.4% | 0.0% | 0.677 |
| TF-IDF + LR (chunked) | val FPR <= 2% | 98.0% | 87.5% | 92.5% | 3.2% | 0.974 |
| TF-IDF + LR (whole doc) | val FPR <= 2% | 98.0% | 87.1% | 92.2% | 3.2% | 0.972 |

## Test set: AUROC by source

AUROC measures how well a detector ranks injected above benign, independent of the threshold (1.0 = perfect, 0.5 = random). Useful when one threshold can't fit every source.

| Detector | agentdojo | injecagent | bipia |
| --- | --- | --- | --- |
| Keyword filter | 0.750 | 0.750 | 0.500 (no signal) |
| TF-IDF + LR (chunked) | 0.915 | 1.000 | 0.914 |
| TF-IDF + LR (whole doc) | 0.914 | 1.000 | 0.907 |

## Test set: by slice

Recall = % of injected docs caught. FPR = % of benign docs wrongly flagged. n = injected/benign docs.

| Slice | n | Keyword filter | TF-IDF + LR (chunked) | TF-IDF + LR (whole doc) |
| --- | --- | --- | --- | --- |
| source=agentdojo | 216/26 | recall 50.0%, FPR 0.0% | recall 92.6%, FPR 15.4% | recall 92.6%, FPR 15.4% |
| source=injecagent | 272/136 | recall 50.0%, FPR 0.0% | recall 100.0%, FPR 0.0% | recall 100.0%, FPR 0.0% |
| source=bipia | 200/167 | recall 0.0%, FPR 0.0% | recall 65.0%, FPR 4.2% | recall 63.5%, FPR 4.2% |
| source=pkgdocs | 0/47 | FPR 0.0% | FPR 2.1% | FPR 2.1% |
| benign, instructional | 0/143 | FPR 0.0% | FPR 5.6% | FPR 5.6% |
| agentdojo, injected, long (>200 words) | 6/0 | recall 100.0% | recall 100.0% | recall 100.0% |
| bipia, injected, long (>200 words) | 76/0 | recall 0.0% | recall 60.5% | recall 56.6% |
| agentdojo, unseen style (system_message) | 36/0 | recall 0.0% | recall 80.6% | recall 80.6% |
| agentdojo, style=direct | 36/0 | recall 0.0% | recall 80.6% | recall 80.6% |
| agentdojo, style=ignore_previous | 36/0 | recall 0.0% | recall 100.0% | recall 100.0% |
| agentdojo, style=injecagent | 36/0 | recall 100.0% | recall 100.0% | recall 100.0% |
| agentdojo, style=important_instructions | 36/0 | recall 100.0% | recall 94.4% | recall 94.4% |
| agentdojo, style=tool_knowledge | 36/0 | recall 100.0% | recall 100.0% | recall 100.0% |
| injecagent, base | 136/0 | recall 0.0% | recall 100.0% | recall 100.0% |
| injecagent, enhanced | 136/0 | recall 100.0% | recall 100.0% | recall 100.0% |
