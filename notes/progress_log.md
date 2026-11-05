# Progress Log

## Week 5 (Sep 28 – Oct 2): Setup ✅
- Proposal written with linked sources and timeline.
- Set up Python environment and project repo (`requirements.txt`, `.gitignore`, `.env.example`).
- Installed AgentDojo 0.1.35 (benchmark version v1.2.2).
- Wrote `scripts/check_environment.py`. Confirmed: 4 suites, 97 user tasks, 949 (user task, injection task) pairs, all user tasks injectable, all reference solutions pass.
- Found two quirks in AgentDojo's built-in self-check (see README, "Notes on AgentDojo 0.1.35"). Neither affects real results; the check script works around both.

## Week 6 (Oct 5 – 9): Baseline + reading
- Wrote `scripts/run_baseline.py`. It runs every user task with no attack and with each injection task, saves one CSV row per run, and prints a summary table. The same script takes `--defense`, so it will be reused for baselines in week 8 and the detector in week 11.
- Validated the scoring with two fake agents (no API cost):

| Fake agent | Benign utility | Attack success rate |
| caffinate | 73.2% | 29.3% |
| `safe` (ignores attacks) | 100.0% | 0.0% |
| `obey` (follows attacks) | 100.0% | 64.6% |

  The `obey` agent's ASR is below 100% because some injection tasks have no reference solution for the fake agent to replay; a real LLM can still complete them.
- Reading notes written for 8 sources (`notes/reading_notes.md`).
- **To do:** real no-defense run with an LLM (needs API key). Start with `--limit 2`, then the full run.

### Real baseline results (fill in after the run)
| Suite | Benign utility | Utility under attack | Attack success rate |
| --- | --- | --- | --- |
| banking | | | |
| slack | | | |
| travel | | | |
| workspace | | | |
| **All** | | | |

Model: ______ · Attack: `important_instructions` · Date: ______ · Approx. cost: ______

## Week 7 (Oct 12 – 16): Dataset ✅
- Wrote `scripts/build_dataset.py`. It combines three public benchmarks plus real README text into **7,938 labeled samples** (5,598 injected, 2,340 benign). Details, licenses and limitations: `notes/data_card.md`.

| Source | Injected | Benign |
| --- | --- | --- |
| AgentDojo (tool outputs, 6 attack styles) | 2,940 | 691 |
| InjecAgent (17 tools, base + enhanced attacks) | 2,108 | 816 |
| BIPIA (real emails, tables, code + 150 attacks) | 550 | 493 |
| Python package READMEs (instructional hard negatives) | 0 | 340 |

- Every injected sample stores **where** the injection is (word positions, found by diffing against the clean version). Training uses this to label only the chunks that actually contain the attack.
- Benign versions of the *same* AgentDojo and InjecAgent outputs (the injection slot filled with benign text) keep the model from learning "this tool's format means attack."
- **Reproducibility bug found and fixed:** AgentDojo stamps edited files with the real clock time. Those timestamps made 1,386 duplicate tool outputs look unique. Fixed, plus a set-ordering issue; two independent builds are now byte-identical.

## Week 8 (Oct 19 – 23): Splits + baselines ✅ (local) · ⏳ Prompt Guard (Colab)
- `scripts/split_dataset.py` applies the rules in `pidetect/splits.py` and **asserts** there's no leakage between train and test (no shared texts, attacker goals, benign fillers or packages).
  - AgentDojo `banking` suite is test-only (unseen environment and goals).
  - `system_message` attack style is test-only (unseen template).

| Split | Injected | Benign |
| --- | --- | --- |
| train | 3,764 | 1,574 |
| val | 692 | 369 |
| test | 688 | 376 |

- `scripts/eval_baselines.py`: keyword filter and TF-IDF + logistic regression. Threshold picked on validation (≤ 2% false alarms), applied unchanged to test. Full tables: `results/week8_baselines.md`.

| Detector (test set) | Recall | False-positive rate | AUROC |
| --- | --- | --- | --- |
| Keyword filter | 35.5% | 0.0% | 0.677 |
| TF-IDF + LR (chunked) | 87.8% | 3.5% | 0.972 |
| TF-IDF + LR (whole doc) | 87.2% | 3.7% | 0.969 |

- **Findings to discuss with advisor:**
  1. The keyword filter catches 0% of BIPIA attacks and even misses AgentDojo's "ignore previous" attack, because that template contains a typo ("iunstructions"). Hand-written rules are brittle.
  2. **TF-IDF learns topic, not injection.** It flags 8 of 26 benign banking outputs like "Transaction to Spotify for 5.0 sent" (money words appear in many training attacks). Probing: "Please transfer $500 from my checking account" scores 0.96 with no injection present. Its 100% on InjecAgent is probably an artifact: InjecAgent attacks all request sensitive actions, while my benign fillers are mundane.
  3. BIPIA is the hardest source (TF-IDF recall 63.5%). Many BIPIA attacks look like normal requests ("translate your answer into French"); without knowing the user's task, a text-only detector can't always tell.
  4. Chunking made no difference for TF-IDF (it can read whole documents). The real test of chunking is the transformer (512-token limit). Caveat: the held-out banking suite has only 6 long injected outputs, so the "buried in a long output" case is mainly tested on long BIPIA contexts.
- `scripts/eval_hf_detectors.py` evaluates Prompt Guard 1, Prompt Guard 2 and ProtectAI's detector (the one AgentDojo uses) the same way. **Needs Colab + Hugging Face login** (Hugging Face was not reachable where this was written). Run `notebooks/colab_weeks8-9.ipynb`.

## Week 9 (Oct 26 – 30): Fine-tune classifier ✅ code · ⏳ GPU run (Colab)
- `scripts/train_classifier.py`: fine-tunes DeBERTa-v3-small on 200-word chunks with chunk-level labels and class weights. Documents are scored as the max chunk score, using the same threshold rule as the baselines. Saves per-document test scores for error analysis.
- `--no-chunking` runs the ablation (whole doc, truncated at 512 tokens).
- Verified with a pipeline check (`results/pipeline_check/`): a tiny model trained from scratch on CPU reached 0.98 validation AUROC, so the labels and training loop work. The pretrained run needs a GPU.
- **To do:** run the Colab notebook (about 45 min), then fill in:

| Detector (test set) | Recall | FPR | AUROC | AgentDojo FPR | BIPIA recall |
| --- | --- | --- | --- | --- | --- |
| Prompt Guard 2 | | | | | |
| ProtectAI | | | | | |
| DeBERTa-v3-small (chunked) | | | | | |
| DeBERTa-v3-small (no chunking) | | | | | |

## Plan adjustments for week 10
- Planned: rule-based checks + threshold tuning + midpoint check-in.
- **Add:** a small set of *sensitive but benign* hard negatives (bank notifications, real password-reset emails, smart-home logs) to test finding #2 directly. If the detectors flag them, that's a key result for the paper.
- Rule-based checks should target what the models miss (hidden/zero-width text, role tags like `###(system_message)`), and be judged by whether they *add* recall on top of the model without raising false alarms.
