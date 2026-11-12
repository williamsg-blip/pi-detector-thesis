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

### Real baseline results ✅
| Suite | Benign utility | Utility under attack | Attack success rate |
| --- | --- | --- | --- |
| banking | 50.0% (16) | 45.8% (144) | 47.2% (144) |
| slack | 85.7% (21) | 57.1% (105) | 64.8% (105) |
| travel | 50.0% (20) | 39.3% (140) | 32.9% (140) |
| workspace | 87.5% (40) | 37.0% (560) | 17.1% (560) |
| **All** | **73.2%** (97) | **40.9%** (949) | **29.3%** (949) |

Model: `gpt-4o-mini-2024-07-18` · Attack: `important_instructions` · Committed: 2026-11-05 · Approx. cost: ______ (from the OpenAI usage page)

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
- Ran on Colab and Kaggle. Full table: `results/week9_comparison.md`. Threshold = validation false alarms ≤ 2%.

| Detector (test set) | Recall | FPR | AUROC | AgentDojo FPR | BIPIA recall |
| --- | --- | --- | --- | --- | --- |
| Prompt Guard 2 | 43.6% | 1.6% | 0.844 | 3.8% | 4.5% |
| ProtectAI | 30.7% | 3.5% | 0.692 | 3.8% | 7.5% |
| DeBERTa-v3-small (chunked) | **98.7%** | 3.7% | **0.997** | 3.8% | **95.5%** |
| DeBERTa-v3-small (no chunking) | 97.5% | **0.3%** | 0.994 | 3.8% | 92.0% |

- **Findings:** fine-tuning beats every off-the-shelf detector by a wide margin (Prompt Guard 2 catches 43.6%). Chunking catches more injections in long documents (BIPIA long docs: 98.7% vs 88.2%) but raises false alarms (3.7% vs 0.3%), because more chunks give more chances for one to look suspicious. Week 10 settles that trade-off on validation data.
- Bugs found and fixed while running: newer `transformers` loads DeBERTa in fp16, which crashed fp16 training (fixed with `model.float()`); the no-chunking run overwrote the comparison table (fixed: the table is now rebuilt from every result file).

## Plan adjustments made after week 9
- Planned: rule-based checks + threshold tuning + midpoint check-in.
- **Add:** a small set of *sensitive but benign* hard negatives (bank notifications, real password-reset emails, smart-home logs) to test finding #2 directly. If the detectors flag them, that's a key result for the paper.
- Rule-based checks should target what the models miss (hidden/zero-width text, role tags like `###(system_message)`), and be judged by whether they *add* recall on top of the model without raising false alarms.

## Week 10 (Nov 2 – 6): Thresholds, rules, and two new test sets ✅ code · ⏳ DeBERTa scoring (Kaggle)
- **New test set 1, sensitive-topic benign** (`scripts/build_sensitive_benign.py`, 175 texts): bank notifications, password resets, smart-lock logs, receipts, IT help articles, personal notes, health portal notices. All benign, all about the actions attackers ask for. Author-written from templates; test-only.
- **New test set 2, disguised attacks** (`scripts/make_obfuscated_test.py`): test injections hidden with invisible characters, hidden HTML, base64, or lookalike letters (150 each), plus 150 benign docs per disguise with a random stretch disguised the same way. A matching set is built from validation data, so choices are never made on test data.
- **Normalization** (`pidetect/normalize.py`) undoes the disguises before the model reads the text. **Rules** (`pidetect/rules.py`) flag the disguises themselves.
- **Threshold sweep and operating point** (`scripts/analyze_week10.py`): cutoffs at 0.5 / 1 / 2 / 5% validation false alarms. The cutoff is the strictest one still catching ≥ 95% of validation injections; normalization and rules are kept only if they improve catches minus false alarms on validation + disguised validation. Saved to `results/week10_operating_points.json` for week 11.
- **Results so far** (keyword + TF-IDF, run locally; full tables in `results/week10_analysis.md`):

| TF-IDF at its operating point | Result |
| --- | --- |
| False alarms on sensitive-topic benign text | **10.3%** overall: **32%** of bank notifications, **24%** of security emails |
| Catches on invisible-character attacks | **8.7%** raw → **86.0%** with normalization |
| Catches on base64 attacks | 32.0% → 92.0% with normalization |
| Rules: catches on disguised attacks | ~100%, but also fire on **98%** of disguised *benign* docs |

- **Findings:**
  1. The sensitive-topic set confirms week 8: TF-IDF flags benign text about money and passwords.
  2. Normalization recovers disguised attacks with almost no extra false alarms, so it's kept.
  3. Rules detect the *disguise*, not the attack, so they're rejected on validation (false alarms jump to 62%). The one exception, `role_markers`, never fired on benign text, but it was written knowing the held-out attack style, so its catches are optimistic.
  4. My first operating-point rule always kept the rules, because validation had no disguised text. That was fixed by adding disguised *validation* data, a good example of why choices must be made on data that resembles the test.
- **To do (Kaggle, `notebooks/kaggle_weeks10-11.ipynb` steps 1–5):** retrain DeBERTa, score every set, run the analysis. Prompt Guard 2 is scored too for comparison.
- **Midpoint check-in:** bring the week 9 table and the week 10 findings above. Questions for the advisor:
  - Is author-written, template-based data acceptable for the sensitive-topic set, given it's clearly labeled?
  - Should the main end-to-end result use the held-out attack (fairer) or the default one (matches the week 6 baseline)? The plan below reports both.

## Week 11 (Nov 9 – 13): Detector inside the agent ✅ code · ⏳ real runs (Kaggle)
- `pidetect/agentdojo_defense.py`: the detector as an AgentDojo defense. It checks every tool output before the agent reads it, and blocked outputs are replaced with "Data omitted because a prompt injection was detected". Same scoring as weeks 8–10, using the week 10 operating point.
- `scripts/run_baseline.py --detector <name>` runs the agent with the detector. A new free mode, `--dry-run gullible`, uses a fake agent that follows any injection it can still see, so the detector is the only protection (a worst case, no API cost).
- `scripts/summarize_end_to_end.py` compares runs, split into **banking (held out)**, **seen suites**, and **all**.
- **Important design point:** the default AgentDojo attack (`important_instructions`) uses the same template as the training data, so every detector looks good against it; even the keyword filter stops it. The **`system_message`** attack was never trained on, and it's the real test.
- **Offline results (gullible fake agent, all 4 suites; `results/week11_offline.md`):**

| Detector | Attack success, default attack | Attack success, held-out attack | Benign banking runs with a false alarm |
| --- | --- | --- | --- |
| None | 64.6% | 64.6% | - |
| Keyword | 1.9% | **64.6%** (stops nothing) | 0% |
| TF-IDF | 0.0% | 2.0% (banking: 6.2%) | **50%** |

  TF-IDF stops most attacks but blocks a benign tool output in half of the benign banking runs (the topic problem again). In a real agent that would break those tasks.
- **To do (Kaggle steps 6–8):** offline runs with DeBERTa, then real-agent runs on banking:
  - A: detector vs. the default attack
  - B: no defense vs. the held-out attack
  - C: detector vs. the held-out attack
  - Optional: the other three suites
