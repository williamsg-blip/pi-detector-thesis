# Detecting Indirect Prompt Injection in LLM Agents

Senior thesis project. A small, fast classifier that checks outside content (tool results, web pages, emails) before an AI agent reads it, and flags hidden instructions. Tested end to end in [AgentDojo](https://github.com/ethz-spylab/agentdojo).

**Research question:** Can a small classifier catch indirect prompt injections well enough to stop most attacks, without blocking so much normal content that the agent stops being useful?

## Project structure

```
pi-detector-thesis/
├── scripts/
│   ├── check_environment.py   # Week 5: confirms everything is installed and working
│   ├── run_baseline.py        # Week 6: end-to-end agent baseline in AgentDojo (reused in week 11)
│   ├── build_dataset.py       # Week 7: builds data/dataset.jsonl from 3 benchmarks + READMEs
│   ├── split_dataset.py       # Week 8: train/val/test split with leakage checks
│   ├── eval_baselines.py      # Week 8: keyword filter + TF-IDF baselines (CPU, no downloads)
│   ├── eval_hf_detectors.py   # Week 8: Prompt Guard 1/2 + ProtectAI baselines (Colab GPU)
│   ├── train_classifier.py    # Week 9: fine-tune DeBERTa-v3-small with chunking (GPU)
│   ├── build_sensitive_benign.py  # Week 10: benign text about money, passwords, locks
│   ├── make_obfuscated_test.py    # Week 10: disguised attacks (invisible chars, hidden HTML, base64, lookalikes)
│   ├── score_sets.py          # Week 10: score every set with one detector (GPU for DeBERTa)
│   ├── analyze_week10.py      # Week 10: threshold sweep, rules, operating points (CPU)
│   └── summarize_end_to_end.py    # Week 11: compare AgentDojo runs (banking held out)
├── pidetect/                  # shared code: chunking, metrics, split rules, reporting
├── notebooks/
│   ├── colab_weeks8-9.ipynb   # GPU parts of weeks 8-9 (Google Colab)
│   └── kaggle_weeks10-11.ipynb    # GPU parts of weeks 10-11 (Kaggle), incl. real agent runs
├── data/                      # train/val/test splits (committed), README snapshot
├── results/                   # result tables (.md) and raw numbers (.json, .csv)
├── notes/
│   ├── reading_notes.md       # Week 6: notes on key papers
│   ├── data_card.md           # Week 7: what's in the dataset, licenses, limitations
│   └── progress_log.md        # Weekly log for advisor check-ins
├── requirements.txt
├── .env.example               # Template for your API key
└── .gitignore
```

## Setup

### Option A: your own computer
```bash
git clone <your-repo-url>
cd pi-detector-thesis
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_environment.py
```

### Option B: Google Colab (free GPU, needed for training in week 9)
In a new notebook, set **Runtime → Change runtime type → T4 GPU**, then run:
```python
!git clone <your-repo-url>
%cd pi-detector-thesis
!pip install -q -r requirements.txt
!python scripts/check_environment.py
```
For the API key in Colab, use the key icon (Secrets) in the left sidebar instead of a `.env` file, and load it with `from google.colab import userdata`.

### Expected output of the environment check
```
AgentDojo suites (v1.2.2):
  [ok] workspace   40 user tasks,  14 injection tasks, 40/40 injectable, ground truth passes (8 ... expected)
  [ok] travel      20 user tasks,   7 injection tasks, 20/20 injectable, ground truth passes (1 ... expected)
  [ok] banking     16 user tasks,   9 injection tasks, 16/16 injectable, ground truth passes
  [ok] slack       21 user tasks,   5 injection tasks, 21/21 injectable, ground truth passes

  Total: 97 user tasks, 949 (user task, injection task) pairs
Environment check PASSED
```

## Running the baseline

**1. Test for free first (no API key):**
```bash
python scripts/run_baseline.py --dry-run safe   # fake agent that ignores attacks
python scripts/run_baseline.py --dry-run obey   # fake agent that follows attacks
```
`safe` should show ~100% utility and 0% attack success; `obey` should show high attack success. This confirms the scoring is correct.

**2. Add your API key:**
```bash
cp .env.example .env   # then paste your OpenAI key into .env
```

**3. Small real run (about 18 agent runs per suite):**
```bash
python scripts/run_baseline.py --limit 2
```
Check your usage on the OpenAI dashboard after this to estimate the cost of the full run.

**4. Full baseline (~1,050 agent runs):**
```bash
python scripts/run_baseline.py
```

The default agent model is `gpt-4o-mini-2024-07-18`, a small, cheap model in AgentDojo's supported list. Confirm it's still available from OpenAI before the full run; run `python -m agentdojo.scripts.benchmark --help` to see every supported model.

### What the numbers mean
| Metric | Meaning | Good defense should... |
| --- | --- | --- |
| Benign utility | % of tasks the agent completes with no attack | stay the same |
| Utility under attack | % of tasks completed while being attacked | stay high |
| Attack success rate (ASR) | % of attack attempts where the attacker's goal happened | go **down** |

## Notes on AgentDojo 0.1.35
Two things in the built-in self-check look like failures but aren't problems with the setup. `check_environment.py` handles both:
1. The built-in "is this task injectable?" check always says no, because it expects tool output as a plain string while this version stores it as content blocks. The script checks injectability with the same function the real attacks use instead.
2. Some injection tasks (workspace 6–13, one in travel) ship without a reference solution, so they can't pass the self-check. Real attacks are still scored normally.

## Detection experiments (weeks 7–9)

The detector is evaluated on its own first (weeks 8–10), then inside the agent (week 11).

**On your computer (CPU, a few minutes):**
```bash
python scripts/split_dataset.py       # re-creates data/{train,val,test}.jsonl and checks for leakage
python scripts/eval_baselines.py      # keyword + TF-IDF -> results/week8_baselines.md
python scripts/train_classifier.py --smoke-test   # 20-second check that the training code runs
```
The splits are already committed, so `build_dataset.py` only needs to run if you change how the data is made. It needs the two benchmark repos cloned into `external/` (instructions at the top of the script) and takes about 8 minutes.

**On Colab (T4 GPU, about 45 minutes):** open `notebooks/colab_weeks8-9.ipynb` and run the cells top to bottom. It evaluates Prompt Guard 1/2 and ProtectAI, fine-tunes DeBERTa-v3-small, runs the no-chunking ablation, and writes `results/week9_comparison.md` with every detector in one table. Prompt Guard needs a free Hugging Face account and license acceptance (steps are in the notebook).

**How every detector is scored (so comparisons are fair):**
1. Each document is split into 200-word chunks with 50% overlap; document score = highest chunk score.
2. The threshold is chosen on **validation** so at most 2% of benign docs are flagged, then applied unchanged to **test**.
3. Results are broken down by source, attack style, unseen attack style, long docs, and instructional benign text.

## Weeks 10–11: thresholds, new test sets, and the end-to-end test

**On your Mac (CPU, no downloads):**
```bash
python scripts/build_sensitive_benign.py      # data/sensitive_benign.jsonl (already committed)
python scripts/make_obfuscated_test.py        # data/{val,test}_obfuscated.jsonl (already committed)
python scripts/score_sets.py keyword && python scripts/score_sets.py tfidf
python scripts/analyze_week10.py              # -> results/week10_analysis.md + week10_operating_points.json
python scripts/run_baseline.py --dry-run gullible --attack system_message --detector tfidf   # free end-to-end test
```

**On Kaggle (GPU):** open `notebooks/kaggle_weeks10-11.ipynb` and run the cells in order. It retrains DeBERTa, scores every set, reruns the analysis, picks the detector on validation data, runs the free offline test, then (with an `OPENAI_API_KEY` secret) the real-agent runs on banking. Each step saves to GitHub when it finishes.

**What's new in the pipeline:**
- `pidetect/normalize.py` removes disguises (invisible characters, lookalike letters, HTML tags, base64) before scoring.
- `pidetect/rules.py` flags disguises directly. Week 10 shows why these weren't kept: they fire on disguised benign text too.
- `pidetect/agentdojo_defense.py` is the detector as an AgentDojo defense. Flagged tool outputs are replaced before the agent reads them.
- `run_baseline.py --detector <name>` uses the operating point saved in week 10. `--dry-run gullible` is a fake agent that follows any injection it can still see, for free worst-case testing.

**Always report two attacks:** the default `important_instructions` (matches the week 6 baseline, but its template is in the training data) and `system_message` (never trained on: the fair test).

## Timeline
| Week | Status | Focus |
| --- | --- | --- |
| 5 | ✅ Done | Environment, repo, environment check |
| 6 | ✅ Done | No-defense agent baseline (29.3% attack success; banking 47.2%), reading |
| 7 | ✅ Done | Dataset: 7,938 samples from AgentDojo, InjecAgent, BIPIA, READMEs |
| 8 | ✅ Done | Splits with leakage checks; keyword, TF-IDF, Prompt Guard, ProtectAI baselines |
| 9 | ✅ Done | Fine-tuned DeBERTa-v3-small: 98.7% recall, 0.997 AUROC (chunked) |
| 10 | ✅ Code + local results · ⏳ DeBERTa scoring (Kaggle) | Sensitive-topic + disguised test sets, normalization, rules, threshold sweep, midpoint check-in |
| 11 | ✅ Code + offline results · ⏳ real agent runs (Kaggle) | Detector inside AgentDojo; default and held-out attacks; banking reported separately |
| 12 | | Compare everything; stretch goals (buffer week) |
| 13 | | Draft paper |
| 14 | | Final paper, code cleanup, slides |
