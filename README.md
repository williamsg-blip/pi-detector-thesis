# Detecting Indirect Prompt Injection in LLM Agents

Senior thesis project. A small, fast classifier that checks outside content (tool results, web pages, emails) before an AI agent reads it, and flags hidden instructions. Tested end to end in [AgentDojo](https://github.com/ethz-spylab/agentdojo).

**Research question:** Can a small classifier catch indirect prompt injections well enough to stop most attacks, without blocking so much normal content that the agent stops being useful?

## Project structure

```
pi-detector-thesis/
├── scripts/
│   ├── check_environment.py   # Week 5: confirms everything is installed and working
│   └── run_baseline.py        # Week 6: no-defense baseline (reused in weeks 8 and 11)
├── results/                   # CSV results (one row per agent run)
├── notes/
│   ├── reading_notes.md       # Week 6: notes on key papers
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

## Timeline
| Week | Status | Focus |
| --- | --- | --- |
| 5 | ✅ Done | Environment, repo, environment check |
| 6 | ✅ Code + notes done · ⏳ real API run | No-defense baseline, reading |
| 7 | | Collect attack + benign data |
| 8 | | Clean data; Prompt Guard / keyword / spotlighting baselines |
| 9 | | Fine-tune classifier |
| 10 | | Rule-based checks, threshold tuning, midpoint check-in |
| 11 | | Plug detector into AgentDojo, end-to-end tests |
| 12 | | Compare baselines; stretch goals (buffer week) |
| 13 | | Draft paper |
| 14 | | Final paper, code cleanup, slides |
