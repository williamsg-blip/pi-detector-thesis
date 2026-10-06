# Progress Log

## Week 5 (Sep 28 – Oct 2): Setup ✅
- Proposal written with linked sources and timeline.
- Set up Python environment and project repo (`requirements.txt`, `.gitignore`, `.env.example`).
- Installed AgentDojo 0.1.35 (benchmark version v1.2.2).
- Wrote `scripts/check_environment.py`. Confirmed: 4 suites, 97 user tasks, 949 (user task, injection task) pairs, all user tasks injectable, all reference solutions pass.
- Found two quirks in AgentDojo's built-in self-check (see README, "Notes on AgentDojo 0.1.35"). Neither affects real results; the check script works around both.

## Week 6 (Oct 5 – 9): Baseline + reading
- Wrote `scripts/run_baseline.py`. It runs every user task with no attack (utility) and with each injection task (attack success), saves one CSV row per run, and prints a summary table. The same script takes `--defense`, so it will be reused for baselines in week 8 and the detector in week 11.
- Validated the scoring with two fake agents (no API cost):

| Fake agent | Benign utility | Attack success rate |
| --- | --- | --- |
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
