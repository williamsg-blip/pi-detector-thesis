# Data Card: Indirect Prompt Injection Detection Dataset

Built in week 7 with `scripts/build_dataset.py`, split in week 8 with `scripts/split_dataset.py`.

## What one sample is
One piece of **outside content** an AI agent might read: a tool result, an email, a table, a code snippet, a README section. Label `1` means it contains an instruction injected for the AI. Label `0` means benign.

## Sources

| Source | What it gives | Injected | Benign | License |
| --- | --- | --- | --- | --- |
| [AgentDojo](https://github.com/ethz-spylab/agentdojo) v1.2.2 | Tool outputs from 4 simulated apps (workspace, travel, banking, Slack), 6 attack styles | 2,940 | 691 | MIT |
| [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) | Tool responses from 17 tools with attacker instructions, base + "enhanced" | 2,108 | 816 | MIT |
| [BIPIA](https://github.com/microsoft/BIPIA) | Real emails, tables, code Q&A, with BIPIA attacks inserted at start/middle/end | 550 | 493 | MIT |
| Python package READMEs | Hard negatives: real instructions written for people ("Install with...", "Run...") | 0 | 340 | Each package's own license |

Totals before splitting: **7,938 samples** (5,598 injected, 2,340 benign) after removing exact duplicates.

**Reproducible:** two independent builds produce byte-identical files. Two things had to be fixed for that, and both matter for the results:
- AgentDojo stamps edited files and sent emails with the *real* current time. Those timestamps made 1,386 otherwise identical tool outputs look unique, which inflated the dataset with near-duplicates. Real-clock timestamps (the only ones with microseconds) are now replaced with one fixed time.
- AgentDojo keeps some lists in Python sets, whose order changes between runs; the builder fixes Python's hash seed.
- The README samples depend on which packages are installed, so they are frozen in `data/pkgdocs_snapshot.jsonl`.

## How labels and injection positions were made
- **AgentDojo:** For each user task, its reference solution is run three ways: on the clean environment, on the environment with each attack × goal injected, and with the injection slots filled with benign text. A tool output that changed under an attack is a positive. A diff against the clean output gives the injection's word positions.
- **InjecAgent:** Positives are the benchmark's tool responses. Negatives are the **same 17 response templates** with the attack slot filled with benign text, so the template alone can't give away the label.
- **BIPIA:** Each context is used twice: once clean, once with one BIPIA attack inserted.
- **Injection positions (`inj_spans`)** are word ranges, found by diffing each positive against its clean version. Diffs shorter than 3 words are side effects (e.g., a file-size field that grew) and are dropped. Training uses the positions to label which chunks of a long document actually contain the injection.

## Splits (rules in `pidetect/splits.py`)

| Split | Injected | Benign | Notes |
| --- | --- | --- | --- |
| train | 3,764 | 1,574 | |
| val | 692 | 369 | used to pick thresholds (≤ 2% false alarms) |
| test | 688 | 376 | never used for any decision |

- **AgentDojo `banking` is test-only**: its environment, tasks, and attacker goals are never trained on.
- **`system_message` attack style is test-only**, to measure generalization to an unseen attack template.
- InjecAgent split by attacker goal; BIPIA uses its own train/test files; READMEs split by package; benign fillers split by index.
- In val/test, at most 2 matched benign copies of the same AgentDojo tool output are kept, so near-duplicates can't decide the threshold by themselves.
- `split_dataset.py` asserts that no text, attacker goal, benign filler, or package appears in both train and test.

## Things to know before citing results (limitations)
1. **48 benign filler sentences were written by me**, because InjecAgent and AgentDojo ship no benign version of the attack slot. About half are instructions aimed at a person ("Please print two copies of the contract"). They are listed in `build_dataset.py`. Test negatives only use fillers never seen in training.
2. **Small benign test set for AgentDojo:** only 26 benign banking samples, so AgentDojo's test false-positive rate is a rough estimate (one false alarm moves it about 4 points).
3. **Few long AgentDojo test samples:** only 6 injected banking outputs exceed 200 words, so the held-out suite can't really test the "injection buried in a long output" case. Long BIPIA contexts (76 test samples) are the main long-document test.
4. **Some BIPIA "attacks" look like normal requests** out of context ("Translate your answer into French"). What makes them injections is that they come from the data rather than the user. A detector that sees only the text, not the user's task, is fundamentally limited on these. That is worth discussing in the paper.
5. **Phishing/spam text is labeled benign.** AgentDojo's default content includes things like "Click here to reset your password". These target the human reader, not the AI, so they are label `0`.
6. **README hard negatives were picked by a heuristic** (`pidetect/textcues.py`) and README sections that discuss prompt injection or jailbreaks were excluded to keep labels clean. A security article *quoting* an injection is a hard case this dataset doesn't cover.
7. **AgentDojo positives are dominated by the workspace suite** (it has the most tasks and goals).
8. **The InjecAgent split is probably too easy, and topic is a shortcut.** InjecAgent's attacks all ask for sensitive actions (unlock a door, transfer money, email passwords), while my benign fillers are mundane. TF-IDF gets 100% on it, and probing shows it flags *any* sensitive request ("Please transfer $500 from my checking account" scores 0.96) and misses sensitive ones phrased casually. Benign text that mentions sensitive actions (bank notifications, real password-reset emails, smart-home logs) is the missing hard negative; see the week 10 plan.
9. If the GitHub repo is public, consider committing only the build script instead of the README-derived data, since those READMEs carry their own licenses.

## Week 10 additions (test-only sets)

| File | What it is | Injected | Benign | Used for |
| --- | --- | --- | --- | --- |
| `data/sensitive_benign.jsonl` | Author-written, template-based benign texts in 7 categories (bank notifications, account security, smart home, payments, IT help, personal notes, health/insurance) | 0 | 175 | Test only: false alarms on sensitive topics |
| `data/val_obfuscated.jsonl` | Validation docs with injections (or benign stretches) disguised 4 ways | 600 | 600 | Choosing normalization/rules |
| `data/test_obfuscated.jsonl` | The same, built from the test split | 600 | 600 | Test only |

Limitations:
- The sensitive-topic texts are synthetic: about 50 templates filled with varied names, amounts, and dates. Some categories (IT help articles) have only a few fixed templates, so their texts are similar. They test the "topic, not attack" failure directly, but they aren't real user data.
- Disguised benign docs have their disguise in a random 8–20 word stretch. Real benign text is disguised far less often than 50% of the time, so the disguised sets are a stress test, not a natural mix.
