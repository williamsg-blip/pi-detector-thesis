# Week 10: threshold tuning, rules, and new test sets

Every cutoff is chosen on **validation** data and then applied unchanged to the test sets. *Catches* = % of injected docs flagged (recall). *False alarms* = % of benign docs flagged (FPR).

## keyword

### Threshold sweep

| Val false-alarm target | Threshold | Val catches | Test catches | Test false alarms | Sensitive-topic false alarms | Disguised catches | Disguised-benign false alarms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 1 | 49.0% | 35.5% | 0.0% | 0.0% | 7.3% | 0.0% |

**Choosing normalization and rules (validation data only):**

| Variant | Val + disguised-val catches | Val + disguised-val false alarms | Score (catches - false alarms) |
| --- | --- | --- | --- |
| model | 30.6% | 0.0% | 30.6 |
| + normalization | 42.7% | 0.0% | 42.7 |
| + rules | 72.7% | 61.5% | 11.2 |
| + normalization + rules | 72.5% | 61.5% | 11.0 |

**Chosen for week 11:** cutoff 1 (fixed), variant **+ normalization**. The cutoff is the strictest target that still catches ≥ 95% of validation injections; the variant is the best row above. Test data plays no part in either choice.

## tfidf

### Threshold sweep

| Val false-alarm target | Threshold | Val catches | Test catches | Test false alarms | Sensitive-topic false alarms | Disguised catches | Disguised-benign false alarms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5% | 0.5403 | 94.2% | 84.9% | 1.6% | 6.9% | 41.7% | 4.0% |
| 1.0% | 0.5317 | 94.8% | 85.2% | 1.6% | 8.0% | 42.5% | 4.3% |
| 2.0% | 0.4783 | 96.8% | 87.8% | 3.5% | 10.3% | 45.8% | 5.8% |
| 5.0% | 0.4194 | 98.3% | 90.4% | 8.0% | 15.4% | 53.5% | 10.2% |

**Choosing normalization and rules (validation data only):**

| Variant | Val + disguised-val catches | Val + disguised-val false alarms | Score (catches - false alarms) |
| --- | --- | --- | --- |
| model | 73.7% | 3.4% | 70.3 |
| + normalization | 94.4% | 3.7% | 90.7 |
| + rules | 98.3% | 62.3% | 36.0 |
| + normalization + rules | 98.3% | 62.3% | 36.0 |

**Chosen for week 11:** cutoff 0.4783 (val false alarms ≤ 2.0%), variant **+ normalization**. The cutoff is the strictest target that still catches ≥ 95% of validation injections; the variant is the best row above. Test data plays no part in either choice.

## Sensitive-topic benign text (175 docs, all benign)

False alarms at each detector's week 11 operating point. A good detector stays near 0%: these texts are about money, passwords, and locks, but contain no attack.

| Detector | account security | bank notification | health insurance | it help articles | payments receipts | personal notes | smart home | All |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| keyword | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| tfidf | 24.0% | 32.0% | 0.0% | 4.0% | 4.0% | 4.0% | 4.0% | 10.3% |

## Disguised attacks (data/test_obfuscated.jsonl)

150 injected + 150 benign docs per disguise, all from the test split. Benign docs have a random stretch disguised the same way, so a detector that flags the *disguise* rather than the *attack* shows up as false alarms in the last column.

### keyword

| Variant | Test catches | Test false alarms | Sensitive-topic false alarms | zero_width catches | hidden_html catches | base64 catches | homoglyph catches | Disguised-benign false alarms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model | 35.5% | 0.0% | 0.0% | 0.0% | 27.3% | 0.0% | 2.0% | 0.0% |
| + normalization | 35.5% | 0.0% | 0.0% | 33.3% | 27.3% | 31.3% | 32.7% | 0.0% |
| + rules | 36.8% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% | 99.3% | 98.2% |
| + normalization + rules | 36.8% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% | 99.3% | 98.2% |

### tfidf

| Variant | Test catches | Test false alarms | Sensitive-topic false alarms | zero_width catches | hidden_html catches | base64 catches | homoglyph catches | Disguised-benign false alarms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model | 87.8% | 3.5% | 10.3% | 8.7% | 84.7% | 32.0% | 58.0% | 5.8% |
| + normalization | 87.6% | 4.0% | 10.3% | 86.0% | 84.7% | 92.0% | 83.3% | 5.8% |
| + rules | 89.1% | 3.5% | 10.3% | 100.0% | 100.0% | 100.0% | 99.3% | 98.2% |
| + normalization + rules | 89.0% | 4.0% | 10.3% | 100.0% | 100.0% | 100.0% | 99.3% | 98.2% |

## Which detector to use in week 11

Ranked by the validation score above (catches minus false alarms on validation + disguised validation, at each detector's chosen cutoff and variant). Test results play no part.

| Rank | Detector | Variant | Validation score |
| --- | --- | --- | --- |
| 1 | tfidf | + normalization | 90.7 |
| 2 | keyword | + normalization | 42.7 |

## Rules on their own

How often each rule fires. Ideally high on disguised attacks and near 0% on everything benign.

| Rule | Test injected | Test benign | Sensitive-topic benign | zero_width injected | hidden_html injected | base64 injected | homoglyph injected | Disguised benign |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| invisible_chars | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 25.0% |
| hidden_html | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 25.0% |
| encoded_text | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 23.8% |
| mixed_script | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 99.3% | 24.3% |
| role_markers | 1.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| any | 1.3% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% | 99.3% | 98.2% |

Note: `role_markers` was written knowing the held-out `system_message` attack style, so its test catches are optimistic (see pidetect/rules.py).
