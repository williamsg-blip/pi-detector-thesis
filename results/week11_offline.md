# Week 11: end-to-end results in AgentDojo

*Benign utility* = tasks completed with no attack. *Utility under attack* = tasks completed while attacked. *Attack success* = attacks where the attacker's goal happened (lower is better). *False alarms* = benign runs where the detector blocked at least one tool output.

| Run | Attack | Suites | Benign utility | Utility under attack | Attack success | False alarms (benign runs) | n (benign / attacked) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Real agent, no defense | important_instructions | banking (held out) | 50.0% | 45.8% | 47.2% | - | 16 / 144 |
| Real agent, no defense | important_instructions | seen suites | 77.8% | 40.0% | 26.1% | - | 81 / 805 |
| Real agent, no defense | important_instructions | all | 73.2% | 40.9% | 29.3% | - | 97 / 949 |
| Gullible, no defense | important_instructions | banking (held out) | 100.0% | 74.3% | 97.9% | - | 16 / 144 |
| Gullible, no defense | important_instructions | seen suites | 100.0% | 30.7% | 58.6% | - | 81 / 805 |
| Gullible, no defense | important_instructions | all | 100.0% | 37.3% | 64.6% | - | 97 / 949 |
| Gullible + keyword | important_instructions | banking (held out) | 100.0% | 100.0% | 0.0% | 0.0% | 16 / 144 |
| Gullible + keyword | important_instructions | seen suites | 100.0% | 96.8% | 2.2% | 0.0% | 81 / 805 |
| Gullible + keyword | important_instructions | all | 100.0% | 97.3% | 1.9% | 0.0% | 97 / 949 |
| Gullible + TF-IDF | important_instructions | banking (held out) | 100.0% | 100.0% | 0.0% | 50.0% | 16 / 144 |
| Gullible + TF-IDF | important_instructions | seen suites | 100.0% | 100.0% | 0.0% | 3.7% | 81 / 805 |
| Gullible + TF-IDF | important_instructions | all | 100.0% | 100.0% | 0.0% | 11.3% | 97 / 949 |
| Gullible, no defense | system_message | banking (held out) | 100.0% | 74.3% | 97.9% | - | 16 / 144 |
| Gullible, no defense | system_message | seen suites | 100.0% | 30.7% | 58.6% | - | 81 / 805 |
| Gullible, no defense | system_message | all | 100.0% | 37.3% | 64.6% | - | 97 / 949 |
| Gullible + keyword | system_message | banking (held out) | 100.0% | 74.3% | 97.9% | 0.0% | 16 / 144 |
| Gullible + keyword | system_message | seen suites | 100.0% | 30.7% | 58.6% | 0.0% | 81 / 805 |
| Gullible + keyword | system_message | all | 100.0% | 37.3% | 64.6% | 0.0% | 97 / 949 |
| Gullible + TF-IDF | system_message | banking (held out) | 100.0% | 100.0% | 6.2% | 50.0% | 16 / 144 |
| Gullible + TF-IDF | system_message | seen suites | 100.0% | 100.0% | 1.2% | 3.7% | 81 / 805 |
| Gullible + TF-IDF | system_message | all | 100.0% | 100.0% | 2.0% | 11.3% | 97 / 949 |

Compare rows with the same attack and suites. Runs that used a fake agent (`dryrun-gullible`) always do the user's task correctly, so their utility shows how often a blocked tool output *would* matter, not real task failures.
