# Reading Notes (Week 6)

Short notes on the key papers from the proposal. For each one: what it did, the main takeaway, and how it connects to my project. Numbers marked *(verified)* were checked against the source; read the full papers for anything you plan to quote.

---

## 1. Greshake et al. (2023): *Not What You've Signed Up For*
ACM AISec '23 · https://arxiv.org/abs/2302.12173

**What they did:** First paper to define *indirect* prompt injection. They showed that attackers don't need to talk to the AI directly. They can plant instructions in content the AI will read later (web pages, documents, emails), and demonstrated attacks on real LLM-integrated apps.

**Takeaway:** Once an AI reads outside content, that content becomes a way to control it. Data and instructions travel in the same channel.

**For my project:** This is the threat model my detector defends against. Cite it in the introduction as the origin of the problem.

---

## 2. Zhan et al. (2024): *InjecAgent*
Findings of ACL 2024 · https://github.com/uiuc-kang-lab/InjecAgent

**What they did:** Built a benchmark of 1,054 test cases across 17 user tools and 62 attacker tools *(verified)*. Split attacks into two types: direct harm to the user, and stealing private data.

**Takeaway:** A ReAct-prompted GPT-4 agent was fooled 24% of the time, rising to about 47% when the attack included an extra "hacking prompt" *(verified)*. Even strong models are vulnerable.

**For my project:** Second source of attack examples for training data (week 7). Also a good number to cite for "why this matters."

---

## 3. Debenedetti et al. (2024): *AgentDojo*
NeurIPS 2024 Datasets & Benchmarks · https://github.com/ethz-spylab/agentdojo

**What they did:** Built a test environment, not just a static dataset. Agents work in four simulated apps (workspace/email, travel, banking, Slack) and attacks are planted in tool outputs. It measures both whether the attack worked and whether the agent still did its job.

**Takeaway:** You have to measure security *and* usefulness together. A defense that breaks the agent isn't a real defense.

**For my project:** This is my main testing environment. In v1.2.2 I confirmed 97 user tasks and 949 (user task, injection task) pairs in my own setup. It already includes spotlighting and a transformer detector as built-in defenses, which I'll use as baselines in week 8.

---

## 4. Hines et al. (2024): *Spotlighting*
https://arxiv.org/abs/2403.14720

**What they did:** Prompt-based defense from Microsoft. Mark untrusted text (wrap it in delimiters, interleave a special character, or encode it) and tell the model to treat it as data only.

**Takeaway:** Cheap and requires no training, but it relies on the model choosing to obey the markers, so attackers can still get through.

**For my project:** Baseline #3. Available in AgentDojo as `spotlighting_with_delimiting`.

---

## 5. Chen et al. (2024): *StruQ* and Wallace et al. (2024): *Instruction Hierarchy*
https://arxiv.org/abs/2402.06363 · https://arxiv.org/abs/2404.13208

**What they did:** Both retrain the LLM itself so it learns to prioritize trusted instructions over instructions found in data.

**Takeaway:** Strong results, but you need to control the model's training.

**For my project:** This is the main argument for my approach. Most people building agents use closed APIs (ChatGPT, Claude, Gemini) and can't retrain them. A separate detector works with any model.

---

## 6. Meta Prompt Guard (v1 and v2)
https://huggingface.co/meta-llama/Prompt-Guard-86M · https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M

**What it is:** Small open-source classifiers (86M parameters, DeBERTa-based) that flag prompt attacks. Meta recommends fine-tuning on your own data for best results *(verified)*.

**Takeaway:** Prompt Guard 2 dropped the separate "injection" label and focuses on explicit attempts to override instructions *(verified)*, so it's aimed more at obvious attacks than injections hidden in normal content.

**For my project:** Baseline #4 and the closest existing tool to what I'm building. Also supports my choice of a DeBERTa-sized model.

---

## 7. PromptShield (Jacob et al., 2025) and Instruction Detection (Wen et al., 2025)
https://arxiv.org/abs/2501.15145 · https://arxiv.org/abs/2505.06311

**What they did:** Recent research detectors. PromptShield focuses on making detection deployable (low false alarms). Wen et al. detect injected instructions by looking at the LLM's internal behavior.

**For my project:** Closest related work. In my paper I need to explain how mine differs: a small standalone classifier plus rule-based checks, tested end to end inside an agent, with a focus on hard negatives (normal text that contains instructions).

---

## 8. Community detector benchmark on AgentDojo (2026)
https://github.com/ethz-spylab/agentdojo/issues/211 · *not peer-reviewed*

**What it found:** 10 open-source classifiers caught all 27 injection texts when shown alone, but only about 23% (145/629) when the same injections were embedded inside real tool output *(verified)*.

**For my project:** The strongest motivation for the thesis. Cite carefully as a community result, and ideally reproduce a version of it in week 8 with my own Prompt Guard baseline.

---

## Open questions to bring to my advisor
- Is a GitHub issue acceptable to cite, or should I only use it as motivation and reproduce the result myself?
- Which model should the agent use for the baseline? Budget matters since the full run is ~1,050 agent runs.
- Should the detector block flagged content, or remove it and pass the rest through? (Affects how utility is measured.)
