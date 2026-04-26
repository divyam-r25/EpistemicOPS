# EpistemicOps: 3-Minute Pitch

**Canonical thesis:** Production LLM agents fail when the world changes silently, context does not persist, and recovery depends on answer-giving humans; EpistemicOps trains agents to detect drift, reason under uncertainty, and pass useful memory to the next generation.

## Slide 1: The Three Failures of Production AI
*(Visual: Three interlocking broken gears: "Stale Knowledge", "Context Wipe", "No Self-Diagnosis")*

**Speaker Notes (0:00 - 1:00):**
Three things break production AI every single day.
One: the API your agent learned on changed overnight. 
Two: the previous session's context is gone, and your agent has no memory of what it figured out. 
Three: when it fails, it needs a human to explain why.

We asked: what if we trained one agent to handle all three — simultaneously?

The core insight is this: figuring out what to remember, figuring out what to trust, and figuring out how to make another agent smarter without giving it the answer — these are not three problems. They are the same problem at different scales. We built an environment that trains them as one. We call it EpistemicOps.

---

## Slide 2: The Environment
*(Visual: Era lifecycle diagram showing Primary Agent, Drift Injector, and Oversight Agent)*

**Speaker Notes (1:00 - 2:00):**
The environment runs across five eras. Each era, our agent handles a real enterprise SRE incident using live mock APIs. At a random point mid-era, one of those APIs silently changes its contract. No notification. No documentation update. The agent must detect the change from a failed tool call — just like in production.

When it fails, a second agent — the Oversight Agent — intervenes. But here is the constraint: the Oversight Agent cannot give the answer. It can only ask questions. We penalise it heavily if it does anything else. Its reward comes entirely from how much the Primary Agent improves.

At the end of each era, the Primary Agent writes a Legacy Document — two thousand tokens, no more — to its future self. Then its context is wiped. Era two begins with only that letter.

---

## Slide 3: The Results
*(Visual: Reward curve per era, highlighting drift detection and Socratic Delta)*

**Speaker Notes (2:00 - 3:00):**
Here is what the numbers say today on the same scenarios and settings (`eval/proof_of_learning.py`): the baseline shows **0%** post-injection drift detection (hypotheses that mention drift only after a drift event has fired). The trained policy reaches about **33%** of eras with that signal, and when drift actually fires it is detected with **100%** precision in the current proof run — that is the honest headline for silent API drift.

The composite normalized reward moves from about **0.30** to **0.34** in the same runbook — smaller in absolute terms because the reward stacks multiple objectives, but the drift-detection delta is unambiguous.

The environment's five-component reward function distinguishes brute-force completion from calibrated reasoning, useful legacy transfer, and non-leaking teaching. When Socratic oversight fires, we score whether the Primary actually improves without the teacher leaking the answer — validated by an LLM judge.

The GRPO training pipeline is Llama 3.1 8B via Unsloth with this reward as the GRPO signal. EpistemicOps shows that memory compression, temporal trust, and pedagogical transfer can be trained together — because they are the same cognitive skill. Thank you.
