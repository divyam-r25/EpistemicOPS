# EpistemicOps: 3-Minute Pitch

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
*(Visual: Reward curve before and after training, highlighting the 28% Socratic Delta)*

**Speaker Notes (2:00 - 3:00):**
Here is what our base model does before training: it completes 45% of era tasks, it detects exactly 8% of drift events, and its Legacy Documents improve the next generation's performance by less than 5%.

After training: 72% task completion, 55% drift detection, and Legacy Documents that improve the next generation by 43% on average.

But the number we are most proud of is this one: when the Oversight Agent intervenes, the Primary Agent improves its recovery performance by 28% per session — and the Oversight Agent does it without ever giving the answer.

EpistemicOps is not just an environment. It is a proof of concept that memory compression, temporal trust, and pedagogical transfer can be trained together — because they are the same skill. Thank you.
