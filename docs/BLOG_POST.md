# EpistemicOps: Training Agents to Work Through Silent Change

Production AI agents do not usually fail because they cannot answer a question. They fail because the world changes quietly around them.

An API schema changes. A dashboard field is renamed. A rollback response stops returning a body. The agent still trusts the old behavior and makes the wrong decision.

**EpistemicOps** is an OpenEnv-style RL environment built to train agents for exactly that kind of world.

It focuses on three things that matter in real systems:

1. **Silent drift** — the environment changes without warning.  
2. **Long-horizon memory** — the agent must preserve useful knowledge across eras.  
3. **Socratic oversight** — a teacher can guide the agent, but cannot hand over the answer.

---

## What the environment looks like

EpistemicOps runs as a multi-era incident-response world.

In each era, the **Primary Agent** uses mock enterprise APIs to resolve an incident or complete a task. The available tools include incident, metrics, deploy, log, and notification services.

Midway through an era, the environment can silently change one of those APIs.

For example:
- a numeric field may become a string,
- a response body may disappear,
- pagination may switch from offset-based to cursor-based,
- authentication rules may change,
- or rate limits may tighten suddenly.

The agent is not told that this happened. It has to notice the mismatch from behavior.

If the agent gets stuck, an **Oversight Agent** steps in. Its job is not to solve the task directly. Instead, it asks targeted questions, gives counter-examples, or reframes the problem so the Primary Agent can reason its way forward.

At the end of every era, the Primary Agent writes a short **Legacy Document**. This is a compressed handoff to the next era. Then its context is wiped. Only the legacy survives.

That is the main learning challenge:  
**solve today’s problem, but also leave something useful for tomorrow.**

---

## Why this matters

Most benchmarks test whether a model can answer a question once.

EpistemicOps tests something closer to production reality:
- Can the model adapt when the world changes?
- Can it detect that an assumption is no longer valid?
- Can it recover from unexpected failures?
- Can it preserve useful knowledge across long sessions?

This makes the environment useful for training agents that operate in real systems, not static puzzles.

---

## How training works

The project uses a reward-driven training setup.

The reward signal is composed of:
- whether the agent completed the task,
- whether its confidence was calibrated,
- whether it improved after oversight,
- whether the legacy document helped the next era,
- whether the teacher leaked the answer,
- and whether the behavior looked degenerate or repetitive.

This ensures the model is not rewarded just for producing plausible text, but for **acting correctly in a changing environment**.

---

## What we trained

We trained the agent using a TRL / GRPO-style pipeline with an Unsloth-based setup for efficient fine-tuning.

The goal was to improve:
- drift detection,
- recovery from failures,
- and cross-era knowledge transfer.

---

## Results

We evaluated baseline and trained agents on the same scenarios.

- **Drift detection:** baseline **0%** → trained **33.3%**
- **Average normalized reward:** baseline **0.296** → trained **0.345**
- **Criteria completion:** baseline **68.5%** → trained **72.2%**
- **Legacy document rate:** **100%** for both (utility still differs)
- **When drift occurred:** trained runs achieved **100% precision and recall** in detection (current evaluation window)

These results show that the agent is not just solving tasks, but **learning to adapt to change**.

---

## Experiment tracking

Training runs are tracked using **Weights & Biases**, logging:
- reward over time,
- loss curves,
- learning rate schedules.

This provides clear evidence of training and makes the results reproducible.

---

## What makes EpistemicOps different

EpistemicOps is not a static benchmark.

It is an evolving environment where:
- the world changes,
- memory matters,
- teaching is constrained,
- and improvement must be demonstrated over time.

This aligns directly with:
- **Multi-agent interaction** (Primary + Oversight)
- **Long-horizon planning** (Legacy across eras)
- **World modeling** (Silent drift)
- **Self-improvement** (Reward-driven learning)

---

## Try it yourself

- 🚀 **Live Demo:** https://huggingface.co/spaces/Divyam-r25/EpistemicOps  
- 📓 **Training Notebook:** https://colab.research.google.com/github/divyam-r25/EpistemicOPS/blob/main/training/colab_grpo_training.ipynb  
- 💻 **Source Code:** https://github.com/divyam-r25/EpistemicOPS
