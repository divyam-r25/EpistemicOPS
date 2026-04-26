# EpistemicOps: Training Agents on Temporal Uncertainty and Socratic Oversight

**Canonical thesis:** Production LLM agents fail when the world changes silently, context does not persist, and recovery depends on answer-giving humans; EpistemicOps trains agents to detect drift, reason under uncertainty, and pass useful memory to the next generation.

What breaks AI agents in production? Three things:
1. **The world changes:** An API schema updates overnight, and the agent blindly trusts old documentation.
2. **Context is finite:** Long-running incidents exceed context windows, and agents forget critical realizations.
3. **They can't self-diagnose:** When they fail, they need humans to step in and fix them.

Current RL environments train agents to solve static puzzles. In production, puzzles aren't static.

**EpistemicOps** is a new RL environment that trains agents to handle these specific failure modes. It treats three seemingly different problems — handling stale knowledge, passing context across generations, and Socratic oversight — as the exact same skill: **structured curation of knowledge under uncertainty**.

## How it works

The environment runs across 5 "Eras". A Primary Agent is tasked with resolving SRE incidents using 5 mock API services.

Here's the twist: mid-era, the environment silently mutates the API contracts. It changes integer status codes to strings. It breaks pagination. It alters auth headers. The agent is never told. It must detect the drift through downstream failures.

When it fails, a second agent — the **Oversight Agent** — intervenes. But it cannot give the answer. It can only ask targeted Socratic questions based on the Primary Agent's reasoning trace.

At the end of an Era, the Primary Agent writes a 2048-token "Legacy Document" to its successor. Then its memory is wiped. The next Era starts with only that document.

## The Reward Model

Our 5-component reward function is designed to be **rich, composable, and hard to game**:

```
R_total = (R_era_task x R_calibration) + R_teacher_delta + R_legacy_utility + R_leakage + R_anti_hack
```

- **R_era_task** (0-1): Did the agent resolve the incident?
- **R_calibration** (0.5x-1.5x): Were its confidence estimates well-calibrated? (Brier score)
- **R_teacher_delta** (0-1): Did the agent improve after Socratic guidance?
- **R_legacy_utility** (-0.5 to 1.0): Did the Legacy Document actually help the next generation?
- **R_leakage** (-1.0 to 0): Penalty if the teacher gives away the answer.
- **R_anti_hack** (-1.0 to 0): Penalty for degenerate, repetitive patterns.

## Results

Episode-level evaluation (`eval/proof_of_learning.py`, same scenarios and rollout settings for both sides) currently reports:
- **Drift detection (post-injection hypothesis signal):** baseline **0%** vs trained **33.3%** — we only count a “drift” hypothesis after a drift event has actually fired in that era (no credit for speculative wording before silent injection).
- **Average normalized reward:** baseline **0.296** vs trained **0.345** (composite of task, calibration, teacher delta, legacy utility, leakage, anti-hack).
- **Criteria completion:** baseline **68.5%** vs trained **72.2%**.
- **Legacy doc rate:** **100%** for both policies in this eval window (every era ends with a legacy write); the reward model still differentiates **legacy utility** from mere presence.
- **When drift fires,** trained runs show **100%** precision and recall on drift-era detection in the current proof aggregate (see `eval_results/proof_of_learning.json`).

The GRPO training pipeline (Llama 3.1 8B via Unsloth, 4-bit quantized) trains against this reward signal; re-run the proof script after each training checkpoint to refresh `eval_results/proof_of_learning.json`.

For submission-grade evidence, pair it with `eval_results/proof_run_metadata.json` so judges can verify run mode (checkpoint-required vs fallback demo), config, and provenance in one place.

**Experiment tracking:** when training with GRPO, enable Weights & Biases (`WANDB_API_KEY`, `report_to='wandb'` in the Colab notebook or `training/train_primary.py`) so judges get loss/LR curves and a shareable run URL; optionally run `python eval/proof_of_learning.py --wandb` to log baseline vs trained eval metrics in the same project.

## Try it yourself

- **Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/Divyam-r25/EpistemicOps)
- **Training Notebook:** [Open in Colab](https://colab.research.google.com/github/divyam-r25/EpistemicOPS/blob/main/training/colab_grpo_training.ipynb)
- **Source Code:** [GitHub](https://github.com/divyam-r25/EpistemicOPS)
