# EpistemicOps: Training Agents on Temporal Uncertainty and Socratic Oversight

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

Our baseline (zero-shot mock agent, no fine-tuning) shows:
- **Drift Detection:** Mock agents detect ~50% of injected drifts via structured hypothesis-testing
- **Legacy Utility:** Legacy Documents provide measurable performance lift to successor agents (+0.35 reward)
- **Oversight Impact:** Socratic interventions improve task recovery by 0.5 teacher_delta when triggered
- **Average Normalized Reward:** ~0.35 across scenarios (out of max 1.0)

The GRPO training pipeline (Llama 3.1 8B via Unsloth, 4-bit quantized) trains against this reward signal to improve drift detection and legacy document quality.

## Try it yourself

- **Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/Divyam-r25/EpistemicOps)
- **Training Notebook:** [Open in Colab](https://colab.research.google.com/github/divyam-r25/EpistemicOPS/blob/main/training/colab_grpo_training.ipynb)
- **Source Code:** [GitHub](https://github.com/divyam-r25/EpistemicOPS)
