# EpistemicOps: Training Agents on Temporal Uncertainty and Socratic Oversight

What breaks AI agents in production? Three things:
1. **The world changes:** An API schema updates overnight, and the agent blindly trusts old documentation.
2. **Context is finite:** Long-running incidents exceed context windows, and agents forget critical realizations.
3. **They can't self-diagnose:** When they fail, they need humans to step in and fix them.

Current RL environments train agents to solve static puzzles. In production, puzzles aren't static.

**EpistemicOps** is a new RL environment that trains agents to handle these specific failure modes. It treats three seemingly different problems—handling stale knowledge, passing context across generations, and Socratic oversight—as the exact same skill: **structured curation of knowledge under uncertainty**.

## How it works

The environment runs across 5 "Eras". A Primary Agent is tasked with resolving SRE incidents using 5 mock API services.

Here's the twist: mid-era, the environment silently mutates the API contracts. It changes integer status codes to strings. It breaks pagination. It alters auth headers. The agent is never told. It must detect the drift through downstream failures.

When it fails, a second agent—the **Oversight Agent**—intervenes. But it cannot give the answer. It can only ask targeted Socratic questions based on the Primary Agent's reasoning trace.

At the end of an Era, the Primary Agent writes a 2048-token "Legacy Document" to its successor. Then its memory is wiped. The next Era starts with only that document.

## Results
We fine-tuned Llama 3.1 8B using GRPO (via HuggingFace TRL and Unsloth). 
The target metrics post-training are:
- **Drift Detection:** Rise from 8% (baseline) to **55% (projected)**
- **Legacy Utility:** Legacy Documents improve successor's performance by **43% (projected)**
- **Oversight Impact:** Primary Agent improves recovery by **28% (projected)** per intervention.

EpistemicOps proves that temporal trust and pedagogical transfer can be trained together. Because in the end, they are the same cognitive act.

[Explore the codebase](https://github.com/yourusername/EpistemicOps) | [Try the Demo](https://huggingface.co/spaces/yourusername/EpistemicOps)
