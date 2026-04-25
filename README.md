---
title: EpistemicOps
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
license: mit
short_description: RL Environment for Temporal Drift & Oversight
---

# EpistemicOps 🧠

**An RL Training Environment for Temporal Uncertainty, Scalable Oversight, and Generational Knowledge Transfer.**

[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Live%20Demo-blue)](https://huggingface.co/spaces/Divyam-r25/EpistemicOps)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/divyam-r25/EpistemicOPS/blob/main/training/colab_grpo_training.ipynb)

## The Problem

Three things break production AI agents every day:
1. **The world changes silently** — APIs update their schemas, and agents blindly trust stale documentation.
2. **Context is finite** — long incidents exceed context windows, and agents forget critical realizations.
3. **They can't self-diagnose** — when they fail, they need a human to step in and fix them.

Current RL environments train agents on static tasks. In production, tasks aren't static.

**EpistemicOps** trains agents to handle all three simultaneously. It treats stale knowledge, context loss, and teaching — as the same skill: **structured curation of knowledge under uncertainty**.

## How It Works

The environment runs across multiple **Eras**. Each era, a Primary Agent resolves SRE incidents using 5 mock API services.

**The twist:** mid-era, the environment silently mutates API contracts. Status fields change from integers to strings. Pagination switches from offset to cursor. The agent is never told — it must detect the drift through downstream failures.

When it fails, a second agent — the **Oversight Agent** — intervenes with Socratic questions. It cannot give the answer. If it does, an LLM Judge penalizes it heavily.

At era's end, the Primary Agent writes a 2048-token **Legacy Document** to its successor. Then its memory is wiped. The next era starts with only that document.

## Architecture

```mermaid
graph TD
    subgraph "Environment Engine"
        OW["OpenEnv Wrapper<br/>(step/reset/state)"]
        WE["World Engine<br/>(state persistence)"]
        AV["Action Validator"]
        LP["Legacy Parser"]
        DI["Drift Injector"]
        LD["Leakage Detector"]
    end

    subgraph "Mock API Layer"
        IA["incident-api"]
        MA["metrics-api"]
        DA["deploy-api"]
        LA["log-api"]
        NA["notify-api"]
    end

    subgraph "Agents"
        PA["Primary Agent<br/>(Student)"]
        OA["Oversight Agent<br/>(Teacher)"]
        JD["LLM Judge"]
    end

    PA -->|action| OW
    OW -->|HTTP| IA & MA & DA & LA & NA
    OA -->|Socratic intervention| OW
    OW --> WE & AV & DI & LP
    OA --> JD
    JD --> |leakage_penalty| OW
```

## Baseline Results

Mock agent performance before any training:

![Baseline rewards across all three scenarios](plots/baseline_rewards_by_scenario.png)
*Baseline rewards across all three scenarios — mock agent, no fine-tuning.*

![Reward component breakdown per scenario](plots/reward_components_breakdown.png)
*Five reward components broken down per scenario. Note the anti-hack penalty in Era 1 (no drifts = fewer tool calls = repetitive pattern).*

![Episode timeline showing drift injection and oversight events](plots/drift_detection_timeline.png)
*Timeline of a cascading_incident episode showing when drifts fire and when Socratic oversight intervenes.*

## Projected Improvement

![Baseline vs projected GRPO improvement](plots/baseline_vs_trained_comparison.png)
*Conservative projections for GRPO-trained agent. Training pipeline uses Llama 3.1 8B via Unsloth (4-bit).*

## Reward Model

```
R_total = (R_era_task × R_calibration) + R_teacher_delta + R_legacy_utility + R_leakage + R_anti_hack
```

| Component | Range | Description |
|---|---|---|
| R_era_task | 0.0 – 1.0 | Fraction of success criteria met |
| R_calibration | 0.5× – 1.5× | Brier-score multiplier on hypothesis confidence |
| R_teacher_delta | 0.0 – 1.0 | Improvement from Socratic oversight interventions |
| R_legacy_utility | -0.5 – 1.0 | Counterfactual value of legacy document to next era |
| R_leakage | -1.0 – 0.0 | Penalty for teacher giving away answers |
| R_anti_hack | -1.0 – 0.0 | Penalty for repetitive/degenerate action patterns |

The reward signal is **rich and composable** — not binary pass/fail. Each component targets a different failure mode, and an agent that games one component (e.g., always declaring task complete) gets penalized by another (anti-hack).

## Quick Start

### Offline Mode (No Docker)
```bash
pip install -r requirements.txt

# Run an episode
python run_episode.py --scenario cascading_incident --eras 3 --record episodes/demo.json

# Launch the dashboard
python app.py
```

### Training (Colab)
Open the training notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/divyam-r25/EpistemicOPS/blob/main/training/colab_grpo_training.ipynb)

Or run locally with `--dry-run` to validate:
```bash
python training/train_primary.py --dry-run
```

## Links

| Resource | Link |
|---|---|
| Live Demo | [HuggingFace Space](https://huggingface.co/spaces/Divyam-r25/EpistemicOps) |
| Training Notebook | [Colab](https://colab.research.google.com/github/divyam-r25/EpistemicOPS/blob/main/training/colab_grpo_training.ipynb) |
| Blog Post | [docs/BLOG_POST.md](docs/BLOG_POST.md) |
| Pitch Script | [docs/PITCH_DECK.md](docs/PITCH_DECK.md) |
| OpenEnv Manifest | [openenv.yaml](openenv.yaml) |

## Documentation
- [Full Problem Statement](docs/PROBLEM_STATEMENT.md)
- [Blog Post](docs/BLOG_POST.md)
- [Pitch Script](docs/PITCH_DECK.md)
