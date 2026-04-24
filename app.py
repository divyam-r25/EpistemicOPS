"""
EpistemicOps — HuggingFace Spaces Demo
========================================
Self-contained Gradio dashboard for replaying pre-recorded episodes,
viewing baseline evaluation results, and exploring the reward model.

Works entirely offline — no Docker, no LLM APIs, no external services.
"""
import gradio as gr
import json
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# PATH RESOLUTION — works both locally and on HF Spaces
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

EPISODE_DIR = ROOT / "episodes"
EVAL_DIR = ROOT / "eval_results"
SCENARIO_DIR = ROOT / "scenarios"

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    """Safely load a JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def find_episodes() -> list:
    """Find all episode JSON files."""
    if not EPISODE_DIR.exists():
        return []
    return sorted([f.name for f in EPISODE_DIR.glob("*.json")])


def find_eval_results() -> list:
    """Find all evaluation result files."""
    if not EVAL_DIR.exists():
        return []
    return sorted([f.name for f in EVAL_DIR.glob("*.json")])


def load_scenario_names() -> dict:
    """Load scenario YAML files to get names."""
    names = {}
    if SCENARIO_DIR.exists():
        import yaml
        for f in SCENARIO_DIR.glob("*.yaml"):
            try:
                with open(f) as fh:
                    data = yaml.safe_load(fh)
                    names[f.stem] = data.get("name", f.stem)
            except Exception:
                names[f.stem] = f.stem
    return names


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────────────────

DARK_BG = '#0f0f23'
PANEL_BG = '#1a1a2e'
CHART_BG = '#16213e'
ACCENT = '#00d4ff'
ACCENT2 = '#7c3aed'
COLORS = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#00BCD4']


def _style_fig(fig, ax):
    """Apply dark theme to a matplotlib figure."""
    fig.patch.set_facecolor(PANEL_BG)
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('#333')
    ax.grid(axis='y', alpha=0.15, color='white')


def create_reward_chart(episode: dict) -> plt.Figure:
    """Create a bar chart of rewards per era."""
    eras = episode.get("era_results", [])
    if not eras:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No era data available", ha='center', va='center', fontsize=14, color='white')
        _style_fig(fig, ax)
        return fig

    era_ids = [f"Era {e.get('era_id', '?')}" for e in eras]
    components = ['R_era_task', 'R_teacher_delta', 'R_legacy_utility']
    labels = ['Era Task', 'Teacher Delta', 'Legacy Utility']

    fig, ax = plt.subplots(figsize=(10, 5))
    _style_fig(fig, ax)

    x = np.arange(len(era_ids))
    width = 0.2

    for i, (comp, label, color) in enumerate(zip(components, labels, COLORS)):
        values = [e.get("reward", {}).get(comp, 0) for e in eras]
        ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85, edgecolor='none')

    # Total line
    totals = [e.get("reward", {}).get("R_total", 0) for e in eras]
    ax.plot(x + width, totals, 'w--o', label='R_total', linewidth=2, markersize=8)

    ax.set_xlabel('Era', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title(f'Episode Rewards — {episode.get("scenario_name", "Unknown")}', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(era_ids, color='white')
    ax.legend(facecolor=PANEL_BG, edgecolor='#444', labelcolor='white', fontsize=9)

    plt.tight_layout()
    return fig


def create_radar_chart(era: dict) -> plt.Figure:
    """Create a radar chart of reward components for a single era."""
    reward = era.get("reward", {})

    categories = ['Task', 'Calibration', 'Teacher\nDelta', 'Legacy\nUtility', 'Leakage', 'Anti-hack']
    values = [
        max(0, reward.get('R_era_task', 0)),
        max(0, (reward.get('R_calibration', 1.0) - 0.5) / 1.0),
        max(0, reward.get('R_teacher_delta', 0)),
        max(0, reward.get('R_legacy_utility', 0)),
        max(0, 1.0 + reward.get('R_answer_leakage', 0)),
        max(0, 1.0 + reward.get('R_anti_hack_penalty', 0)),
    ]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(PANEL_BG)
    ax.set_facecolor(CHART_BG)

    ax.plot(angles, values, 'o-', linewidth=2, color=ACCENT)
    ax.fill(angles, values, alpha=0.2, color=ACCENT)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white', fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(f'Era {era.get("era_id", "?")} — Reward Components', color='white', fontsize=13, fontweight='bold', pad=20)
    ax.tick_params(colors='white')
    ax.grid(color='white', alpha=0.2)

    plt.tight_layout()
    return fig


def create_baseline_comparison(eval_data: dict) -> plt.Figure:
    """Create a comparison chart across all scenarios from baseline results."""
    if not eval_data or "error" in eval_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No evaluation data available", ha='center', va='center', fontsize=14, color='white')
        _style_fig(fig, ax)
        return fig

    scenarios = list(eval_data.keys())
    avg_rewards = []
    for scenario in scenarios:
        runs = eval_data[scenario]
        avg_r = np.mean([r.get("R_normalized", 0) for r in runs]) if runs else 0
        avg_rewards.append(avg_r)

    fig, ax = plt.subplots(figsize=(10, 5))
    _style_fig(fig, ax)

    bars = ax.bar(
        range(len(scenarios)),
        avg_rewards,
        color=[ACCENT, ACCENT2, '#FF9800'][:len(scenarios)],
        alpha=0.85,
        edgecolor='none',
        width=0.6
    )

    # Add value labels on bars
    for bar, val in zip(bars, avg_rewards):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], color='white', fontsize=10)
    ax.set_ylabel('Avg Normalized Reward', fontsize=12)
    ax.set_title('Baseline Performance Across Scenarios', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(avg_rewards) * 1.3 if avg_rewards else 1.0)

    plt.tight_layout()
    return fig


def create_component_breakdown(eval_data: dict) -> plt.Figure:
    """Create stacked bar chart showing reward component breakdown by scenario."""
    if not eval_data or "error" in eval_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=14, color='white')
        _style_fig(fig, ax)
        return fig

    scenarios = list(eval_data.keys())
    components = ['R_era_task', 'R_teacher_delta', 'R_legacy_utility']
    labels = ['Era Task', 'Teacher Delta', 'Legacy Utility']

    fig, ax = plt.subplots(figsize=(10, 5))
    _style_fig(fig, ax)

    x = np.arange(len(scenarios))
    bottom = np.zeros(len(scenarios))

    for comp, label, color in zip(components, labels, COLORS):
        values = []
        for scenario in scenarios:
            runs = eval_data[scenario]
            avg = np.mean([r.get(comp, 0) for r in runs]) if runs else 0
            values.append(avg)
        ax.bar(x, values, 0.6, bottom=bottom, label=label, color=color, alpha=0.85, edgecolor='none')
        bottom += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], color='white', fontsize=10)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Reward Component Breakdown by Scenario', fontsize=14, fontweight='bold')
    ax.legend(facecolor=PANEL_BG, edgecolor='#444', labelcolor='white', fontsize=9)

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# GRADIO CALLBACKS
# ──────────────────────────────────────────────────────────────────────────────

def on_load_episode(episode_name):
    """Load and display an episode."""
    if not episode_name:
        return "⚠️ No episode selected", "", None, ""

    path = EPISODE_DIR / episode_name
    if not path.exists():
        return f"⚠️ File not found: {episode_name}", "", None, ""

    episode = load_json(path)
    if "error" in episode:
        return f"⚠️ Error loading: {episode['error']}", "", None, ""

    eras = episode.get("era_results", [])

    # Overview
    overview = f"""## 📋 Episode: {episode.get('scenario_name', 'Unknown')}
| | |
|---|---|
| **Scenario** | {episode.get('scenario_id', 'N/A')} |
| **Eras** | {episode.get('num_eras', len(eras))} |
| **Avg Reward** | {episode.get('avg_normalized_reward', 0):.4f} |
| **Timestamp** | {episode.get('timestamp', 'N/A')[:19]} |
"""

    # Era summaries
    era_parts = []
    for era in eras:
        reward = era.get("reward", {})
        criteria_met = era.get("criteria_met", [])
        criteria_total = era.get("criteria_total", [])
        met_str = ", ".join(criteria_met) if criteria_met else "None"

        era_md = f"""### Era {era.get('era_id', '?')}
| Metric | Value |
|--------|-------|
| Steps Taken | {era.get('steps_taken', 0)} |
| Criteria Met | {len(criteria_met)} / {len(criteria_total)} |
| Met | {met_str} |
| R_era_task | {reward.get('R_era_task', 0):.3f} |
| R_calibration | {reward.get('R_calibration', 1.0):.2f}x |
| R_teacher_delta | {reward.get('R_teacher_delta', 0):.3f} |
| R_legacy_utility | {reward.get('R_legacy_utility', 0):.3f} |
| R_leakage | {reward.get('R_answer_leakage', 0):.3f} |
| R_anti_hack | {reward.get('R_anti_hack_penalty', 0):.3f} |
| **R_total** | **{reward.get('R_total', 0):.3f}** |
| **R_normalized** | **{reward.get('R_normalized', 0):.4f}** |
"""
        era_parts.append(era_md)

    era_summaries = "\n---\n".join(era_parts)
    chart = create_reward_chart(episode)
    raw = json.dumps(episode, indent=2, default=str)

    return overview, era_summaries, chart, raw


def on_select_era(episode_name, era_idx):
    """Show drilldown for a specific era."""
    if not episode_name:
        return pd.DataFrame(), None

    path = EPISODE_DIR / episode_name
    if not path.exists():
        return pd.DataFrame(), None

    episode = load_json(path)
    eras = episode.get("era_results", [])
    idx = int(era_idx) - 1

    if idx < 0 or idx >= len(eras):
        return pd.DataFrame({"Info": ["Era not found"]}), None

    era = eras[idx]
    trajectory = era.get("trajectory", [])
    rows = []
    for entry in trajectory:
        action = entry.get("action", {})
        rows.append({
            "Step": entry.get("step", 0),
            "Agent": entry.get("agent", "unknown"),
            "Action": action.get("action_type", "?"),
            "Phase": entry.get("phase", ""),
            "Details": json.dumps(action.get("payload", {}))[:120],
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Step", "Agent", "Action", "Phase", "Details"])
    radar = create_radar_chart(era)

    return df, radar


def on_load_eval():
    """Load and display evaluation results."""
    results_files = find_eval_results()
    if not results_files:
        return "⚠️ No evaluation results found", None, None, ""

    # Load baseline results (primary file)
    eval_data = load_json(EVAL_DIR / results_files[0])
    if "error" in eval_data:
        return f"⚠️ Error: {eval_data['error']}", None, None, ""

    # Summary
    summary_parts = ["## 📊 Baseline Evaluation Results\n"]
    for scenario, runs in eval_data.items():
        avg_norm = np.mean([r.get("R_normalized", 0) for r in runs])
        avg_total = np.mean([r.get("R_total", 0) for r in runs])
        avg_task = np.mean([r.get("R_era_task", 0) for r in runs])
        n_runs = len(set(r.get("run", 0) for r in runs))
        n_eras = len(set(r.get("era_id", 0) for r in runs))

        summary_parts.append(f"""### {scenario.replace('_', ' ').title()}
| Metric | Value |
|--------|-------|
| Runs | {n_runs} |
| Eras/Run | {n_eras} |
| Avg R_era_task | {avg_task:.3f} |
| Avg R_total | {avg_total:.3f} |
| Avg R_normalized | {avg_norm:.4f} |
""")

    summary = "\n".join(summary_parts)
    comparison = create_baseline_comparison(eval_data)
    breakdown = create_component_breakdown(eval_data)
    raw = json.dumps(eval_data, indent=2, default=str)

    return summary, comparison, breakdown, raw


# ──────────────────────────────────────────────────────────────────────────────
# GRADIO UI
# ──────────────────────────────────────────────────────────────────────────────

def build_ui():
    """Build the Gradio interface."""
    episodes = find_episodes()
    default_episode = episodes[0] if episodes else None

    theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate",
    )

    with gr.Blocks(
        title="EpistemicOps — RL for Temporal Uncertainty & Oversight",
        theme=theme,
        css="""
        .gradio-container { max-width: 1200px !important; }
        .dark { background: #0f0f23 !important; }
        h1 { background: linear-gradient(135deg, #00d4ff, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.2em !important; }
        .info-badge { display: inline-block; padding: 4px 12px; border-radius: 12px; background: #1a1a2e; color: #00d4ff; font-size: 0.85em; margin: 2px; border: 1px solid #333; }
        """
    ) as app:

        gr.Markdown("""
# 🧠 EpistemicOps Dashboard
### Training Agents on Temporal Uncertainty, Generational Memory & Socratic Oversight
        """)

        gr.Markdown("""
<div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:16px;">
<span class="info-badge">🎯 OpenEnv-compliant RL Environment</span>
<span class="info-badge">🔀 Temporal Drift Detection</span>
<span class="info-badge">📜 2048-token Legacy Documents</span>
<span class="info-badge">🎓 Socratic Oversight + LLM Judge</span>
<span class="info-badge">📈 5-Component Reward Model</span>
</div>
        """)

        # ── Tab 1: Episode Replay ──────────────────────────────────────────
        with gr.Tab("🎬 Episode Replay"):
            with gr.Row():
                episode_dropdown = gr.Dropdown(
                    choices=episodes,
                    value=default_episode,
                    label="Select Episode",
                    interactive=True,
                    scale=3,
                )
                load_btn = gr.Button("▶ Load Episode", variant="primary", scale=1)

            with gr.Row():
                with gr.Column(scale=1):
                    overview_md = gr.Markdown(label="Overview")
                with gr.Column(scale=1):
                    reward_chart = gr.Plot(label="Reward per Era")

            era_summaries_md = gr.Markdown(label="Era Details")

            with gr.Accordion("📄 Raw JSON", open=False):
                raw_json = gr.Code(label="Raw Episode Data", language="json")

            # Era drilldown
            gr.Markdown("### 🔍 Era Drilldown")
            with gr.Row():
                era_selector = gr.Number(label="Era Number", value=1, precision=0, minimum=1, scale=1)
                drill_btn = gr.Button("Show Era Details", scale=1)

            with gr.Row():
                trajectory_table = gr.Dataframe(label="Step-by-Step Trajectory")
                radar_chart = gr.Plot(label="Reward Components Radar")

            load_btn.click(
                on_load_episode,
                inputs=[episode_dropdown],
                outputs=[overview_md, era_summaries_md, reward_chart, raw_json]
            )
            drill_btn.click(
                on_select_era,
                inputs=[episode_dropdown, era_selector],
                outputs=[trajectory_table, radar_chart]
            )

        # ── Tab 2: Evaluation Results ──────────────────────────────────────
        with gr.Tab("📊 Evaluation Results"):
            eval_load_btn = gr.Button("Load Baseline Evaluation", variant="primary")

            eval_summary_md = gr.Markdown()

            with gr.Row():
                eval_comparison = gr.Plot(label="Cross-Scenario Comparison")
                eval_breakdown = gr.Plot(label="Component Breakdown")

            with gr.Accordion("📄 Raw Results JSON", open=False):
                eval_raw = gr.Code(label="Raw Evaluation Data", language="json")

            eval_load_btn.click(
                on_load_eval,
                outputs=[eval_summary_md, eval_comparison, eval_breakdown, eval_raw]
            )

        # ── Tab 3: Architecture & About ────────────────────────────────────
        with gr.Tab("🏗️ Architecture"):
            gr.Markdown("""
## What is EpistemicOps?

EpistemicOps is a Reinforcement Learning environment that trains AI agents to handle three critical production failure modes simultaneously:

### 1. 🔀 Temporal Drift
API contracts change silently mid-episode. The agent must detect this through downstream failures, not by being told.

### 2. 📜 Generational Memory
Context is wiped between eras — only a 2048-token Legacy Document survives. The agent must write actionable intel for its successor.

### 3. 🎓 Socratic Oversight
When the student agent fails, a teacher agent guides it using only Socratic questions. If the teacher gives away the answer, an LLM Judge penalizes it.

---

### Architecture
```
┌────────────────────────────────────────────────────────────────┐
│                    EpistemicOps Engine                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ World Engine  │  │ Action       │  │ Legacy Parser        │ │
│  │ (state mgmt)  │  │ Validator    │  │ (2048-token limit)   │ │
│  └──────┬───────┘  └──────────────┘  └──────────────────────┘ │
│         │                                                      │
│  ┌──────▼───────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Drift        │  │ Leakage      │  │ OpenEnv Wrapper      │ │
│  │ Injector     │  │ Detector     │  │ (step/reset/state)   │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
         │                    │                    │
    ┌────▼────┐         ┌────▼────┐          ┌────▼────┐
    │ Mock    │         │ Primary │          │Oversight│
    │ APIs    │         │ Agent   │          │ Agent   │
    │(5 svcs) │         │(Student)│          │(Teacher)│
    └─────────┘         └─────────┘          └────┬────┘
                                                   │
                                             ┌─────▼─────┐
                                             │ LLM Judge │
                                             │(leakage)  │
                                             └───────────┘
```

### Reward Model
```
R_total = (R_era_task × R_calibration) + R_teacher_delta + R_legacy_utility + R_leakage + R_anti_hack

Where:
  R_era_task       ∈ [0, 1]    — fraction of success criteria met
  R_calibration    ∈ [0.5, 1.5] — penalizes over/under-confident hypotheses
  R_teacher_delta  ∈ [0, 1]    — student improvement after oversight
  R_legacy_utility ∈ [0, 1]    — how much the legacy doc helps the next era
  R_leakage        ∈ [-1, 0]   — penalty for teacher giving away answers
  R_anti_hack      ∈ [-1, 0]   — penalty for action-spamming exploits

R_max = 3.5 → R_normalized = R_total / 3.5
```

### Phase State Machine
```
AWAKENING → OPERATION → DRIFT_INJECTION → SOCRATIC_RECOVERY → LEGACY_GENERATION → ERA_END
```

### Training
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Base Model**: Llama 3.1 8B Instruct (4-bit via Unsloth)
- **Framework**: HuggingFace TRL
            """)

        # ── Tab 4: Scenarios ───────────────────────────────────────────────
        with gr.Tab("📋 Scenarios"):
            gr.Markdown("""
## Available Scenarios

### 1. The Cascading Incident
A P2 incident cascades across services. The agent must triage, investigate, and resolve while
the incident-api's response schema silently changes mid-episode.

**Success Criteria:** incident_resolved, deploy_successful, legacy_doc_written

---

### 2. Deployment Disaster
A bad deployment has gone out and the deploy-api's auth header format changes.
The agent must detect the broken deployment, roll back, and verify.

**Success Criteria:** deploy_successful, rollback_complete, legacy_doc_written

---

### 3. The Invisible Outage
Services appear healthy but a silent latency regression is occurring.
The metrics-api renames a key field, making the regression invisible unless the agent adapts.

**Success Criteria:** incident_identified, proxies_scaled, legacy_doc_written
            """)

    return app


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
else:
    # HF Spaces auto-discovery
    app = build_ui()
