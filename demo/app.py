"""
EpistemicOps Gradio Demo
=========================
Works in three modes:
1. Replay Mode (default): Load and replay a pre-recorded episode JSON
2. Live Mode: Run episodes in real-time against the environment
3. Results Mode: Compare baseline vs trained reward curves

Usage:
    python demo/app.py
    python demo/app.py --episode episodes/test_run.json
"""
import gradio as gr
import json
import sys
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_episode(path: str) -> dict:
    """Load a recorded episode JSON."""
    with open(path) as f:
        return json.load(f)


def find_default_episode() -> str:
    """Find a default episode file."""
    candidates = [
        Path(__file__).parent.parent / "episodes" / "test_run.json",
        Path(__file__).parent.parent / "episodes" / "sample_episode.json",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return ""


def build_era_summary(era: dict) -> str:
    """Build a markdown summary for an era."""
    reward = era.get("reward", {})
    criteria_met = era.get("criteria_met", [])
    criteria_total = era.get("criteria_total", [])
    steps = era.get("steps_taken", 0)
    
    met_str = ", ".join(criteria_met) if criteria_met else "None"
    total_str = ", ".join(criteria_total) if criteria_total else "N/A"
    
    md = f"""### Era {era.get('era_id', '?')}
| Metric | Value |
|--------|-------|
| Steps Taken | {steps} |
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
    return md


def build_trajectory_table(era: dict) -> pd.DataFrame:
    """Build a pandas DataFrame of the trajectory steps."""
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
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Step", "Agent", "Action", "Phase", "Details"])


def create_reward_chart(episode: dict) -> plt.Figure:
    """Create a bar chart of rewards per era."""
    eras = episode.get("era_results", [])
    if not eras:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=14)
        return fig
    
    era_ids = [f"Era {e.get('era_id', '?')}" for e in eras]
    components = ['R_era_task', 'R_teacher_delta', 'R_legacy_utility']
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    
    x = np.arange(len(era_ids))
    width = 0.2
    
    for i, (comp, color) in enumerate(zip(components, colors)):
        values = [e.get("reward", {}).get(comp, 0) for e in eras]
        ax.bar(x + i * width, values, width, label=comp.replace('R_', ''), color=color, alpha=0.85)
    
    # Total line
    totals = [e.get("reward", {}).get("R_total", 0) for e in eras]
    ax.plot(x + width, totals, 'w--o', label='R_total', linewidth=2, markersize=8)
    
    ax.set_xlabel('Era', color='white', fontsize=12)
    ax.set_ylabel('Reward', color='white', fontsize=12)
    ax.set_title(f'Episode Rewards — {episode.get("scenario_name", "Unknown")}', color='white', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(era_ids, color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
    ax.grid(axis='y', alpha=0.2, color='white')
    
    plt.tight_layout()
    return fig


def create_component_radar(era: dict) -> plt.Figure:
    """Create a radar chart of reward components for a single era."""
    reward = era.get("reward", {})
    
    categories = ['Task', 'Calibration', 'Teacher\nDelta', 'Legacy\nUtility', 'Leakage', 'Anti-hack']
    values = [
        max(0, reward.get('R_era_task', 0)),
        max(0, (reward.get('R_calibration', 1.0) - 0.5) / 1.0),  # Normalize 0.5-1.5 to 0-1
        max(0, reward.get('R_teacher_delta', 0)),
        max(0, reward.get('R_legacy_utility', 0)),
        max(0, 1.0 + reward.get('R_answer_leakage', 0)),  # Flip: -1 = bad → 0, 0 = good → 1
        max(0, 1.0 + reward.get('R_anti_hack_penalty', 0)),
    ]
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#00d4ff')
    ax.fill(angles, values, alpha=0.2, color='#00d4ff')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white', fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(f'Era {era.get("era_id", "?")} — Reward Components', color='white', fontsize=13, fontweight='bold', pad=20)
    ax.tick_params(colors='white')
    ax.grid(color='white', alpha=0.2)
    
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# GRADIO UI
# ──────────────────────────────────────────────────────────────────────────────

def on_load_episode(file_path):
    """Load episode from file path."""
    if not file_path or not Path(file_path).exists():
        return "No episode file found. Run: python run_episode.py --record episodes/test_run.json", None, None, None
    
    episode = load_episode(file_path)
    eras = episode.get("era_results", [])
    
    # Overview markdown
    overview = f"""## Episode: {episode.get('scenario_name', 'Unknown')}
| | |
|---|---|
| **Scenario** | {episode.get('scenario_id', 'N/A')} |
| **Eras** | {episode.get('num_eras', 0)} |
| **Avg Reward** | {episode.get('avg_normalized_reward', 0):.4f} |
| **Timestamp** | {episode.get('timestamp', 'N/A')} |
"""
    
    # Era summaries
    era_summaries = "\n---\n".join(build_era_summary(e) for e in eras)
    
    # Reward chart
    chart = create_reward_chart(episode)
    
    return overview, era_summaries, chart, json.dumps(episode, indent=2, default=str)


def on_select_era(file_path, era_idx):
    """Show details for a specific era."""
    if not file_path or not Path(file_path).exists():
        return pd.DataFrame(), None
    
    episode = load_episode(file_path)
    eras = episode.get("era_results", [])
    era_idx = int(era_idx) - 1
    
    if era_idx < 0 or era_idx >= len(eras):
        return pd.DataFrame(), None
    
    era = eras[era_idx]
    trajectory_df = build_trajectory_table(era)
    radar = create_component_radar(era)
    
    return trajectory_df, radar


def build_ui():
    """Build the Gradio interface."""
    default_path = find_default_episode()
    
    with gr.Blocks(
        title="EpistemicOps Dashboard",
        theme=gr.themes.Soft(
            primary_hue="cyan",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
        css="""
        .gradio-container { max-width: 1200px !important; }
        .dark { background: #0f0f23 !important; }
        """
    ) as app:
        gr.Markdown("""
# EpistemicOps Dashboard
### Training Agents on Temporal Uncertainty and Socratic Oversight
        """)
        
        with gr.Tab("Episode Replay"):
            with gr.Row():
                episode_path = gr.Textbox(
                    label="Episode JSON Path",
                    value=default_path,
                    placeholder="episodes/test_run.json"
                )
                load_btn = gr.Button("Load Episode", variant="primary")
            
            with gr.Row():
                with gr.Column(scale=1):
                    overview_md = gr.Markdown(label="Overview")
                with gr.Column(scale=1):
                    reward_chart = gr.Plot(label="Reward per Era")
            
            era_summaries_md = gr.Markdown(label="Era Details")
            raw_json = gr.Code(label="Raw JSON", language="json", visible=False)
            
            # Era drilldown
            gr.Markdown("### Era Drilldown")
            with gr.Row():
                era_selector = gr.Number(label="Era Number", value=1, precision=0, minimum=1)
                drill_btn = gr.Button("Show Era Details")
            
            with gr.Row():
                trajectory_table = gr.Dataframe(label="Step-by-Step Trajectory")
                radar_chart = gr.Plot(label="Reward Components Radar")
            
            load_btn.click(
                on_load_episode,
                inputs=[episode_path],
                outputs=[overview_md, era_summaries_md, reward_chart, raw_json]
            )
            drill_btn.click(
                on_select_era,
                inputs=[episode_path, era_selector],
                outputs=[trajectory_table, radar_chart]
            )
        
        with gr.Tab("About"):
            gr.Markdown("""
## What is EpistemicOps?

EpistemicOps is a Reinforcement Learning environment that trains AI agents to handle three critical production failure modes simultaneously:

1. **Temporal Drift**: API contracts change silently mid-episode
2. **Generational Memory**: Context is wiped between eras — only a Legacy Document survives  
3. **Socratic Oversight**: When agents fail, a teacher agent guides them using only questions

### Architecture
```
Primary Agent ←→ EpistemicOps Environment ←→ 5 Mock APIs
      ↑                    ↓
Oversight Agent       Drift Injector
      ↑                    ↓
  LLM Judge          Reward Model (5 components)
```

### Quick Start
```bash
pip install -r requirements.txt
python run_episode.py --scenario cascading_incident --record episodes/demo.json
python demo/app.py --episode episodes/demo.json
```
            """)
    
    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", default=None, help="Default episode file to load")
    args = parser.parse_args()
    
    app = build_ui()
    app.launch(share=False)
