import gradio as gr
import json
import sys
import os
import asyncio
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def load_episode(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def find_default_episode() -> str:
    for name in ["test_run.json", "sample_episode.json"]:
        p = Path(__file__).parent / "episodes" / name
        if p.exists():
            return str(p)
    return ""

def build_era_summary(era: dict) -> str:
    reward = era.get("reward", {})
    criteria_met = era.get("criteria_met", [])
    criteria_total = era.get("criteria_total", [])
    steps = era.get("steps_taken", 0)
    drifts = era.get("drifts_fired", 0)
    oversight = era.get("oversight_interventions", 0)

    drift_badge = f"🔴 **{drifts} drift(s)**" if drifts > 0 else "🟢 No drifts"
    oversight_badge = f"🧑‍🏫 **{oversight} intervention(s)**" if oversight > 0 else "—"

    return f"""### Era {era.get('era_id', '?')}  {drift_badge}  |  {oversight_badge}
| Metric | Value |
|--------|-------|
| Steps Taken | {steps} |
| Criteria Met | {len(criteria_met)} / {len(criteria_total)} |
| Met | {", ".join(criteria_met) or "None"} |
| R_era_task | {reward.get('R_era_task', 0):.3f} |
| R_calibration | {reward.get('R_calibration', 1.0):.2f}× |
| R_teacher_delta | {reward.get('R_teacher_delta', 0):.3f} |
| R_legacy_utility | {reward.get('R_legacy_utility', 0):.3f} |
| R_leakage | {reward.get('R_answer_leakage', 0):.3f} |
| R_anti_hack | {reward.get('R_anti_hack_penalty', 0):.3f} |
| **R_total** | **{reward.get('R_total', 0):.3f}** |
| **R_normalized** | **{reward.get('R_normalized', 0):.4f}** |
"""

def build_trajectory_table(era: dict) -> pd.DataFrame:
    phase_emoji = {
        "OPERATION": "⚙️", "DRIFT_INJECTION": "🔴",
        "SOCRATIC_RECOVERY": "🧑‍🏫", "LEGACY_GENERATION": "📝", "AWAKENING": "🌅"
    }
    rows = []
    for entry in era.get("trajectory", []):
        action = entry.get("action", {})
        phase = entry.get("phase", "")
        rows.append({
            "Step": entry.get("step", 0),
            "Agent": entry.get("agent", "unknown"),
            "Action": action.get("action_type", "?"),
            "Phase": f"{phase_emoji.get(phase, '')} {phase}",
            "Details": json.dumps(action.get("payload", {}))[:150],
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Step", "Agent", "Action", "Phase", "Details"])

def create_reward_chart(episode: dict) -> plt.Figure:
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
        ax.bar(x + i * width, values, width, label=comp.replace('R_', '').replace('_', ' ').title(), color=color, alpha=0.85)

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

def create_drift_timeline(episode: dict) -> plt.Figure:
    eras = episode.get("era_results", [])
    era_colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0']

    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    total_steps = 0
    for era in eras:
        era_steps = era.get("steps_taken", 0)
        era_id = era.get("era_id", 0)
        ax.axvspan(total_steps, total_steps + era_steps, alpha=0.15, color=era_colors[era_id % 5])
        ax.text(total_steps + era_steps / 2, 1.5, f"Era {era_id}", ha='center', va='center', color='white', fontsize=10, fontweight='bold')

        if era.get("drifts_fired", 0) > 0:
            for t in era.get("trajectory", []):
                if t.get("phase") == "DRIFT_INJECTION" and t.get("agent") == "primary":
                    x = total_steps + t.get("step", 0)
                    ax.axvline(x=x, color='#FF4444', linewidth=2, alpha=0.8)
                    ax.text(x, 2.5, "⚡DRIFT", ha='center', color='#FF4444', fontsize=8, rotation=45)
                    break

        if era.get("oversight_interventions", 0) > 0:
            for t in era.get("trajectory", []):
                if t.get("agent") == "oversight":
                    x = total_steps + t.get("step", 0)
                    ax.axvline(x=x, color='#00BCD4', linewidth=2, alpha=0.8, linestyle='--')
                    ax.text(x, 0.5, "🧑‍🏫", ha='center', fontsize=10)
                    break

        total_steps += era_steps

    ax.set_xlim(0, total_steps)
    ax.set_ylim(0, 3)
    ax.set_xlabel('Step (cumulative)', color='white', fontsize=11)
    ax.set_title('Episode Timeline: Drift Injection & Oversight Events', color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    ax.set_yticks([])
    plt.tight_layout()
    return fig

def load_baseline_results():
    path = Path(__file__).parent / "eval_results" / "baseline_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def create_baseline_comparison_chart(baseline: dict) -> plt.Figure:
    if not baseline:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No baseline results found", ha='center', va='center', fontsize=14, color='white')
        fig.patch.set_facecolor('#1a1a2e')
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a2e')

    metrics = ['R_era_task', 'R_total', 'R_normalized']
    titles = ['Era Task Completion', 'Total Reward', 'Normalized Reward']
    colors_list = ['#4CAF50', '#2196F3', '#FF9800']

    for ax, metric, title, color in zip(axes, metrics, titles, colors_list):
        ax.set_facecolor('#16213e')
        scenario_names, values = [], []
        for scenario_id, runs in baseline.items():
            valid = [r for r in runs if "error" not in r]
            if valid:
                avg = sum(r.get(metric, 0) for r in valid) / len(valid)
                scenario_names.append(scenario_id.replace('_', '\n'))
                values.append(avg)
        ax.bar(scenario_names, values, color=color, alpha=0.85)
        ax.set_title(title, color='white', fontsize=12, fontweight='bold')
        ax.tick_params(colors='white')
        ax.set_ylim(0, max(values) * 1.3 if values else 1)
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', color='white', fontsize=10)
        ax.grid(axis='y', alpha=0.15, color='white')

    fig.suptitle('Baseline Evaluation Across Scenarios', color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def on_load_episode(file_path):
    if not file_path or not Path(file_path).exists():
        return "No episode file found.", None, None, None, None
    episode = load_episode(file_path)
    eras = episode.get("era_results", [])
    total_drifts = sum(e.get("drifts_fired", 0) for e in eras)
    total_oversight = sum(e.get("oversight_interventions", 0) for e in eras)
    avg_r = episode.get("avg_normalized_reward", 0)

    overview = f"""## Episode: {episode.get('scenario_name', 'Unknown')}
| | |
|---|---|
| **Scenario** | `{episode.get('scenario_id', 'N/A')}` |
| **Eras** | {episode.get('num_eras', 0)} |
| **Avg Normalized Reward** | **{avg_r:.4f}** |
| **Total Drifts Fired** | {total_drifts} |
| **Total Oversight Interventions** | {total_oversight} |
| **Timestamp** | {episode.get('timestamp', 'N/A')} |
"""
    era_summaries = "\n---\n".join(build_era_summary(e) for e in eras)
    return overview, era_summaries, create_reward_chart(episode), create_drift_timeline(episode), json.dumps(episode, indent=2, default=str)

def on_select_era(file_path, era_idx):
    if not file_path or not Path(file_path).exists():
        return pd.DataFrame(), None
    episode = load_episode(file_path)
    eras = episode.get("era_results", [])
    idx = int(era_idx) - 1
    if idx < 0 or idx >= len(eras):
        return pd.DataFrame(), None
    era = eras[idx]
    return build_trajectory_table(era), create_component_radar(era)

def on_run_simulation(scenario_id, num_eras):
    try:
        from run_episode import run_full_episode
        record_path = str(Path(__file__).parent / "episodes" / f"live_{scenario_id}.json")
        asyncio.run(run_full_episode(scenario_id=scenario_id, num_eras=int(num_eras), record_path=record_path))
        episode = load_episode(record_path)
        eras = episode.get("era_results", [])
        overview = f"## ✅ Simulation Complete: {episode.get('scenario_name', scenario_id)}\n"
        overview += f"**Avg Reward: {episode.get('avg_normalized_reward', 0):.4f}**\n\n"
        for e in eras:
            r = e.get("reward", {})
            overview += f"- Era {e['era_id']}: R_norm={r.get('R_normalized', 0):.4f}, drifts={e.get('drifts_fired', 0)}, oversight={e.get('oversight_interventions', 0)}\n"
        return overview, create_reward_chart(episode), create_drift_timeline(episode)
    except Exception:
        import traceback
        return f"## ❌ Simulation Failed\n```\n{traceback.format_exc()}\n```", None, None


def build_ui():
    default_path = find_default_episode()
    baseline = load_baseline_results()

    with gr.Blocks(
        title="EpistemicOps Dashboard",
        theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="blue", neutral_hue="slate"),
        css="""
        .gradio-container { max-width: 1300px !important; }
        .dark { background: #0f0f23 !important; }
        h1 { background: linear-gradient(90deg, #00d4ff, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        """
    ) as app:
        gr.Markdown("# 🧠 EpistemicOps Dashboard\n### Training Agents on Temporal Drift, Generational Memory & Socratic Oversight")

        with gr.Tab("📊 Episode Replay"):
            with gr.Row():
                episode_path = gr.Textbox(label="Episode JSON Path", value=default_path, placeholder="episodes/test_run.json", scale=3)
                load_btn = gr.Button("🔄 Load Episode", variant="primary", scale=1)
            with gr.Row():
                with gr.Column(scale=1):
                    overview_md = gr.Markdown(label="Overview")
                with gr.Column(scale=1):
                    reward_chart = gr.Plot(label="Reward per Era")
            timeline_chart = gr.Plot(label="Drift & Oversight Timeline")
            era_summaries_md = gr.Markdown(label="Era Details")
            raw_json = gr.Code(label="Raw JSON", language="json", visible=False)

            gr.Markdown("### 🔍 Era Drilldown")
            with gr.Row():
                era_selector = gr.Number(label="Era Number", value=1, precision=0, minimum=1)
                drill_btn = gr.Button("Show Era Details")
            with gr.Row():
                trajectory_table = gr.Dataframe(label="Step-by-Step Trajectory")
                radar_chart = gr.Plot(label="Reward Components Radar")

            load_btn.click(on_load_episode, inputs=[episode_path], outputs=[overview_md, era_summaries_md, reward_chart, timeline_chart, raw_json])
            drill_btn.click(on_select_era, inputs=[episode_path, era_selector], outputs=[trajectory_table, radar_chart])

        with gr.Tab("🚀 Live Simulation"):
            gr.Markdown("### Run a simulation episode (offline mode — no Docker needed)")
            with gr.Row():
                scenario_dd = gr.Dropdown(choices=["cascading_incident", "deployment_disaster", "invisible_outage"], value="cascading_incident", label="Scenario")
                eras_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Number of Eras")
                run_btn = gr.Button("▶️ Run Simulation", variant="primary")
            sim_output = gr.Markdown(label="Simulation Results")
            with gr.Row():
                sim_chart = gr.Plot(label="Reward Chart")
                sim_timeline = gr.Plot(label="Timeline")
            run_btn.click(on_run_simulation, inputs=[scenario_dd, eras_slider], outputs=[sim_output, sim_chart, sim_timeline])

        with gr.Tab("📈 Baseline Results"):
            if baseline:
                gr.Markdown("### Baseline Performance — Mock Agent, No Training")
                gr.Plot(value=create_baseline_comparison_chart(baseline), label="Baseline Comparison")
                for scenario_id, runs in baseline.items():
                    valid = [r for r in runs if "error" not in r]
                    if valid:
                        avg_total = sum(r["R_total"] for r in valid) / len(valid)
                        avg_norm = sum(r["R_normalized"] for r in valid) / len(valid)
                        avg_task = sum(r["R_era_task"] for r in valid) / len(valid)
                        avg_drifts = sum(r.get("drifts_fired", 0) for r in valid) / len(valid)
                        gr.Markdown(f"**{scenario_id}**: R_total={avg_total:.3f} | R_norm={avg_norm:.4f} | R_task={avg_task:.3f} | Drifts={avg_drifts:.1f}")
            else:
                gr.Markdown("No baseline results found. Run `python training/baseline_eval.py` first.")

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## What is EpistemicOps?

EpistemicOps is a Reinforcement Learning environment that trains AI agents to handle three critical production failure modes simultaneously:

1. **🔴 Temporal Drift**: API contracts change silently mid-episode. The agent must detect schema changes through downstream failures.
2. **📝 Generational Memory**: Context is wiped between eras — only a 2048-token Legacy Document survives to the next generation.
3. **🧑‍🏫 Socratic Oversight**: When agents fail, a teacher agent guides them using only questions — never giving the answer directly.

### Reward Model (5 Components)
```
R_total = (R_era_task × R_calibration) + R_teacher_delta + R_legacy_utility + R_leakage + R_anti_hack
```

### Architecture
```
Primary Agent (Student) ←→ EpistemicOps Environment ←→ 5 Mock APIs (Docker)
       ↑                         ↓
Oversight Agent (Teacher)   Drift Injector → Silent schema changes
       ↑                         ↓
   LLM Judge              Reward Model (5 components, max 3.5)
```
""")

        if default_path:
            app.load(on_load_episode, inputs=[episode_path], outputs=[overview_md, era_summaries_md, reward_chart, timeline_chart, raw_json])

    return app

if __name__ == "__main__":
    app = build_ui()
    app.launch(share=False)
