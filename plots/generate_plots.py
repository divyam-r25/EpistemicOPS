"""
Generate baseline reward plots for README and submission.

Usage:
    python plots/generate_plots.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = Path(__file__).parent
EVAL_DIR = Path(__file__).parent.parent / "eval_results"

DARK_BG = '#1a1a2e'
PANEL_BG = '#16213e'
COLORS = {
    'task':    '#4CAF50',
    'calib':   '#00BCD4',
    'teacher': '#2196F3',
    'legacy':  '#FF9800',
    'leak':    '#E91E63',
    'hack':    '#9C27B0',
    'total':   '#00d4ff',
}


def load_baseline():
    path = EVAL_DIR / "baseline_results.json"
    if not path.exists():
        print(f"No baseline results at {path}. Running baseline_eval first...")
        import asyncio
        from training.baseline_eval import run_baseline_evaluation
        asyncio.run(run_baseline_evaluation())
    with open(path) as f:
        return json.load(f)


def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color='white', fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, color='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.grid(axis='y', alpha=0.15, color='white')


def plot_1_baseline_rewards(baseline):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)

    scenarios = []
    avg_totals = []
    avg_norms = []
    for sid, runs in baseline.items():
        valid = [r for r in runs if "error" not in r]
        if valid:
            scenarios.append(sid.replace('_', '\n'))
            avg_totals.append(sum(r["R_total"] for r in valid) / len(valid))
            avg_norms.append(sum(r["R_normalized"] for r in valid) / len(valid))

    x = np.arange(len(scenarios))
    w = 0.35
    ax.bar(x - w/2, avg_totals, w, label='R_total', color=COLORS['total'], alpha=0.9)
    ax.bar(x + w/2, avg_norms, w, label='R_normalized', color=COLORS['teacher'], alpha=0.9)

    for i, (t, n) in enumerate(zip(avg_totals, avg_norms)):
        ax.text(i - w/2, t + 0.02, f'{t:.2f}', ha='center', color='white', fontsize=10, fontweight='bold')
        ax.text(i + w/2, n + 0.02, f'{n:.3f}', ha='center', color='white', fontsize=10)

    style_ax(ax, 'Baseline Rewards by Scenario (Mock Agent, No Training)', ylabel='Reward')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, color='white', fontsize=11)
    ax.legend(facecolor=DARK_BG, edgecolor='#444', labelcolor='white', fontsize=10)
    ax.set_ylim(0, max(avg_totals) * 1.4)
    plt.tight_layout()

    out = PLOTS_DIR / "baseline_rewards_by_scenario.png"
    fig.savefig(out, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] {out}")


def plot_2_reward_components(baseline):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)

    components = ['R_era_task', 'R_teacher_delta', 'R_legacy_utility', 'R_answer_leakage', 'R_anti_hack_penalty']
    labels = ['Era Task', 'Teacher Δ', 'Legacy Util', 'Leakage', 'Anti-hack']
    colors = [COLORS['task'], COLORS['teacher'], COLORS['legacy'], COLORS['leak'], COLORS['hack']]

    scenarios = list(baseline.keys())
    x = np.arange(len(scenarios))
    width = 0.15

    for i, (comp, label, color) in enumerate(zip(components, labels, colors)):
        values = []
        for sid in scenarios:
            valid = [r for r in baseline[sid] if "error" not in r]
            avg = sum(r.get(comp, 0) for r in valid) / max(1, len(valid))
            values.append(avg)
        offset = (i - len(components) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label, color=color, alpha=0.85)

    style_ax(ax, 'Reward Component Breakdown by Scenario', ylabel='Avg Reward')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], color='white', fontsize=10)
    ax.legend(facecolor=DARK_BG, edgecolor='#444', labelcolor='white', fontsize=9, loc='upper right')
    ax.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
    plt.tight_layout()

    out = PLOTS_DIR / "reward_components_breakdown.png"
    fig.savefig(out, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] {out}")


def plot_3_before_after(baseline):
    proof_path = EVAL_DIR / "proof_of_learning.json"
    if not proof_path.exists():
        print(f"  [WARN] {proof_path} not found; skipping baseline_vs_trained_comparison.png")
        return

    with open(proof_path) as f:
        proof = json.load(f)

    base = proof.get("summary", {}).get("baseline", {})
    trained = proof.get("summary", {}).get("trained", {})
    metrics = [
        ("avg_reward", "Avg Reward"),
        ("avg_criteria_completion", "Criteria Completion"),
        ("drift_detection_rate", "Drift Detection"),
        ("incident_resolution_rate", "Incident Resolution"),
        ("legacy_doc_rate", "Legacy Doc"),
    ]
    labels = [m[1] for m in metrics]
    baseline_vals = [base.get(m[0], 0.0) for m in metrics]
    trained_vals = [trained.get(m[0], 0.0) for m in metrics]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, baseline_vals, w, label='Baseline', color='#666', alpha=0.8)
    ax.bar(x + w / 2, trained_vals, w, label='Trained', color=COLORS['total'], alpha=0.9)

    for i, (b, t) in enumerate(zip(baseline_vals, trained_vals)):
        ax.text(i - w / 2, b + 0.01, f'{b:.2f}', ha='center', color='#ccc', fontsize=9)
        ax.text(i + w / 2, t + 0.01, f'{t:.2f}', ha='center', color='white', fontsize=9, fontweight='bold')

    style_ax(ax, 'Baseline vs Trained (Measured, Not Projected)', ylabel='Score')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color='white', fontsize=9)
    ax.set_ylim(0, max(max(baseline_vals), max(trained_vals)) * 1.35 if labels else 1.0)
    ax.legend(facecolor=DARK_BG, edgecolor='#444', labelcolor='white', fontsize=9)
    plt.tight_layout()

    out = PLOTS_DIR / "baseline_vs_trained_comparison.png"
    fig.savefig(out, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] {out}")


def plot_4_episode_timeline(baseline):
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    era_colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0']

    # Use cascading_incident data
    runs = baseline.get("cascading_incident", [])
    valid = [r for r in runs if "error" not in r]

    total_steps = 0
    for i, era_data in enumerate(valid[:6]):
        era_id = era_data.get("era_id", i + 1)
        steps = era_data.get("steps_taken", 15)
        drifts = era_data.get("drifts_fired", 0)
        oversight = era_data.get("oversight_interventions", 0)
        r_norm = era_data.get("R_normalized", 0)

        ax.axvspan(total_steps, total_steps + steps, alpha=0.15, color=era_colors[i % 5])
        ax.text(total_steps + steps / 2, 2.2, f"Era {era_id}\nR={r_norm:.2f}",
                ha='center', va='center', color='white', fontsize=9, fontweight='bold')

        if drifts > 0:
            drift_step = total_steps + int(steps * 0.6)
            ax.axvline(x=drift_step, color='#FF4444', linewidth=2.5, alpha=0.9)
            ax.text(drift_step, 3.2, "DRIFT", ha='center', color='#FF4444', fontsize=8, fontweight='bold')

        if oversight > 0:
            ov_step = total_steps + int(steps * 0.7)
            ax.axvline(x=ov_step, color='#00BCD4', linewidth=2, alpha=0.8, linestyle='--')
            ax.text(ov_step, 0.8, "OV", ha='center', fontsize=10, color='#00BCD4', fontweight='bold')

        total_steps += steps

    ax.set_xlim(0, total_steps)
    ax.set_ylim(0, 4)
    ax.set_xlabel('Step (cumulative)', color='white', fontsize=11)
    ax.set_title('Episode Timeline — Cascading Incident (baseline eval, 3 eras per run)', color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    ax.set_yticks([])

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#FF4444', linewidth=2, label='Drift Injection'),
        Line2D([0], [0], color='#00BCD4', linewidth=2, linestyle='--', label='Socratic Oversight'),
    ]
    ax.legend(handles=legend_elements, facecolor=DARK_BG, edgecolor='#444', labelcolor='white', fontsize=9, loc='upper right')

    plt.tight_layout()
    out = PLOTS_DIR / "drift_detection_timeline.png"
    fig.savefig(out, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] {out}")


if __name__ == "__main__":
    np.random.seed(42)
    print("Generating plots...")
    baseline = load_baseline()
    plot_1_baseline_rewards(baseline)
    plot_2_reward_components(baseline)
    plot_3_before_after(baseline)
    plot_4_episode_timeline(baseline)
    print(f"\nAll plots saved to {PLOTS_DIR}/")
