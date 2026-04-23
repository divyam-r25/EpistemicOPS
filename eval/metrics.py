"""
eval/metrics.py
===============
Compute all primary evaluation metrics from a list of episode results.
Used by benchmark.py and the Colab notebook to produce the before/after table.
"""
from typing import List, Dict
import statistics


def compute_all_metrics(episode_results: List[List[Dict]]) -> Dict:
    """
    Compute all primary metrics from raw episode results.

    Args:
        episode_results: List of episodes, each episode is a list of era dicts.
            Each era dict must have: drift_detection_rate, legacy_doc_written,
            steps_taken, total_drifts, drifts_detected.

    Returns:
        Dict with mean and std for each metric.
    """
    flat = [era for episode in episode_results for era in episode]

    def mean_std(values):
        if not values:
            return 0.0, 0.0
        m = statistics.mean(values)
        s = statistics.stdev(values) if len(values) > 1 else 0.0
        return round(m, 4), round(s, 4)

    drift_rates = [e["drift_detection_rate"] for e in flat]
    legacy_rates = [1.0 if e.get("legacy_doc_written") else 0.0 for e in flat]
    step_counts = [e.get("steps_taken", 40) for e in flat]

    drift_mean, drift_std = mean_std(drift_rates)
    legacy_mean, legacy_std = mean_std(legacy_rates)
    steps_mean, steps_std = mean_std(step_counts)

    return {
        "drift_detection_rate":      {"mean": drift_mean, "std": drift_std},
        "legacy_doc_completion_rate": {"mean": legacy_mean, "std": legacy_std},
        "avg_steps_per_era":         {"mean": steps_mean, "std": steps_std},
        "num_episodes":              len(episode_results),
        "num_eras_total":            len(flat),
    }


def format_comparison_table(baseline: Dict, trained: Dict) -> str:
    """Produce a human-readable before/after comparison table."""
    rows = [
        ("Drift Detection Rate",
         baseline["drift_detection_rate"]["mean"],
         trained["drift_detection_rate"]["mean"]),
        ("Legacy Doc Completion",
         baseline["legacy_doc_completion_rate"]["mean"],
         trained["legacy_doc_completion_rate"]["mean"]),
    ]
    lines = [
        "=" * 65,
        f"{'Metric':<30}  {'Baseline':>10}  {'Trained':>10}  {'Δ':>8}",
        "-" * 65,
    ]
    for name, before, after in rows:
        delta = after - before
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"{name:<30}  {before:>10.1%}  {after:>10.1%}  "
            f"{sign}{delta:>7.1%}"
        )
    lines.append("=" * 65)
    return "\n".join(lines)
