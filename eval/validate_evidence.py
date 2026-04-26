"""
Validate judge-facing evidence artifacts for consistency and completeness.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate(root: Path) -> list[str]:
    warnings: list[str] = []
    eval_dir = root / "eval_results"
    plots_dir = root / "plots"

    proof = _load_json(eval_dir / "proof_of_learning.json")
    metadata = _load_json(eval_dir / "proof_run_metadata.json")
    if not proof:
        warnings.append("Missing eval_results/proof_of_learning.json")
        return warnings
    if not metadata:
        warnings.append("Missing eval_results/proof_run_metadata.json")

    required_plot_files = ["proof_reward_curve.png", "proof_before_vs_after.png"]
    for name in required_plot_files:
        if not (plots_dir / name).exists():
            warnings.append(f"Missing plots/{name}")

    summary = proof.get("summary", {})
    for side in ("baseline", "trained"):
        side_metrics = summary.get(side, {})
        if side_metrics.get("drift_true_positive", 0) == 0 and side_metrics.get("drift_recall", 0) > 0:
            warnings.append(f"{side}: drift_recall > 0 but no drift_true_positive")
        if side_metrics.get("judge_fallback_count", 0) > side_metrics.get("judge_interventions_scored", 0):
            warnings.append(f"{side}: judge_fallback_count exceeds interventions scored")

    checks = proof.get("consistency_checks", {})
    for side in ("baseline", "trained"):
        for warning in checks.get(side, []):
            if "detected=" in warning and "fired=0" in warning:
                warnings.append(f"{side}: {warning}")

    return warnings


def main():
    parser = argparse.ArgumentParser(description="Validate evidence artifacts.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when warnings are found.",
    )
    args = parser.parse_args()
    root = Path(__file__).parent.parent
    warnings = validate(root)
    if warnings:
        print("Evidence validation warnings:")
        for warning in warnings:
            print(f"- {warning}")
        if args.strict:
            raise SystemExit(1)
        return
    print("Evidence artifacts look consistent.")


if __name__ == "__main__":
    main()
