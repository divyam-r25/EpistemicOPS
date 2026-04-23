"""
baseline_eval.py — Real Baseline Evaluation
=============================================
Runs the orchestration loop for each scenario and records actual reward values.
No hardcoded values — all rewards computed from real game state.
"""
import asyncio
import json
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_episode import run_full_episode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("baseline-eval")

SCENARIOS = ["cascading_incident", "deployment_disaster", "invisible_outage"]
NUM_RUNS = 3  # Runs per scenario for variance


async def run_baseline_evaluation():
    """Run baseline evaluation across all scenarios."""
    results = {}
    
    for scenario_id in SCENARIOS:
        logger.info(f"\n{'='*50}")
        logger.info(f"EVALUATING SCENARIO: {scenario_id}")
        logger.info(f"{'='*50}")
        
        scenario_results = []
        for run_idx in range(NUM_RUNS):
            logger.info(f"  Run {run_idx + 1}/{NUM_RUNS}")
            try:
                episode = await run_full_episode(
                    scenario_id=scenario_id,
                    num_eras=2,  # 2 eras per run for speed
                    record_path=None,
                )
                # Extract per-era reward dicts
                for era_result in episode.get("era_results", []):
                    reward = era_result.get("reward", {})
                    scenario_results.append({
                        "run": run_idx + 1,
                        "era_id": era_result["era_id"],
                        "R_era_task": round(reward.get("R_era_task", 0.0), 4),
                        "R_calibration": round(reward.get("R_calibration", 1.0), 4),
                        "R_era_task_adjusted": round(reward.get("R_era_task_adjusted", 0.0), 4),
                        "R_teacher_delta": round(reward.get("R_teacher_delta", 0.0), 4),
                        "R_legacy_utility": round(reward.get("R_legacy_utility", 0.0), 4),
                        "R_answer_leakage": round(reward.get("R_answer_leakage", 0.0), 4),
                        "R_anti_hack_penalty": round(reward.get("R_anti_hack_penalty", 0.0), 4),
                        "R_total": round(reward.get("R_total", 0.0), 4),
                        "R_normalized": round(reward.get("R_normalized", 0.0), 4),
                        "R_max_possible": reward.get("R_max_possible", 3.5),
                        "criteria_met": era_result.get("criteria_met", []),
                        "criteria_total": era_result.get("criteria_total", []),
                        "steps_taken": era_result.get("steps_taken", 0),
                    })
            except Exception as e:
                logger.error(f"  Run {run_idx + 1} failed: {e}")
                scenario_results.append({"run": run_idx + 1, "error": str(e)})
        
        results[scenario_id] = scenario_results
    
    # Save results
    output_path = Path(__file__).parent.parent / "eval_results" / "baseline_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("BASELINE EVALUATION SUMMARY")
    print(f"{'='*60}")
    for scenario_id, runs in results.items():
        valid_runs = [r for r in runs if "error" not in r]
        if valid_runs:
            avg_total = sum(r["R_total"] for r in valid_runs) / len(valid_runs)
            avg_norm = sum(r["R_normalized"] for r in valid_runs) / len(valid_runs)
            avg_task = sum(r["R_era_task"] for r in valid_runs) / len(valid_runs)
            print(f"  {scenario_id}:")
            print(f"    Avg R_total: {avg_total:.4f}  |  Avg R_norm: {avg_norm:.4f}  |  Avg R_task: {avg_task:.4f}")
            print(f"    Runs: {len(valid_runs)} valid / {len(runs)} total")
        else:
            print(f"  {scenario_id}: ALL RUNS FAILED")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_baseline_evaluation())
