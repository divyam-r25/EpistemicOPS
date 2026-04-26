"""
baseline_eval.py -- Runs all scenarios and records real reward values.
"""
import asyncio
import json
import sys
import random
import argparse
import platform
import os
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_episode import run_full_episode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("baseline-eval")

SCENARIOS = ["cascading_incident", "deployment_disaster", "invisible_outage"]
NUM_RUNS = 3  # Runs per scenario for variance
ERAS_PER_RUN = 3


def _stable_scenario_seed(scenario_id: str) -> int:
    return sum(ord(ch) for ch in scenario_id) % 1000


async def run_baseline_evaluation(scenarios=None, num_runs: int = NUM_RUNS, eras_per_run: int = ERAS_PER_RUN):
    scenarios = scenarios or SCENARIOS
    results = {}
    
    for scenario_id in scenarios:
        logger.info(f"\n{'='*50}")
        logger.info(f"EVALUATING SCENARIO: {scenario_id}")
        logger.info(f"{'='*50}")
        
        scenario_results = []
        for run_idx in range(num_runs):
            logger.info(f"  Run {run_idx + 1}/{num_runs}")
            
            random.seed(42 + run_idx * 17 + _stable_scenario_seed(scenario_id))
            
            try:
                episode = await run_full_episode(
                    scenario_id=scenario_id,
                    num_eras=eras_per_run,
                    record_path=None,
                    primary_profile="baseline",
                    primary_use_llm=False,
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
                        "drifts_fired": era_result.get("drifts_fired", 0),
                        "drifts_detected": era_result.get("drifts_detected", 0),
                        "oversight_interventions": era_result.get("oversight_interventions", 0),
                    })
            except Exception as e:
                logger.error(f"  Run {run_idx + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                scenario_results.append({"run": run_idx + 1, "error": str(e)})
        
        results[scenario_id] = scenario_results
    
    # Save results
    eval_dir = Path(__file__).parent.parent / "eval_results"
    output_path = eval_dir / "baseline_results.json"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    metadata = {
        "scenarios": scenarios,
        "num_runs": num_runs,
        "eras_per_run": eras_per_run,
        "offline_mode": os.getenv("EPISTEMICOPS_OFFLINE", "").lower() == "true",
        "python_version": platform.python_version(),
    }
    with open(eval_dir / "baseline_eval_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
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
            avg_drifts = sum(r.get("drifts_fired", 0) for r in valid_runs) / len(valid_runs)
            avg_oversight = sum(r.get("oversight_interventions", 0) for r in valid_runs) / len(valid_runs)
            print(f"  {scenario_id}:")
            print(f"    Avg R_total: {avg_total:.4f}  |  Avg R_norm: {avg_norm:.4f}  |  Avg R_task: {avg_task:.4f}")
            print(f"    Avg Drifts: {avg_drifts:.1f}  |  Avg Oversight: {avg_oversight:.1f}")
            print(f"    Runs: {len(valid_runs)} valid / {len(runs)} total")
        else:
            print(f"  {scenario_id}: ALL RUNS FAILED")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline evaluation with explicit config.")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS), help="Comma-separated scenario ids")
    parser.add_argument("--runs-per-scenario", type=int, default=NUM_RUNS)
    parser.add_argument("--eras-per-run", type=int, default=ERAS_PER_RUN)
    cli_args = parser.parse_args()
    cli_scenarios = [s.strip() for s in cli_args.scenarios.split(",") if s.strip()]
    asyncio.run(
        run_baseline_evaluation(
            scenarios=cli_scenarios,
            num_runs=cli_args.runs_per_scenario,
            eras_per_run=cli_args.eras_per_run,
        )
    )
