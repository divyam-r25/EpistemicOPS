"""
eval/benchmark.py
=================
Runs the trained model on held-out Scenario 3 (invisible_outage) and
compares performance against the baseline. Produces the numbers you
show judges: drift detection rate, legacy utility, task completion.
"""
import json
import asyncio
import logging
from pathlib import Path
from environment.openenv_wrapper import EpistemicOpsEnv
from environment.scenario_loader import ScenarioLoader
from reward import compute_total_reward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

HELD_OUT_SCENARIO = "invisible_outage"
NUM_EPISODES = 5  # Run 5 episodes to average variance


async def run_episode(env: EpistemicOpsEnv, scenario_config: dict,
                      agent, era_id: int) -> dict:
    """Run one complete era and return metrics."""
    obs = env.reset(scenario_config, era_id=era_id)
    done = False
    step = 0
    drifts_detected = 0
    hypotheses_declared = []
    conversation_history = []

    while not done and step < 40:
        action = agent.generate_action(obs, conversation_history)

        # Track conversation history
        conversation_history.append({"role": "assistant", "action": action})

        # Track hypothesis declarations for drift detection metric
        if action.get("action_type") == "declare_hypothesis":
            hypotheses_declared.append(action["payload"])

        obs, reward, done, info = await env.step("primary", action)
        conversation_history.append({"role": "environment", "obs": obs})

        # Count correct drift detections
        if (action.get("action_type") == "declare_hypothesis" and
                "drift" in action["payload"].get("hypothesis", "").lower()):
            drifts_detected += 1

        step += 1

    return {
        "era_id": era_id,
        "steps_taken": step,
        "drifts_detected": drifts_detected,
        "total_drifts": len(env.world.state.drift_events_fired),
        "drift_detection_rate": drifts_detected / max(1, len(env.world.state.drift_events_fired)),
        "legacy_doc_written": env.current_legacy_doc is not None,
        "legacy_doc_length": len(env.current_legacy_doc or ""),
    }


async def run_benchmark(agent, label: str) -> dict:
    """Run full benchmark on held-out scenario."""
    loader = ScenarioLoader()
    scenario = loader.get_scenario(HELD_OUT_SCENARIO)
    if not scenario:
        raise ValueError(f"Scenario {HELD_OUT_SCENARIO} not found")

    scenario_config = scenario.model_dump()
    all_results = []

    for episode in range(NUM_EPISODES):
        logger.info(f"[{label}] Episode {episode + 1}/{NUM_EPISODES}")
        env = EpistemicOpsEnv()
        episode_results = []

        for era_id in range(1, scenario.num_eras + 1):
            result = await run_episode(env, scenario_config, agent, era_id)
            episode_results.append(result)

        all_results.append(episode_results)

    # Aggregate
    all_drift_rates = [r["drift_detection_rate"]
                       for ep in all_results for r in ep]
    all_legacy = [r["legacy_doc_written"]
                  for ep in all_results for r in ep]

    summary = {
        "label": label,
        "scenario": HELD_OUT_SCENARIO,
        "num_episodes": NUM_EPISODES,
        "avg_drift_detection_rate": sum(all_drift_rates) / len(all_drift_rates),
        "legacy_doc_completion_rate": sum(all_legacy) / len(all_legacy),
        "raw_results": all_results,
    }

    return summary


def save_results(results: dict, filename: str):
    output_dir = Path("./eval_results")
    output_dir.mkdir(exist_ok=True)
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {path}")


def print_comparison(baseline: dict, trained: dict):
    print("\n" + "="*60)
    print("BEFORE vs AFTER TRAINING")
    print("="*60)
    metrics = [
        ("Drift Detection Rate",
         baseline["avg_drift_detection_rate"],
         trained["avg_drift_detection_rate"]),
        ("Legacy Doc Completion",
         baseline["legacy_doc_completion_rate"],
         trained["legacy_doc_completion_rate"]),
    ]
    for name, before, after in metrics:
        delta = after - before
        sign = "+" if delta >= 0 else ""
        print(f"{name:30s}  {before:.1%} → {after:.1%}  ({sign}{delta:.1%})")
    print("="*60)


if __name__ == "__main__":
    # Usage: python eval/benchmark.py
    # Runs mock agents for now; replace with real model agents at onsite
    from agents.primary_agent import PrimaryAgent

    baseline_agent = PrimaryAgent()  # zero-shot (no fine-tuning)

    async def main():
        baseline = await run_benchmark(baseline_agent, "Baseline (Zero-Shot)")
        save_results(baseline, "benchmark_baseline.json")
        print_comparison(baseline, baseline)  # same for now; replace with trained

    asyncio.run(main())
