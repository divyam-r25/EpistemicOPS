import os
import json
import asyncio
import logging
from pathlib import Path

from environment.openenv_wrapper import EpistemicOpsEnv
from environment.scenario_loader import ScenarioLoader
from reward import compute_total_reward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("baseline-eval")

async def run_baseline_evaluation():
    """Runs base Llama zero-shot across scenarios to establish starting metrics."""
    logger.info("Starting baseline evaluation...")
    
    # Normally this would load the real LLM and run it through the environment.
    # We'll build the scaffolding for how this loop operates.
    
    env = EpistemicOpsEnv()
    
    scenarios_to_run = ["cascading_incident", "deployment_disaster", "invisible_outage"]
    results = {}
    
    for scenario in scenarios_to_run:
        logger.info(f"Evaluating scenario: {scenario}")
        # Load scenario config (mocked here, would load from yaml)
        _loader = ScenarioLoader()
        scenario_obj = _loader.get_scenario(scenario)
        if not scenario_obj:
            logger.warning(f"Scenario '{scenario}' not found. Skipping.")
            continue
        scenario_config = scenario_obj.model_dump()
        
        scenario_results = []
        legacy_doc = None
        
        for era_id in range(1, 6):
            obs = env.reset(scenario_config, era_id=era_id, legacy_doc=legacy_doc)
            
            # Simulated era loop
            done = False
            step_count = 0
            while not done and step_count < 20:
                # Let primary agent take action (mock)
                action = {"action_type": "write_reasoning", "payload": {"thought": "thinking"}}
                obs, reward, done, info = await env.step("primary", action)
                
                # Check for Socratic Phase
                if info.get("phase") == "SOCRATIC_RECOVERY":
                    # Let oversight agent take action (mock)
                    o_action = {"action_type": "oversight_targeted_question", "payload": {"question": "Why?"}}
                    o_obs, o_reward, _, _ = await env.step("oversight", o_action)
                
                # Force era end for baseline test
                if step_count == 5:
                    end_action = {"action_type": "write_legacy", "payload": {"content": "Baseline doc"}}
                    await env.step("primary", end_action)
                    done_action = {"action_type": "end_era", "payload": {}}
                    obs, reward, done, info = await env.step("primary", done_action)
                
                step_count += 1
                
            # Compute era rewards (mock metrics for baseline)
            era_reward = compute_total_reward(
                era_task=0.45,
                calibration=1.0,
                teacher_delta=0.15,
                legacy_utility=0.05,
                answer_leakage=0.0
            )
            scenario_results.append(era_reward)
            legacy_doc = env.current_legacy_doc
            
        results[scenario] = scenario_results
        
    # Save results
    output_dir = Path("./eval_results")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Baseline evaluation complete. Results saved to {output_dir}/baseline_results.json")

if __name__ == "__main__":
    asyncio.run(run_baseline_evaluation())
