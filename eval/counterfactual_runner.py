"""
eval/counterfactual_runner.py
==============================
Runs Era N+1 twice: once WITH the Legacy Document, once WITHOUT.
The performance delta is entirely attributable to the quality of
what the previous agent chose to write.
"""
import asyncio
import logging
from typing import Optional
from environment.openenv_wrapper import EpistemicOpsEnv
from environment.scenario_loader import ScenarioLoader

logger = logging.getLogger("counterfactual")

EMPTY_LEGACY_DOC = "No information available from the previous era."
NUM_RUNS = 5  # Run each condition 5 times to reduce variance


async def measure_legacy_utility(
    scenario_id: str,
    era_id: int,
    legacy_doc: str,
    agent
) -> dict:
    """
    Measure how much the legacy_doc improves Era N+1 performance.

    Returns:
        utility: float — performance delta (positive = doc helped)
        run_a_scores: list — scores WITH legacy doc
        run_b_scores: list — scores WITHOUT (control)
    """
    loader = ScenarioLoader()
    scenario = loader.get_scenario(scenario_id)
    if not scenario:
        raise ValueError(f"Scenario {scenario_id} not found")
    scenario_config = scenario.model_dump()

    run_a_scores = []  # WITH legacy doc
    run_b_scores = []  # WITHOUT (control)

    for run in range(NUM_RUNS):
        # Run A: With legacy doc
        score_a = await _run_era(scenario_config, era_id, legacy_doc, agent)
        run_a_scores.append(score_a)

        # Run B: Without legacy doc (control)
        score_b = await _run_era(scenario_config, era_id, EMPTY_LEGACY_DOC, agent)
        run_b_scores.append(score_b)

    mean_a = sum(run_a_scores) / len(run_a_scores)
    mean_b = sum(run_b_scores) / len(run_b_scores)
    utility = mean_a - mean_b

    logger.info(
        f"Legacy Utility [{scenario_id} era {era_id}]: "
        f"with_doc={mean_a:.3f}, without_doc={mean_b:.3f}, delta={utility:.3f}"
    )

    return {
        "utility": utility,
        "mean_with_doc": mean_a,
        "mean_without_doc": mean_b,
        "run_a_scores": run_a_scores,
        "run_b_scores": run_b_scores,
    }


async def _run_era(scenario_config: dict, era_id: int,
                   legacy_doc: str, agent) -> float:
    """Run one complete era and return normalised task score."""
    env = EpistemicOpsEnv()
    obs = env.reset(scenario_config, era_id=era_id, legacy_doc=legacy_doc)

    done = False
    step = 0
    while not done and step < 40:
        action = agent.generate_action(obs)
        obs, _, done, _ = await env.step("primary", action)
        step += 1

    # Score = fraction of success criteria met
    era_config = next(
        (e for e in scenario_config["eras"] if e["era_id"] == era_id), {}
    )
    criteria = era_config.get("success_criteria", [])
    met = 0
    if env.current_legacy_doc and "legacy_doc_written" in criteria:
        met += 1
    # Add other criteria checks here as they are implemented

    return met / max(1, len(criteria))
