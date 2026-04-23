"""
run_episode.py — Full Episode Orchestrator
============================================
THE critical missing piece: wires together environment, agents, reward,
drift injection, oversight, and LLM judge into a complete episode loop.

Usage:
    python run_episode.py --scenario cascading_incident
    python run_episode.py --scenario cascading_incident --eras 2
    python run_episode.py --scenario cascading_incident --record episodes/run1.json
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from environment.openenv_wrapper import EpistemicOpsEnv
from environment.scenario_loader import ScenarioLoader
from agents.primary_agent import PrimaryAgent
from agents.oversight_agent import OversightAgent
from agents.llm_judge import LLMJudge
from reward import compute_total_reward
from reward.era_task_reward import compute_era_task_reward
from reward.calibration_reward import compute_calibration_reward
from reward.teacher_delta_reward import compute_teacher_delta_reward
from reward.legacy_utility_reward import compute_legacy_utility_reward
from reward.leakage_penalty import compute_leakage_penalty
from reward.anti_hack_penalty import compute_anti_hack_penalty

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("orchestrator")


async def run_era(
    env: EpistemicOpsEnv,
    scenario_config: dict,
    era_config: dict,
    era_id: int,
    primary: PrimaryAgent,
    oversight: OversightAgent,
    judge: LLMJudge,
    legacy_doc: str = None,
    max_steps: int = 40,
) -> dict:
    """Run a single era and return detailed results."""
    logger.info(f"═══ ERA {era_id} START ═══")
    logger.info(f"Task: {era_config.get('task_brief', 'N/A')}")

    obs = env.reset(scenario_config, era_id=era_id, legacy_doc=legacy_doc)
    conversation_history = []
    done = False
    step = 0
    score_before_oversight = 0.0
    score_after_oversight = 0.0
    oversight_triggered = False
    era_trajectory = []  # For recording

    while not done and step < max_steps:
        # ── 1. Primary Agent acts ──────────────────────────────────────
        action = primary.generate_action(obs, conversation_history)
        action_type = action.get("action_type", "unknown")
        logger.info(f"  Step {step:2d} │ Primary: {action_type}")

        # Record conversation
        conversation_history.append({"role": "assistant", "action": action})

        # Execute
        obs, reward, done, info = await env.step("primary", action)
        conversation_history.append({"role": "environment", "obs": obs})

        # Record trajectory step
        era_trajectory.append({
            "step": step,
            "agent": "primary",
            "action": action,
            "phase": info.get("phase", ""),
            "reward": reward,
            "done": done,
        })

        # ── 2. Check if we entered SOCRATIC_RECOVERY ───────────────────
        if info.get("phase") == "SOCRATIC_RECOVERY" and not oversight_triggered:
            oversight_triggered = True
            # Snapshot score before oversight for teacher_delta
            criteria = era_config.get("success_criteria", [])
            met_before = env.world.evaluate_success_criteria(criteria)
            score_before_oversight = len(met_before) / max(1, len(criteria))
            logger.info(f"  Step {step:2d} │ ⚡ SOCRATIC_RECOVERY triggered (score_before={score_before_oversight:.2f})")

            # Generate oversight intervention
            drift_events = env.world.state.drift_events_fired
            drift_config = drift_events[-1] if drift_events else {}
            if hasattr(drift_config, 'model_dump'):
                drift_config = drift_config.model_dump()

            intervention = oversight.generate_intervention(
                env.primary_reasoning_trace,
                drift_config,
                env.oversight_interventions
            )
            logger.info(f"  Step {step:2d} │ Oversight: {intervention.get('action_type', 'N/A')}")

            # Execute oversight action
            o_obs, o_reward, _, o_info = await env.step("oversight", intervention)
            conversation_history.append({
                "role": "oversight",
                "msg": intervention.get("payload", {}).get(
                    list(intervention.get("payload", {}).keys())[0] if intervention.get("payload") else "question",
                    ""
                )
            })

            era_trajectory.append({
                "step": step,
                "agent": "oversight",
                "action": intervention,
                "phase": o_info.get("phase", ""),
            })

            # Judge the intervention
            judge_result = await judge.evaluate_intervention(
                drift_config,
                env.primary_reasoning_trace,
                json.dumps(intervention)
            )
            logger.info(f"  Step {step:2d} │ Judge: targeting={judge_result.get('targeting', 0):.2f}, restraint={judge_result.get('restraint', 0):.2f}")

        # Auto end-era if agent wrote legacy and hasn't ended
        if action_type == "write_legacy" and step >= max_steps - 2:
            end_action = {"action_type": "end_era", "payload": {}}
            obs, reward, done, info = await env.step("primary", end_action)
            era_trajectory.append({"step": step, "agent": "primary", "action": end_action, "done": done})

        step += 1

    # ── Force legacy doc + end_era if agent didn't ─────────────────────
    if not done and not env.current_legacy_doc:
        legacy_action = {"action_type": "write_legacy", "payload": {"content": primary._generate_mock_legacy_doc(obs) if hasattr(primary, '_generate_mock_legacy_doc') else "Baseline legacy document."}}
        await env.step("primary", legacy_action)
        end_action = {"action_type": "end_era", "payload": {}}
        await env.step("primary", end_action)
    elif not done and env.current_legacy_doc:
        end_action = {"action_type": "end_era", "payload": {}}
        await env.step("primary", end_action)

    # ── 3. Compute Real Rewards ────────────────────────────────────────
    criteria = era_config.get("success_criteria", [])
    met_criteria = env.world.evaluate_success_criteria(criteria)

    r_era_task = compute_era_task_reward(met_criteria, criteria)
    r_calibration = compute_calibration_reward(env.world.state.hypotheses_declared)
    
    # Teacher delta
    if oversight_triggered:
        met_after = env.world.evaluate_success_criteria(criteria)
        score_after_oversight = len(met_after) / max(1, len(criteria))
    num_interventions = len(env.oversight_interventions)
    r_teacher_delta = compute_teacher_delta_reward(
        score_before_oversight, score_after_oversight, num_interventions
    )

    # Legacy utility (simplified — full counterfactual is run separately)
    r_legacy_utility = 0.0
    if env.current_legacy_doc:
        doc = env.current_legacy_doc
        drift_capture = env.parser.score_drift_capture(
            doc,
            [d if isinstance(d, dict) else d.model_dump() if hasattr(d, 'model_dump') else {} 
             for d in env.world.state.drift_events_fired]
        )
        undocumented = len(env.world.state.drift_events_fired) - int(drift_capture * len(env.world.state.drift_events_fired))
        r_legacy_utility = compute_legacy_utility_reward(
            performance_with_legacy=r_era_task,
            performance_without_legacy=max(0.0, r_era_task - 0.15),
            trust_ratings_accurate=drift_capture > 0.5,
            undocumented_drifts=undocumented
        )

    # Leakage penalty
    max_leakage = max((i.get("leakage", 0.0) for i in env.oversight_interventions), default=0.0)
    r_leakage = compute_leakage_penalty(max_leakage)

    # Anti-hack
    r_anti_hack = compute_anti_hack_penalty(env.action_history, max_steps)

    # Total
    total = compute_total_reward(
        era_task=r_era_task,
        calibration=r_calibration,
        teacher_delta=r_teacher_delta,
        legacy_utility=r_legacy_utility,
        answer_leakage=r_leakage,
        anti_hack_penalty=r_anti_hack,
    )

    logger.info(f"═══ ERA {era_id} RESULTS ═══")
    logger.info(f"  Criteria met: {met_criteria} / {criteria}")
    logger.info(f"  R_era_task={r_era_task:.3f}  R_calibration={r_calibration:.2f}x  R_teacher_delta={r_teacher_delta:.3f}")
    logger.info(f"  R_legacy_utility={r_legacy_utility:.3f}  R_leakage={r_leakage:.3f}  R_anti_hack={r_anti_hack:.3f}")
    logger.info(f"  ★ R_total={total['R_total']:.3f}  R_normalized={total['R_normalized']:.4f}")

    return {
        "era_id": era_id,
        "steps_taken": step,
        "criteria_met": met_criteria,
        "criteria_total": criteria,
        "drifts_fired": len(env.world.state.drift_events_fired),
        "drifts_detected": sum(1 for h in env.world.state.hypotheses_declared if "drift" in str(h).lower()),
        "legacy_doc_written": env.current_legacy_doc is not None,
        "legacy_doc": env.current_legacy_doc,
        "oversight_interventions": num_interventions,
        "reward": total,
        "trajectory": era_trajectory,
    }


async def run_full_episode(
    scenario_id: str,
    num_eras: int = None,
    record_path: str = None,
) -> dict:
    """Run a complete multi-era episode."""
    loader = ScenarioLoader()
    scenario = loader.get_scenario(scenario_id)
    if not scenario:
        raise ValueError(f"Scenario '{scenario_id}' not found. Available: {[s.id for s in loader.get_all_scenarios()]}")

    scenario_config = scenario.model_dump()
    if num_eras is None:
        num_eras = scenario.num_eras

    primary = PrimaryAgent()
    oversight = OversightAgent()
    judge = LLMJudge()
    env = EpistemicOpsEnv()

    legacy_doc = None
    episode_results = []

    logger.info(f"╔══════════════════════════════════════════╗")
    logger.info(f"║  EPISODE: {scenario.name}")
    logger.info(f"║  Eras: {num_eras}  │  ID: {scenario_id}")
    logger.info(f"╚══════════════════════════════════════════╝")

    for era_id in range(1, num_eras + 1):
        era_config = next(
            (e for e in scenario_config.get("eras", []) if e.get("era_id") == era_id),
            {"era_id": era_id, "task_brief": "", "success_criteria": [], "drift_window": {"earliest_step": 10, "latest_step": 20}, "max_steps": 40}
        )
        max_steps = era_config.get("max_steps", 40)

        era_result = await run_era(
            env, scenario_config, era_config, era_id,
            primary, oversight, judge,
            legacy_doc=legacy_doc,
            max_steps=max_steps,
        )
        episode_results.append(era_result)
        legacy_doc = era_result.get("legacy_doc")

    # ── Episode summary ───────────────────────────────────────────────
    total_rewards = [r["reward"]["R_normalized"] for r in episode_results]
    avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0

    episode = {
        "scenario_id": scenario_id,
        "scenario_name": scenario.name,
        "num_eras": num_eras,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "avg_normalized_reward": round(avg_reward, 4),
        "era_results": episode_results,
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"EPISODE COMPLETE — Avg Normalized Reward: {avg_reward:.4f}")
    for r in episode_results:
        logger.info(f"  Era {r['era_id']}: R_norm={r['reward']['R_normalized']:.4f}, criteria={len(r['criteria_met'])}/{len(r['criteria_total'])}")
    logger.info(f"{'='*50}")

    # Save recording if requested
    if record_path:
        path = Path(record_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Strip trajectory from saved results to keep file size reasonable
        save_data = {**episode}
        for era in save_data["era_results"]:
            era.pop("legacy_doc", None)  # Can be long
        with open(path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        logger.info(f"Episode recorded to {path}")

    return episode


def main():
    parser = argparse.ArgumentParser(description="EpistemicOps Episode Runner")
    parser.add_argument("--scenario", default="cascading_incident", help="Scenario ID")
    parser.add_argument("--eras", type=int, default=None, help="Number of eras (default: all)")
    parser.add_argument("--record", default=None, help="Path to save episode recording JSON")
    args = parser.parse_args()

    result = asyncio.run(run_full_episode(
        scenario_id=args.scenario,
        num_eras=args.eras,
        record_path=args.record,
    ))

    print(f"\n[OK] Episode finished. Average reward: {result['avg_normalized_reward']:.4f}")


if __name__ == "__main__":
    main()
