"""
Generate judge-ready before/after evidence.

This script compares baseline vs trained behavior in the same environment.

Modes:
- profile mode (default): uses deterministic policy profiles from PrimaryAgent
- checkpoint mode: loads a local HF checkpoint for the trained side

Outputs:
- eval_results/proof_of_learning.json
- plots/proof_reward_curve.png
- plots/proof_before_vs_after.png
- eval_results/proof_behavior_examples.md
"""
from __future__ import annotations

import asyncio
import argparse
import json
import os
import re
import statistics
import sys
import platform
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_episode import run_full_episode
from agents.primary_agent import PrimaryAgent

DEFAULT_SCENARIOS = ["cascading_incident", "deployment_disaster", "invisible_outage"]
DEFAULT_ERAS_PER_RUN = 3
DEFAULT_RUNS_PER_SCENARIO = 3

# Deltas only for judge-facing rates (avoid confusing count diffs).
_DELTA_KEYS = (
    "avg_reward",
    "avg_criteria_completion",
    "drift_detection_rate",
    "incident_resolution_rate",
    "legacy_doc_rate",
    "drift_precision",
    "drift_recall",
    "judge_fallback_rate",
)


def _safe_mean(values):
    return statistics.fmean(values) if values else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return (numerator / denominator) if denominator else 0.0


def _get_repo_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _build_runtime_metadata(repo_root: Path) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "matplotlib_version": getattr(matplotlib, "__version__", "unknown"),
        "git_commit": _get_repo_commit(repo_root),
    }


class CheckpointPrimaryAgent:
    """Primary agent backed by a local transformers checkpoint."""

    def __init__(self, checkpoint_path: str):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            torch = __import__("torch")
        except ImportError as exc:
            raise RuntimeError(
                "checkpoint mode requires transformers and torch. "
                "Install them first or run profile mode."
            ) from exc

        self._torch = torch
        self._checkpoint_path = checkpoint_path
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()

    @staticmethod
    def _extract_json_blob(text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    def generate_action(self, observation: dict, conversation_history: list | None = None) -> dict:
        if observation.get("phase") == "AWAKENING":
            legacy_doc = observation.get("legacy_document", "")
            return {
                "action_type": "ready_to_operate",
                "payload": {"world_model_summary": f"Ready. Legacy doc available: {bool(legacy_doc)}"},
            }

        prompt = (
            f"{PrimaryAgent.SYSTEM_PROMPT}\n\n"
            "Current Observation:\n"
            f"{json.dumps(observation, indent=2)}\n\n"
            "Output ONLY a valid JSON action object."
        )

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = self._tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        action = self._extract_json_blob(generated)
        if not action or "action_type" not in action:
            return {"action_type": "write_reasoning", "payload": {"thought": "Invalid model JSON output; continuing safely."}}
        return action


def _aggregate(policy_results):
    rewards = []
    drift_detected = []
    incident_resolved = []
    legacy_written = []
    criteria_fraction = []
    drift_tp = 0
    drift_fp = 0
    drift_fn = 0
    drift_tn = 0
    judge_total = 0
    judge_fallback = 0

    for run in policy_results:
        episode = run["episode"]
        for era in episode.get("era_results", []):
            reward = era.get("reward", {})
            rewards.append(float(reward.get("R_normalized", 0.0)))
            has_drift = era.get("drifts_fired", 0) > 0
            declared_drift = era.get("drifts_detected", 0) > 0
            drift_detected.append(1 if declared_drift else 0)
            if has_drift and declared_drift:
                drift_tp += 1
            elif (not has_drift) and declared_drift:
                drift_fp += 1
            elif has_drift and (not declared_drift):
                drift_fn += 1
            else:
                drift_tn += 1
            incident_resolved.append(1 if "incident_resolved" in era.get("criteria_met", []) else 0)
            legacy_written.append(1 if era.get("legacy_doc_written") else 0)
            total_criteria = len(era.get("criteria_total", []))
            met_criteria = len(era.get("criteria_met", []))
            criteria_fraction.append((met_criteria / total_criteria) if total_criteria else 0.0)
            for step in era.get("trajectory", []):
                if step.get("agent") == "oversight":
                    judge_total += 1
                    if step.get("judge_fallback", False):
                        judge_fallback += 1

    return {
        "avg_reward": round(_safe_mean(rewards), 4),
        "avg_criteria_completion": round(_safe_mean(criteria_fraction), 4),
        "drift_detection_rate": round(_safe_mean(drift_detected), 4),
        "incident_resolution_rate": round(_safe_mean(incident_resolved), 4),
        "legacy_doc_rate": round(_safe_mean(legacy_written), 4),
        "drift_true_positive": drift_tp,
        "drift_false_positive": drift_fp,
        "drift_false_negative": drift_fn,
        "drift_true_negative": drift_tn,
        "drift_precision": round(_safe_div(drift_tp, drift_tp + drift_fp), 4),
        "drift_recall": round(_safe_div(drift_tp, drift_tp + drift_fn), 4),
        "judge_interventions_scored": judge_total,
        "judge_fallback_count": judge_fallback,
        "judge_fallback_rate": round(_safe_div(judge_fallback, judge_total), 4),
    }


def _extract_behavior_examples(baseline_episode, trained_episode):
    def summarize_era_actions(era):
        lines = []
        for item in era.get("trajectory", [])[:12]:
            action_type = item.get("action", {}).get("action_type", "unknown")
            payload = item.get("action", {}).get("payload", {})
            payload_text = json.dumps(payload)
            short_payload = payload_text[:110] + ("..." if len(payload_text) > 110 else "")
            lines.append(f"- step {item.get('step', '?')}: {action_type} | {short_payload}")
        return "\n".join(lines)

    base_era = next((e for e in baseline_episode.get("era_results", []) if e.get("drifts_fired", 0) > 0), None)
    trained_era = next((e for e in trained_episode.get("era_results", []) if e.get("drifts_fired", 0) > 0), None)
    base_era = base_era or (baseline_episode.get("era_results", [{}])[0])
    trained_era = trained_era or (trained_episode.get("era_results", [{}])[0])

    before_text = summarize_era_actions(base_era)
    after_text = summarize_era_actions(trained_era)
    return before_text, after_text


def _run_consistency_checks(policy_results):
    warnings = []
    for run in policy_results:
        episode = run.get("episode", {})
        scenario_id = run.get("scenario_id", "unknown")
        run_idx = run.get("run_idx", "?")
        for era in episode.get("era_results", []):
            fired = int(era.get("drifts_fired", 0))
            detected = int(era.get("drifts_detected", 0))
            if fired == 0 and detected > 0:
                warnings.append(
                    f"{scenario_id} run {run_idx} era {era.get('era_id', '?')}: detected={detected} while fired=0"
                )
            if fired > 0 and detected == 0:
                warnings.append(
                    f"{scenario_id} run {run_idx} era {era.get('era_id', '?')}: fired={fired} but no drift hypothesis declared"
                )
    return warnings


def _plot_reward_curve(results, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#101426")
    ax.set_facecolor("#171c31")

    for policy, color in [("baseline", "#9aa3b2"), ("trained", "#4dd4ac")]:
        by_episode = []
        for run in results[policy]:
            episode = run["episode"]
            by_episode.append(float(episode.get("avg_normalized_reward", 0.0)))
        ax.plot(range(1, len(by_episode) + 1), by_episode, marker="o", label=policy.title(), color=color, linewidth=2)

    ax.set_title("Average Episode Reward (Before vs After)", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode Index", color="white")
    ax.set_ylabel("Avg Normalized Reward", color="white")
    ax.tick_params(colors="white")
    ax.grid(alpha=0.2, color="white")
    ax.legend(facecolor="#101426", edgecolor="#444", labelcolor="white")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="#101426")
    plt.close(fig)


def _plot_metric_bars(summary, output_path: Path):
    metrics = [
        ("avg_reward", "Avg Reward"),
        ("avg_criteria_completion", "Criteria"),
        ("drift_detection_rate", "Drift Detect"),
        ("incident_resolution_rate", "Incident Resolve"),
        ("legacy_doc_rate", "Legacy Doc"),
    ]
    baseline_vals = [summary["baseline"][key] for key, _ in metrics]
    trained_vals = [summary["trained"][key] for key, _ in metrics]
    labels = [label for _, label in metrics]

    x = list(range(len(metrics)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#101426")
    ax.set_facecolor("#171c31")
    ax.bar([i - width / 2 for i in x], baseline_vals, width=width, label="Baseline", color="#9aa3b2")
    ax.bar([i + width / 2 for i in x], trained_vals, width=width, label="Trained", color="#4dd4ac")

    for i, (b, t) in enumerate(zip(baseline_vals, trained_vals)):
        ax.text(i - width / 2, b + 0.01, f"{b:.2f}", ha="center", color="white", fontsize=9)
        ax.text(i + width / 2, t + 0.01, f"{t:.2f}", ha="center", color="white", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white")
    ax.set_ylim(0, max(max(baseline_vals), max(trained_vals)) + 0.2)
    ax.set_title("Before vs After Metrics", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.grid(axis="y", alpha=0.2, color="white")
    ax.legend(facecolor="#101426", edgecolor="#444", labelcolor="white")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="#101426")
    plt.close(fig)


async def _run_policy(policy_label: str, scenarios, runs_per_scenario: int, eras_per_run: int, *, agent_mode: str, checkpoint_path: str | None = None):
    checkpoint_agent = None
    if agent_mode == "checkpoint":
        checkpoint_agent = CheckpointPrimaryAgent(checkpoint_path)

    runs = []
    for scenario_id in scenarios:
        for run_idx in range(runs_per_scenario):
            record_name = f"episodes/{policy_label}_{scenario_id}_run{run_idx + 1}.json"
            if agent_mode == "checkpoint":
                episode = await _run_with_external_agent(
                    scenario_id=scenario_id,
                    num_eras=eras_per_run,
                    record_path=record_name,
                    primary_agent=checkpoint_agent,
                )
            else:
                episode = await run_full_episode(
                    scenario_id=scenario_id,
                    num_eras=eras_per_run,
                    record_path=record_name,
                    primary_profile=policy_label,
                    primary_use_llm=False,
                )
            runs.append(
                {
                    "scenario_id": scenario_id,
                    "run_idx": run_idx + 1,
                    "episode": episode,
                    "record_path": record_name,
                }
            )
    return runs


async def _run_with_external_agent(scenario_id: str, num_eras: int, record_path: str, primary_agent):
    """Run episode by reusing run_episode internals with an externally provided agent."""
    from environment.openenv_wrapper import EpistemicOpsEnv
    from environment.scenario_loader import ScenarioLoader
    from agents.oversight_agent import OversightAgent
    from agents.llm_judge import LLMJudge
    from run_episode import run_era
    from datetime import datetime, timezone

    loader = ScenarioLoader()
    scenario = loader.get_scenario(scenario_id)
    if not scenario:
        raise ValueError(f"Scenario '{scenario_id}' not found")

    scenario_config = scenario.model_dump()
    oversight = OversightAgent()
    judge = LLMJudge()
    env = EpistemicOpsEnv()

    legacy_doc = None
    results = []
    for era_id in range(1, num_eras + 1):
        era_config = next((e for e in scenario_config.get("eras", []) if e.get("era_id") == era_id), {})
        era_result = await run_era(
            env,
            scenario_config,
            era_config,
            era_id,
            primary_agent,
            oversight,
            judge,
            legacy_doc=legacy_doc,
            max_steps=era_config.get("max_steps", 40),
        )
        results.append(era_result)
        legacy_doc = era_result.get("legacy_doc")

    avg = _safe_mean([r.get("reward", {}).get("R_normalized", 0.0) for r in results])
    episode = {
        "scenario_id": scenario_id,
        "scenario_name": scenario.name,
        "num_eras": num_eras,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "avg_normalized_reward": round(avg, 4),
        "era_results": results,
    }

    path = Path(record_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(episode, f, indent=2)
    return episode


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate before/after proof artifacts.")
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS), help="Comma-separated scenario ids")
    parser.add_argument("--runs-per-scenario", type=int, default=DEFAULT_RUNS_PER_SCENARIO)
    parser.add_argument("--eras-per-run", type=int, default=DEFAULT_ERAS_PER_RUN)
    parser.add_argument("--baseline-profile", default="baseline", choices=["baseline", "trained"])
    parser.add_argument("--trained-profile", default="trained", choices=["baseline", "trained"])
    parser.add_argument(
        "--trained-agent-source",
        default="profile",
        choices=["profile", "checkpoint"],
        help="Use profile policy or local checkpoint for trained side",
    )
    parser.add_argument(
        "--trained-checkpoint-path",
        default="",
        help="Path to a local HF checkpoint used when --trained-agent-source=checkpoint",
    )
    parser.add_argument(
        "--on-checkpoint-error",
        default="fallback_profile",
        choices=["fallback_profile", "fail"],
        help="Behavior when checkpoint mode cannot start (e.g., missing torch/checkpoint).",
    )
    parser.add_argument(
        "--proof-mode",
        default="demo",
        choices=["demo", "final"],
        help="demo allows fallback; final enforces checkpoint-backed evidence.",
    )
    parser.add_argument(
        "--require-checkpoint",
        action="store_true",
        help="Fail closed unless trained side runs from a checkpoint.",
    )
    return parser.parse_args()


async def main():
    args = _parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    repo_root = Path(__file__).parent.parent
    if not scenarios:
        raise ValueError("No scenarios provided.")
    if args.proof_mode == "final":
        args.require_checkpoint = True
    if args.trained_agent_source == "checkpoint" and not args.trained_checkpoint_path:
        raise ValueError("Provide --trained-checkpoint-path when using checkpoint mode.")
    if args.require_checkpoint and args.trained_agent_source != "checkpoint":
        raise ValueError("--require-checkpoint requires --trained-agent-source=checkpoint.")
    if args.require_checkpoint:
        args.on_checkpoint_error = "fail"

    print("Running baseline policy...")
    baseline_runs = await _run_policy(
        args.baseline_profile,
        scenarios,
        args.runs_per_scenario,
        args.eras_per_run,
        agent_mode="profile",
    )
    print("Running trained policy...")
    effective_trained_source = args.trained_agent_source
    effective_trained_label = args.trained_profile if args.trained_agent_source == "profile" else "trained_ckpt"
    effective_checkpoint_path = args.trained_checkpoint_path or None
    fallback_reason = None
    try:
        trained_runs = await _run_policy(
            effective_trained_label,
            scenarios,
            args.runs_per_scenario,
            args.eras_per_run,
            agent_mode=effective_trained_source,
            checkpoint_path=effective_checkpoint_path,
        )
    except Exception as exc:
        if args.trained_agent_source == "checkpoint" and args.on_checkpoint_error == "fallback_profile":
            fallback_reason = f"{type(exc).__name__}: {exc}"
            print(
                "[WARN] Checkpoint mode failed; falling back to profile mode. "
                f"Reason: {fallback_reason}"
            )
            effective_trained_source = "profile"
            effective_trained_label = args.trained_profile
            effective_checkpoint_path = None
            trained_runs = await _run_policy(
                effective_trained_label,
                scenarios,
                args.runs_per_scenario,
                args.eras_per_run,
                agent_mode="profile",
                checkpoint_path=None,
            )
        else:
            raise

    summary = {
        "baseline": _aggregate(baseline_runs),
        "trained": _aggregate(trained_runs),
    }
    consistency = {
        "baseline": _run_consistency_checks(baseline_runs),
        "trained": _run_consistency_checks(trained_runs),
    }

    # Use the same scenario (cascading_incident, run 1) for behavior snapshots.
    preferred = "cascading_incident" if "cascading_incident" in scenarios else scenarios[0]
    baseline_ref = next(r for r in baseline_runs if r["scenario_id"] == preferred)
    trained_ref = next(r for r in trained_runs if r["scenario_id"] == preferred)
    before_actions, after_actions = _extract_behavior_examples(
        baseline_ref["episode"], trained_ref["episode"]
    )

    output = {
        "config": {
            "scenarios": scenarios,
            "runs_per_scenario": args.runs_per_scenario,
            "eras_per_run": args.eras_per_run,
            "baseline_profile": args.baseline_profile,
            "trained_profile": args.trained_profile,
            "trained_agent_source": effective_trained_source,
            "trained_checkpoint_path": effective_checkpoint_path,
            "mode": "profile_vs_profile" if effective_trained_source == "profile" else "profile_vs_checkpoint",
            "offline_mode": os.getenv("EPISTEMICOPS_OFFLINE", "").lower() == "true",
            "checkpoint_fallback_used": fallback_reason is not None,
            "checkpoint_fallback_reason": fallback_reason,
            "on_checkpoint_error": args.on_checkpoint_error,
            "proof_mode": args.proof_mode,
            "require_checkpoint": args.require_checkpoint,
        },
        "summary": summary,
        "deltas": {
            key: round(summary["trained"][key] - summary["baseline"][key], 4)
            for key in _DELTA_KEYS
            if key in summary["baseline"] and key in summary["trained"]
        },
        "consistency_checks": consistency,
        "examples": {
            "before": before_actions,
            "after": after_actions,
            "baseline_episode": baseline_ref["record_path"],
            "trained_episode": trained_ref["record_path"],
        },
    }

    runtime_meta = _build_runtime_metadata(repo_root)
    run_metadata = {
        "runtime": runtime_meta,
        "requested_config": {
            "scenarios": scenarios,
            "runs_per_scenario": args.runs_per_scenario,
            "eras_per_run": args.eras_per_run,
            "baseline_profile": args.baseline_profile,
            "trained_profile": args.trained_profile,
            "trained_agent_source": args.trained_agent_source,
            "trained_checkpoint_path": args.trained_checkpoint_path or None,
            "proof_mode": args.proof_mode,
            "require_checkpoint": args.require_checkpoint,
            "on_checkpoint_error": args.on_checkpoint_error,
        },
        "effective_config": output["config"],
        "artifact_paths": {
            "proof_json": "eval_results/proof_of_learning.json",
            "metadata_json": "eval_results/proof_run_metadata.json",
            "behavior_examples": "eval_results/proof_behavior_examples.md",
            "reward_curve": "plots/proof_reward_curve.png",
            "metric_chart": "plots/proof_before_vs_after.png",
        },
        "warnings": consistency["baseline"] + consistency["trained"],
    }

    eval_dir = repo_root / "eval_results"
    plots_dir = repo_root / "plots"
    eval_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(eval_dir / "proof_of_learning.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    with open(eval_dir / "proof_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    with open(eval_dir / "proof_behavior_examples.md", "w", encoding="utf-8") as f:
        f.write("# Behavioral Difference: Before vs After\n\n")
        f.write("## Before (Baseline)\n")
        f.write(before_actions + "\n\n")
        f.write("## After (Trained)\n")
        f.write(after_actions + "\n")

    _plot_reward_curve({"baseline": baseline_runs, "trained": trained_runs}, plots_dir / "proof_reward_curve.png")
    _plot_metric_bars(summary, plots_dir / "proof_before_vs_after.png")

    print("Saved:")
    print(f"  - {eval_dir / 'proof_of_learning.json'}")
    print(f"  - {eval_dir / 'proof_run_metadata.json'}")
    print(f"  - {eval_dir / 'proof_behavior_examples.md'}")
    print(f"  - {plots_dir / 'proof_reward_curve.png'}")
    print(f"  - {plots_dir / 'proof_before_vs_after.png'}")
    print("\nSummary:")
    print(json.dumps(output["summary"], indent=2))
    if run_metadata["warnings"]:
        print("\nConsistency warnings:")
        for warning in run_metadata["warnings"]:
            print(f"  - {warning}")


if __name__ == "__main__":
    asyncio.run(main())
