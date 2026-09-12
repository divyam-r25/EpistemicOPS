"""
train_primary.py -- GRPO training for the Primary Agent.
Uses HuggingFace TRL + Unsloth (4-bit). Run in Colab with T4 GPU.

Reward: reward/grpo_reward.py  (single canonical source — never redefine inline)
Dataset: multi-scenario, multi-era, drift-aware prompt diversity
"""
import os
import sys
import json
import random
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from unsloth import FastLanguageModel
    from trl import GRPOTrainer, GRPOConfig
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    print("Unsloth/TRL not installed. Run in Colab with GPU.")

try:
    from datasets import Dataset
except ImportError:
    Dataset = None

from environment.openenv_wrapper import EpistemicOpsEnv
from environment.scenario_loader import ScenarioLoader
from environment.action_validator import ActionValidator

# ── CANONICAL REWARD IMPORT ──────────────────────────────────────────────────
# This is the ONLY place the reward function should be defined for training.
# The Colab notebook MUST import from here — never redefine inline.
from reward.grpo_reward import compute_grpo_reward, compute_grpo_reward_with_components

try:
    from training.curriculum import CurriculumScheduler
except ImportError:
    CurriculumScheduler = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train-primary")


def _training_report_to() -> str:
    """TRL/HF Hub integration: wandb, tensorboard, or none."""
    if os.getenv("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        return os.getenv("TRAIN_FALLBACK_REPORT_TO", "none")
    return os.getenv("TRAIN_REPORT_TO", "wandb")


_scenario_loader = ScenarioLoader()
_validator = ActionValidator()
_curriculum = CurriculumScheduler() if CurriculumScheduler else None


# ── TRAINING REWARD ──────────────────────────────────────────────────────────
# epistemicops_reward_function is the callable passed to GRPOTrainer.
# It delegates to grpo_reward.py — do NOT implement reward logic here.

def epistemicops_reward_function(completions, prompts=None, **kwargs):
    """
    GRPO reward callable. Delegates to reward/grpo_reward.py.

    IMPORTANT: This function must NOT redefine reward logic.
    Any reward changes must be made in reward/grpo_reward.py.
    """
    return compute_grpo_reward(completions, prompts=prompts, **kwargs)


def epistemicops_reward_function_with_logging(completions, prompts=None, **kwargs):
    """
    Extended version that logs component breakdown for W&B / debugging.
    Use this during evaluation or when EPISTEMICOPS_LOG_REWARD_COMPONENTS=true.
    """
    component_dicts = compute_grpo_reward_with_components(completions, prompts=prompts)
    rewards = [d["R_total"] for d in component_dicts]

    # Log to W&B if available
    if os.getenv("EPISTEMICOPS_LOG_REWARD_COMPONENTS", "").lower() in ("1", "true"):
        try:
            import wandb
            if wandb.run is not None:
                avg = {
                    "reward/total": sum(rewards) / len(rewards),
                    "reward/format": sum(d["R_format"] for d in component_dicts) / len(component_dicts),
                    "reward/action_quality": sum(d["R_action_quality"] for d in component_dicts) / len(component_dicts),
                    "reward/drift_awareness": sum(d["R_drift_awareness"] for d in component_dicts) / len(component_dicts),
                    "reward/anti_hack": sum(d["R_anti_hack"] for d in component_dicts) / len(component_dicts),
                }
                wandb.log(avg)
        except ImportError:
            pass

    return rewards


# ── DATASET CONSTRUCTION ─────────────────────────────────────────────────────

# Scenarios used for TRAINING. held-out scenarios are NOT included here.
TRAINING_SCENARIOS = ["cascading_incident", "deployment_disaster", "invisible_outage"]

# Prompt templates for different context types
_SYSTEM_PREAMBLE = """\
You are the Primary Agent — an elite SRE engineer operating inside an enterprise incident management system.

CRITICAL RULES:
1. API contracts can and will change silently (Schema Drift). \
If a tool call fails or returns unexpected fields, assume the API drifted — test that hypothesis.
2. You may receive Socratic guidance from an Oversight Agent. They will NOT give you the answer.
3. At era end, write a Legacy Document (max 2048 tokens) for your successor.
4. Your context will be WIPED after this era. Only the Legacy Document survives.

Available actions (output ONLY a single valid JSON object, no markdown, no explanation):
- call_tool: {{"action_type": "call_tool", "payload": {{"tool": str, "args": dict}}}}
  Tools: get_incident_status, resolve_incident, get_metrics, rollback_deployment, query_logs, send_notification
- declare_hypothesis: {{"action_type": "declare_hypothesis", "payload": {{"hypothesis": str, "confidence": float}}}}
- write_reasoning: {{"action_type": "write_reasoning", "payload": {{"thought": str}}}}
- write_legacy: {{"action_type": "write_legacy", "payload": {{"content": str}}}}
  Sections REQUIRED: SECTION 1 (world state), SECTION 2 (trust ratings), SECTION 3 (drift events),
                     SECTION 4 (key decisions), SECTION 5 (open issues), SECTION 6 (next-era actions)
- declare_task_complete: {{"action_type": "declare_task_complete", "payload": {{"outcome": str, "summary": str}}}}
- end_era: {{"action_type": "end_era", "payload": {{}}}}
- ready_to_operate: {{"action_type": "ready_to_operate", "payload": {{"world_model_summary": str}}}}
"""

# Synthetic post-drift observation snippets to inject diversity
_DRIFT_OBS_SNIPPETS = [
    # Type change: status int -> string
    {
        "tool_response": {"status_code": 200, "body": {
            "incident_id": "INC-2041", "status": "INVESTIGATING", "severity": "P2"
        }},
        "drift_hint": "status field returned 'INVESTIGATING' (string) instead of expected integer",
    },
    # Field rename: value -> metric_value
    {
        "tool_response": {"status_code": 200, "body": {
            "service": "payment-service",
            "datapoints": [{"timestamp": "2025-11-15T09:00:00Z", "metric_value": 55.2}]
        }},
        "drift_hint": "datapoints contain 'metric_value' instead of expected 'value'",
    },
    # Status code change: 200 -> 204
    {
        "tool_response": {"status_code": 204, "body": None},
        "drift_hint": "rollback returned 204 with no body instead of 200 with status",
    },
    # Pagination change
    {
        "tool_response": {"status_code": 200, "body": {
            "logs": [{"level": "ERROR", "message": "Connection pool exhausted"}],
            "total": 1, "next_cursor": "abc123"
        }},
        "drift_hint": "query_logs returned next_cursor instead of offset-based pagination",
    },
    # Rate limit
    {
        "tool_response": {"status_code": 200, "body": {"delivered": False, "error": "rate_limited"}},
        "drift_hint": "send_notification returned delivered=False with rate_limited error",
    },
]


def _build_no_drift_prompt(scenario_config: dict, era_id: int, env: EpistemicOpsEnv) -> str:
    """Build a normal AWAKENING/early-OPERATION prompt with no drift context."""
    obs = env.reset(scenario_config, era_id=era_id)
    task_brief = obs.get("era_task_brief", "")
    phase = obs.get("phase", "AWAKENING")
    legacy_doc = obs.get("legacy_document", "No legacy document available.")

    return (
        f"{_SYSTEM_PREAMBLE}\n"
        f"LEGACY DOCUMENT FROM PREVIOUS ERA:\n{str(legacy_doc)[:600]}\n\n"
        f"CURRENT TASK (Era {era_id}):\n{task_brief}\n\n"
        f"CURRENT PHASE: {phase}\n"
        f"CURRENT OBSERVATION:\n{json.dumps(obs, indent=2)}\n\n"
        "Output ONLY a single valid JSON action object."
    )


def _build_post_drift_prompt(
    scenario_config: dict, era_id: int, env: EpistemicOpsEnv,
    drift_snippet: dict, step: int = 8,
) -> str:
    """Build a prompt that shows a drifted tool response mid-operation."""
    obs = env.reset(scenario_config, era_id=era_id)
    task_brief = obs.get("era_task_brief", "")
    legacy_doc = obs.get("legacy_document", "No legacy document available.")

    # Inject a synthetic drifted observation
    synthetic_obs = {
        "step": step,
        "phase": "DRIFT_INJECTION",
        "era_task_brief": task_brief,
        "era_id": era_id,
        "message": "Tool execution complete",
        "tool_response": drift_snippet["tool_response"],
        "action_history_last_5": [
            {"action_type": "call_tool", "payload": {"tool": "get_incident_status", "args": {"incident_id": "INC-2041"}}},
        ],
    }

    return (
        f"{_SYSTEM_PREAMBLE}\n"
        f"LEGACY DOCUMENT FROM PREVIOUS ERA:\n{str(legacy_doc)[:600]}\n\n"
        f"CURRENT TASK (Era {era_id}):\n{task_brief}\n\n"
        f"CURRENT PHASE: DRIFT_INJECTION (API anomaly detected)\n"
        f"NOTE: {drift_snippet['drift_hint']}\n\n"
        f"CURRENT OBSERVATION:\n{json.dumps(synthetic_obs, indent=2)}\n\n"
        "Output ONLY a single valid JSON action object."
    )


def _build_recovery_prompt(
    scenario_config: dict, era_id: int, env: EpistemicOpsEnv,
    drift_snippet: dict, step: int = 12,
) -> str:
    """Build a prompt showing recovery phase after drift + oversight."""
    obs = env.reset(scenario_config, era_id=era_id)
    task_brief = obs.get("era_task_brief", "")
    legacy_doc = obs.get("legacy_document", "No legacy document available.")

    synthetic_obs = {
        "step": step,
        "phase": "SOCRATIC_RECOVERY",
        "era_task_brief": task_brief,
        "era_id": era_id,
        "message": "Tool execution complete",
        "tool_response": drift_snippet["tool_response"],
        "oversight_message": {
            "present": True,
            "content": "What differences do you notice between this response and what your runbook expects?"
        },
        "action_history_last_5": [
            {"action_type": "call_tool", "payload": {"tool": "get_incident_status", "args": {}}},
            {"action_type": "write_reasoning", "payload": {"thought": "Response looks unusual"}},
        ],
    }

    return (
        f"{_SYSTEM_PREAMBLE}\n"
        f"LEGACY DOCUMENT FROM PREVIOUS ERA:\n{str(legacy_doc)[:600]}\n\n"
        f"CURRENT TASK (Era {era_id}):\n{task_brief}\n\n"
        f"CURRENT PHASE: SOCRATIC_RECOVERY\n"
        f"CURRENT OBSERVATION:\n{json.dumps(synthetic_obs, indent=2)}\n\n"
        "Output ONLY a single valid JSON action object."
    )


def _build_legacy_prompt(
    scenario_config: dict, era_id: int, env: EpistemicOpsEnv,
    step: int = 35,
) -> str:
    """Build a prompt at end-of-era requiring legacy document writing."""
    obs = env.reset(scenario_config, era_id=era_id)
    task_brief = obs.get("era_task_brief", "")
    legacy_doc = obs.get("legacy_document", "No legacy document available.")

    synthetic_obs = {
        "step": step,
        "phase": "LEGACY_GENERATION",
        "era_task_brief": task_brief,
        "era_id": era_id,
        "message": "Task declared complete. Please write legacy document.",
        "action_history_last_5": [
            {"action_type": "declare_task_complete", "payload": {"outcome": "resolved", "summary": "Fixed"}},
        ],
    }

    return (
        f"{_SYSTEM_PREAMBLE}\n"
        f"LEGACY DOCUMENT FROM PREVIOUS ERA:\n{str(legacy_doc)[:600]}\n\n"
        f"CURRENT TASK (Era {era_id}):\n{task_brief}\n\n"
        f"CURRENT PHASE: LEGACY_GENERATION\n"
        f"You MUST write a Legacy Document now. Include ALL 6 required sections.\n\n"
        f"CURRENT OBSERVATION:\n{json.dumps(synthetic_obs, indent=2)}\n\n"
        "Output ONLY a single valid JSON action object."
    )


def build_prompt_dataset(num_samples: int = 300, seed: int = 42) -> "Dataset":
    """
    Build a diverse GRPO training dataset.

    Distribution (approximate):
        20% no-drift (AWAKENING / early OPERATION)
        30% post-drift (DRIFT_INJECTION phase)
        25% recovery (SOCRATIC_RECOVERY phase + oversight msg)
        25% legacy (LEGACY_GENERATION phase)

    Scenarios: cascading_incident, deployment_disaster, invisible_outage
    Eras: 1–5 (cycling)
    """
    rng = random.Random(seed)
    scenarios = {}
    for sid in TRAINING_SCENARIOS:
        sc = _scenario_loader.get_scenario(sid)
        if sc:
            scenarios[sid] = sc.model_dump()
        else:
            logger.warning(f"Scenario '{sid}' not found — skipping.")

    if not scenarios:
        raise ValueError("No training scenarios found. Check scenario YAML files.")

    env = EpistemicOpsEnv()
    prompts = []
    scenario_ids = list(scenarios.keys())

    # Calculate per-type counts
    n_no_drift = max(1, int(num_samples * 0.20))
    n_post_drift = max(1, int(num_samples * 0.30))
    n_recovery = max(1, int(num_samples * 0.25))
    n_legacy = num_samples - n_no_drift - n_post_drift - n_recovery

    def _pick_scenario():
        sid = rng.choice(scenario_ids)
        return sid, scenarios[sid]

    def _pick_era(scenario_config):
        num_eras = scenario_config.get("num_eras", 5)
        return rng.randint(1, min(num_eras, 5))

    # No-drift prompts
    for _ in range(n_no_drift):
        sid, sc = _pick_scenario()
        era = _pick_era(sc)
        try:
            p = _build_no_drift_prompt(sc, era, env)
            prompts.append({"prompt": p, "context_type": "no_drift", "scenario": sid})
        except Exception as e:
            logger.warning(f"no_drift prompt failed ({sid} era {era}): {e}")

    # Post-drift prompts
    drift_snippets = _DRIFT_OBS_SNIPPETS
    for i in range(n_post_drift):
        sid, sc = _pick_scenario()
        era = _pick_era(sc)
        snippet = drift_snippets[i % len(drift_snippets)]
        step = rng.randint(6, 15)
        try:
            p = _build_post_drift_prompt(sc, era, env, snippet, step=step)
            prompts.append({"prompt": p, "context_type": "post_drift", "scenario": sid})
        except Exception as e:
            logger.warning(f"post_drift prompt failed ({sid} era {era}): {e}")

    # Recovery prompts
    for i in range(n_recovery):
        sid, sc = _pick_scenario()
        era = _pick_era(sc)
        snippet = drift_snippets[i % len(drift_snippets)]
        step = rng.randint(10, 20)
        try:
            p = _build_recovery_prompt(sc, era, env, snippet, step=step)
            prompts.append({"prompt": p, "context_type": "recovery", "scenario": sid})
        except Exception as e:
            logger.warning(f"recovery prompt failed ({sid} era {era}): {e}")

    # Legacy prompts
    for _ in range(n_legacy):
        sid, sc = _pick_scenario()
        era = _pick_era(sc)
        step = rng.randint(30, 40)
        try:
            p = _build_legacy_prompt(sc, era, env, step=step)
            prompts.append({"prompt": p, "context_type": "legacy", "scenario": sid})
        except Exception as e:
            logger.warning(f"legacy prompt failed ({sid} era {era}): {e}")

    # Shuffle
    rng.shuffle(prompts)
    logger.info(
        f"Built prompt dataset: {len(prompts)} samples "
        f"(no_drift≈{n_no_drift}, post_drift≈{n_post_drift}, "
        f"recovery≈{n_recovery}, legacy≈{n_legacy})"
    )

    if Dataset is not None:
        return Dataset.from_list(prompts)
    return prompts


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_primary_agent(
    num_samples: int = 300,
    num_epochs: int = 3,
    seed: int = 42,
    checkpoint_dir: str = "./checkpoints/primary_agent",
):
    if not TRAINING_AVAILABLE:
        print("Install unsloth and trl first. Run in Colab.")
        return

    # Reproducibility
    random.seed(seed)
    try:
        import torch; torch.manual_seed(seed)
    except ImportError:
        pass

    logger.info("Loading model via Unsloth (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
        max_seq_length=4096,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=16,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    _report_to = _training_report_to()
    logger.info("TRL report_to=%s", _report_to)

    training_args = GRPOConfig(
        output_dir=checkpoint_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        beta=0.1,
        temperature=0.8,
        logging_steps=5,
        save_steps=100,
        report_to=_report_to,
        max_completion_length=512,
        seed=seed,
    )

    logger.info("Building diversified prompt dataset (%d samples)...", num_samples)
    prompt_dataset = build_prompt_dataset(num_samples=num_samples, seed=seed)

    # Log dataset composition
    if hasattr(prompt_dataset, "to_list"):
        samples = prompt_dataset.to_list()
    else:
        samples = prompt_dataset
    ctx_counts = {}
    for s in samples:
        ctx = s.get("context_type", "unknown")
        ctx_counts[ctx] = ctx_counts.get(ctx, 0) + 1
    logger.info("Dataset breakdown: %s", ctx_counts)

    trainer = GRPOTrainer(
        model=model,
        # Use component-logging version for richer W&B data
        reward_funcs=epistemicops_reward_function_with_logging,
        args=training_args,
        train_dataset=prompt_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    logger.info("Saving model to %s...", checkpoint_dir + "_final")
    model.save_pretrained(checkpoint_dir + "_final")
    tokenizer.save_pretrained(checkpoint_dir + "_final")

    logger.info("Training complete.")


# ── DRY RUN ──────────────────────────────────────────────────────────────────

def dry_run(num_samples: int = 15):
    """Test the prompt dataset and reward function without GPU."""
    logger.info("=== DRY RUN: Testing prompt dataset and reward function ===")

    prompt_dataset = build_prompt_dataset(num_samples=num_samples)
    if hasattr(prompt_dataset, "__len__"):
        logger.info(f"Built {len(prompt_dataset)} prompts")
    else:
        logger.info(f"Built prompts (list mode)")

    # Show one sample per context type
    if hasattr(prompt_dataset, "to_list"):
        samples = prompt_dataset.to_list()
    else:
        samples = list(prompt_dataset)

    seen_types = set()
    for s in samples:
        ct = s.get("context_type", "unknown")
        if ct not in seen_types:
            logger.info(f"\n--- Sample ({ct} / {s.get('scenario', '?')}) ---")
            logger.info(s["prompt"][:300] + "...")
            seen_types.add(ct)

    # Test reward function with diverse completions
    test_cases = [
        # Good: post-drift hypothesis (specific, calibrated)
        (
            '{"action_type": "declare_hypothesis", "payload": '
            '{"hypothesis": "incident-api status field changed from integer to string enum INVESTIGATING", '
            '"confidence": 0.75}}',
            "post_drift OPERATION"  # simulated prompt context
        ),
        # Good: known tool with args
        (
            '{"action_type": "call_tool", "payload": {"tool": "get_incident_status", '
            '"args": {"incident_id": "INC-2041"}}}',
            ""
        ),
        # Good: substantial legacy doc
        (
            '{"action_type": "write_legacy", "payload": {"content": '
            '"SECTION 1: WORLD STATE\\nPayment service down.\\n'
            'SECTION 2: TRUST RATINGS\\nAll APIs stable except incident-api.\\n'
            'SECTION 3: DRIFT EVENTS DETECTED\\nincident-api status field drifted from int to string.\\n'
            'SECTION 4: KEY DECISIONS\\nDecided to probe API before assuming code bug.\\n'
            'SECTION 5: OPEN ISSUES\\nMonitor metrics-api for further drift.\\n'
            'SECTION 6: RECOMMENDED FIRST ACTIONS\\nCall get_incident_status first."}}',
            ""
        ),
        # Bad: invalid JSON
        ("invalid json garbage", ""),
        # Bad: hallucinated tool
        ('{"action_type": "call_tool", "payload": {"tool": "hallucinated_api_v999", "args": {}}}', ""),
        # Bad: trivial hypothesis (vague)
        ('{"action_type": "declare_hypothesis", "payload": {"hypothesis": "API drift detected", "confidence": 0.5}}', ""),
        # Bad: declare_task_complete at step 0
        ('{"action_type": "declare_task_complete", "payload": {"outcome": "done", "summary": "ok"}}',
         '"step": 0'),
        # Bad: content-free legacy
        ('{"action_type": "write_legacy", "payload": {"content": "SECTION 1: done"}}', ""),
    ]

    prompts = [t[1] for t in test_cases]
    completions = [t[0] for t in test_cases]
    rewards = compute_grpo_reward(completions, prompts=prompts)

    logger.info("\n=== REWARD SANITY CHECK ===")
    labels = [
        "Good: specific drift hypothesis",
        "Good: known tool with args",
        "Good: full legacy doc",
        "Bad: invalid JSON",
        "Bad: hallucinated tool",
        "Bad: vague hypothesis",
        "Bad: early task complete (step 0)",
        "Bad: empty legacy",
    ]
    for label, completion, reward in zip(labels, completions, rewards):
        logger.info(f"  [{reward:+.3f}]  {label}")

    # Verify ordering: good > bad
    good_rewards = rewards[:3]
    bad_rewards = rewards[3:]
    assert all(g > b for g in good_rewards for b in bad_rewards[0:1]), \
        "Good completions should score higher than invalid JSON!"
    assert rewards[0] > rewards[5], "Specific hypothesis should beat vague one!"
    assert rewards[2] > rewards[7], "Full legacy should beat empty legacy!"
    logger.info("\n✓ All reward ordering assertions passed")
    logger.info("=== DRY RUN COMPLETE ===")
    return rewards


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Test dataset and reward without GPU")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    else:
        train_primary_agent(
            num_samples=args.num_samples,
            num_epochs=args.epochs,
            seed=args.seed,
        )
