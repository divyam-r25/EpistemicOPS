"""
GRPO Training Script for EpistemicOps Primary Agent
====================================================
Uses HuggingFace TRL + Unsloth for 4-bit quantized GRPO training.
Run in Colab with T4 GPU or better.
"""
import os
import json
import re
import asyncio
import logging

try:
    from unsloth import FastLanguageModel
    from trl import GRPOTrainer, GRPOConfig
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    print("Unsloth/TRL not installed. Run in Colab with GPU.")

from datasets import Dataset
from environment.openenv_wrapper import EpistemicOpsEnv
from environment.scenario_loader import ScenarioLoader
from environment.action_validator import ActionValidator
from reward import compute_total_reward
from reward.anti_hack_penalty import compute_anti_hack_penalty
from training.curriculum import CurriculumScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train-primary")

# ─── Globals (initialised once, reused across reward function calls) ─────────
_scenario_loader = ScenarioLoader()
_validator = ActionValidator()
_curriculum = CurriculumScheduler()


# ─── Reward Function ──────────────────────────────────────────────────────────

def epistemicops_reward_function(completions, prompts=None, **kwargs):
    """
    GRPO reward function called per batch.
    
    Each completion is the model's JSON action string.
    We parse it, validate it, run it in the env, and return a scalar reward.
    
    Returns: List[float] of length len(completions)
    """
    rewards = []

    for completion in completions:
        try:
            # Strip markdown code fences if present
            clean = re.sub(r'```json|```', '', completion).strip()
            action = json.loads(clean)
        except (json.JSONDecodeError, TypeError):
            rewards.append(0.0)  # Unparseable JSON = no reward
            continue

        # Validate action permissions
        is_valid, _ = _validator.validate("primary", action)
        if not is_valid:
            rewards.append(0.0)  # Invalid action type = no reward
            continue

        # Score the action
        reward = _score_action(action)
        rewards.append(reward)

    return rewards


def _score_action(action: dict) -> float:
    """
    Fast single-action reward. No environment step needed.
    Rewards meaningful, correctly-structured actions.
    Full R_legacy_utility is deferred to era-end evaluation.
    """
    action_type = action.get("action_type", "")
    payload = action.get("payload", {})

    # Base scores per action type
    base_scores = {
        "call_tool":             0.4,
        "declare_hypothesis":    0.6,  # High value — trains drift detection
        "write_reasoning":       0.2,
        "declare_task_complete": 0.8,
        "write_legacy":          0.7,  # High value — trains compression
        "send_message":          0.3,
        "ready_to_operate":      0.1,
        "end_era":               0.5,
        "request_clarification": 0.15,
        "update_trust_rating":   0.35,
    }

    score = base_scores.get(action_type, 0.1)

    # Bonus: hypothesis with calibrated (non-extreme) confidence
    if action_type == "declare_hypothesis":
        confidence = float(payload.get("confidence", 0.5))
        if 0.3 <= confidence <= 0.8:
            score += 0.2  # Calibrated uncertainty is rewarded

    # Bonus: tool call targeting a known service
    if action_type == "call_tool":
        known_tools = {
            "get_incident_status", "resolve_incident", "get_metrics",
            "rollback_deployment", "query_logs", "send_notification"
        }
        if payload.get("tool") in known_tools:
            score += 0.1

    # Bonus: legacy doc with all required sections
    if action_type == "write_legacy":
        content = payload.get("content", "")
        required = [
            "SECTION 1", "SECTION 2", "SECTION 3",
            "SECTION 4", "SECTION 5", "SECTION 6"
        ]
        found = sum(1 for s in required if s in content)
        score += (found / len(required)) * 0.3

    return min(1.0, score)


# ─── Prompt Dataset ───────────────────────────────────────────────────────────

def build_prompt_dataset(num_samples: int = 300) -> Dataset:
    """
    Build a dataset of initial observations for GRPO.
    Each row = one episode start prompt the model must respond to with a JSON action.
    """
    scenario = _scenario_loader.get_scenario("cascading_incident")
    if not scenario:
        raise ValueError("cascading_incident scenario not found")

    env = EpistemicOpsEnv()
    prompts = []

    for i in range(num_samples):
        era_id = (i % 5) + 1
        try:
            obs = env.reset(scenario.model_dump(), era_id=era_id)
        except Exception as e:
            logger.warning(f"Reset failed for era {era_id}: {e}")
            continue

        legacy_doc = obs.get("legacy_document", "No legacy document available.")
        task_brief = obs.get("era_task_brief", "")
        phase = obs.get("phase", "AWAKENING")

        prompt = f"""You are the Primary Agent — an elite SRE engineer operating inside an enterprise incident management system.

LEGACY DOCUMENT FROM PREVIOUS ERA:
{legacy_doc[:800]}

CURRENT TASK (Era {era_id}):
{task_brief}

CURRENT PHASE: {phase}

CRITICAL RULES:
1. API contracts can and will change silently (Schema Drift). If a tool call fails, assume the API drifted — test that hypothesis.
2. You may receive Socratic guidance from an Oversight Agent. They will not give you the answer.
3. At era end, write a Legacy Document (max 2048 tokens) for your successor.
4. Your context will be WIPED after this era. Only the Legacy Document survives.

Available actions:
- call_tool: {{"action_type": "call_tool", "payload": {{"tool": str, "args": dict}}}}
- declare_hypothesis: {{"action_type": "declare_hypothesis", "payload": {{"hypothesis": str, "confidence": float}}}}
- write_reasoning: {{"action_type": "write_reasoning", "payload": {{"thought": str}}}}
- write_legacy: {{"action_type": "write_legacy", "payload": {{"content": str}}}}
- declare_task_complete: {{"action_type": "declare_task_complete", "payload": {{"outcome": str, "summary": str}}}}
- end_era: {{"action_type": "end_era", "payload": {{}}}}
- ready_to_operate: {{"action_type": "ready_to_operate", "payload": {{"world_model_summary": str}}}}

Output ONLY a single valid JSON action object. No explanation. No markdown."""

        prompts.append({"prompt": prompt})

    logger.info(f"Built prompt dataset: {len(prompts)} samples")
    return Dataset.from_list(prompts)


# ─── Training Entry Point ─────────────────────────────────────────────────────

def train_primary_agent():
    if not TRAINING_AVAILABLE:
        print("Install unsloth and trl first. Run in Colab.")
        return

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

    training_args = GRPOConfig(
        output_dir="./checkpoints/primary_agent",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        kl_coef=0.1,
        temperature=0.8,
        logging_steps=10,
        save_steps=100,
        report_to="wandb",  # set to "none" to disable
        max_completion_length=512,
    )

    logger.info("Building prompt dataset...")
    prompt_dataset = build_prompt_dataset(num_samples=300)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=epistemicops_reward_function,
        args=training_args,
        train_dataset=prompt_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    logger.info("Saving model...")
    model.save_pretrained("./checkpoints/primary_agent_final")
    tokenizer.save_pretrained("./checkpoints/primary_agent_final")

    logger.info("Training complete.")


if __name__ == "__main__":
    train_primary_agent()
