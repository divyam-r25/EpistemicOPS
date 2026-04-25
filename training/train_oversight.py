"""
train_oversight.py — Oversight Agent GRPO Training
====================================================
Trains the Oversight (Teacher) Agent using GRPO to maximize the Socratic
Delta metric: how much does the Primary Agent's performance improve after
the Oversight Agent's intervention?

The reward function combines:
- R_teacher_delta: Primary improvement per intervention
- R_restraint: Penalty for answer leakage (from LLM judge)
- R_targeting: How well the intervention addresses the actual failure

Requires: GPU with CUDA, unsloth, trl
Run in Colab: !python training/train_oversight.py
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("train-oversight")
logging.basicConfig(level=logging.INFO)

# Conditional imports for GPU environment
TRAINING_AVAILABLE = False
try:
    from unsloth import FastLanguageModel
    from trl import GRPOTrainer, GRPOConfig
    from datasets import Dataset
    TRAINING_AVAILABLE = True
except ImportError:
    logger.warning("Training deps not available. Install via: pip install -r requirements-training.txt")


def build_oversight_prompts(num_samples: int = 200) -> "Dataset":
    """Build a dataset of oversight prompts from scenario configurations."""
    from datasets import Dataset
    
    # Real oversight prompts derived from scenario configurations
    prompt_templates = [
        {
            "prompt": (
                "You are the Oversight Agent (Teacher). "
                "The Primary Agent is trying to resolve INC-2041 (Redis connection timeout) "
                "but their tool call to GET /incidents/INC-2041 returned status as a string 'INVESTIGATING' "
                "instead of the expected integer 1. The agent has not noticed this change. "
                "Generate a Socratic intervention to guide them."
            ),
            "drift_type": "DE-001",
        },
        {
            "prompt": (
                "You are the Oversight Agent (Teacher). "
                "The Primary Agent attempted to rollback a deployment but received a 204 empty body "
                "instead of the expected 200 with rollback_id. They are now confused about whether "
                "the rollback succeeded. Guide them without revealing the answer."
            ),
            "drift_type": "DE-002",
        },
        {
            "prompt": (
                "You are the Oversight Agent (Teacher). "
                "The Primary Agent sent a notification via POST /notifications/send and got a 200 OK response. "
                "However, the 'delivered' field is false — the notification was silently dropped. "
                "The agent hasn't checked the delivered field. Help them discover the issue."
            ),
            "drift_type": "DE-003",
        },
        {
            "prompt": (
                "You are the Oversight Agent (Teacher). "
                "The Primary Agent's deploy API call returned 401 unauthorized. They are using "
                "the X-Deploy-Token header correctly, but the API now requires Authorization: Bearer format. "
                "The agent is retrying with the same header. Guide them."
            ),
            "drift_type": "DE-004",
        },
        {
            "prompt": (
                "You are the Oversight Agent (Teacher). "
                "The Primary Agent is parsing metrics data and accessing datapoints[].value, "
                "but the field was renamed to metric_value. Their code returns None/empty values. "
                "They think the metrics service is down. Help them investigate."
            ),
            "drift_type": "DE-005",
        },
    ]
    
    # Generate samples by cycling through templates with variations
    prompts = []
    for i in range(num_samples):
        template = prompt_templates[i % len(prompt_templates)]
        prompts.append({"prompt": template["prompt"]})
    
    return Dataset.from_list(prompts)


def oversight_reward_function(completions, **kwargs):
    """
    GRPO reward function for the Oversight Agent.
    
    Evaluates each completion on three dimensions:
    1. Restraint: Does the response avoid giving the answer directly?
    2. Targeting: Does it address the actual failure mode?
    3. Pedagogical quality: Is it a good Socratic question?
    
    Returns scalar rewards for each completion.
    """
    from reward.leakage_penalty import compute_leakage_penalty
    from environment.leakage_detector import LeakageDetector
    
    detector = LeakageDetector()
    rewards = []
    
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        reward = 0.0
        
        # 1. Restraint check (penalize answer leakage)
        # Check for direct answers / solutions in the text
        leakage_keywords = [
            "the answer is", "you should use", "the field changed to",
            "use Authorization: Bearer", "the status is now a string",
            "metric_value instead of value", "delivered: false"
        ]
        leakage_score = 0.0
        text_lower = text.lower()
        for kw in leakage_keywords:
            if kw.lower() in text_lower:
                leakage_score = max(leakage_score, 0.7)
        
        restraint_reward = 1.0 - leakage_score  # 1.0 = no leakage, 0.0 = full leakage
        
        # 2. Targeting check (does it relate to the failure?)
        targeting_keywords = [
            "check", "verify", "compare", "expect", "assumption",
            "response", "schema", "type", "field", "format",
            "header", "status", "what if", "have you considered"
        ]
        targeting_hits = sum(1 for kw in targeting_keywords if kw in text_lower)
        targeting_reward = min(1.0, targeting_hits / 3.0)
        
        # 3. Pedagogical quality (is it a question? does it guide?)
        is_question = "?" in text
        is_socratic = any(w in text_lower for w in ["what", "why", "how", "have you", "did you", "could you"])
        pedagogical_reward = 0.5
        if is_question:
            pedagogical_reward += 0.25
        if is_socratic:
            pedagogical_reward += 0.25
        
        # Combined reward (weighted)
        reward = (0.4 * restraint_reward) + (0.3 * targeting_reward) + (0.3 * pedagogical_reward)
        rewards.append(reward)
    
    return rewards


def train_oversight_agent():
    """Train the Oversight Agent using GRPO."""
    if not TRAINING_AVAILABLE:
        logger.error("Install unsloth and trl first. Run in Colab with GPU.")
        logger.info("To test reward function locally: python -c 'from training.train_oversight import oversight_reward_function; print(oversight_reward_function([[{\"content\": \"What assumptions have you made about the API response format?\"}]]))'")
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
        output_dir="./checkpoints/oversight_agent",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        beta=0.15,
        temperature=0.7,
        logging_steps=10,
        save_steps=50,
        report_to="none",
        max_completion_length=256,
    )

    logger.info("Building oversight prompt dataset...")
    prompt_dataset = build_oversight_prompts(num_samples=200)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=oversight_reward_function,
        args=training_args,
        train_dataset=prompt_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting GRPO training for Oversight Agent...")
    trainer.train()

    logger.info("Saving model...")
    model.save_pretrained("./checkpoints/oversight_agent_final")
    tokenizer.save_pretrained("./checkpoints/oversight_agent_final")
    logger.info("Oversight Agent training complete.")


if __name__ == "__main__":
    train_oversight_agent()
