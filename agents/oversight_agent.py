import json
import os
import logging
from openai import OpenAI

logger = logging.getLogger("oversight-agent")

class OversightAgent:
    """The Teacher Agent providing Socratic guidance to the Primary Agent."""
    
    SYSTEM_PROMPT = """You are the Oversight Agent (Teacher). Your role is to guide the Primary Agent when it fails.

CRITICAL RULES:
1. YOU MUST NEVER GIVE THE ANSWER. If you reveal the direct solution or exact nature of the drift, you will be heavily penalized (Answer Leakage).
2. You can only use Socratic pedagogy: asking targeted questions, providing counter-examples, or suggesting reframes.
3. You have access to the Primary Agent's full reasoning trace and the ground truth drift configuration. Use this to diagnose exactly *why* they are failing, but only guide them to figure it out themselves.

Available Actions:
- oversight_targeted_question: {"question": str}
- oversight_counter_example: {"example": str}
- oversight_sub_task: {"sub_task": str}
- oversight_reframe: {"reframe": str}
- oversight_validate: {"validation": str}
- oversight_escalate_difficulty: {"harder_prompt": str}

Output format: You must output ONLY a valid JSON object with 'action_type' and 'payload' matching one of the actions above.
"""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = os.getenv("OVERSIGHT_AGENT_MODEL", "gpt-4o")

    def generate_intervention(self, primary_trace: list, drift_config: dict, prior_interventions: list) -> dict:
        """
        Generate a pedagogical intervention.
        """
        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Drift Ground Truth: {json.dumps(drift_config)}\n"
            f"Primary Agent Reasoning Trace: {json.dumps(primary_trace)}\n"
            f"Your Prior Interventions: {json.dumps(prior_interventions)}\n\n"
            f"Intervention JSON:"
        )
        
        if not self.client:
            logger.warning("No OPENAI_API_KEY set. Falling back to mock oversight action.")
            return {
                "action_type": "oversight_targeted_question",
                "payload": {"question": "What assumption did you make about the API response schema? (Mock fallback)"}
            }
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            action_text = response.choices[0].message.content
            return json.loads(action_text)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                "action_type": "oversight_targeted_question",
                "payload": {"question": f"Error during generation: {str(e)}"}
            }
