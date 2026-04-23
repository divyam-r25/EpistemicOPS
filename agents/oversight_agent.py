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
4. Adapt your strategy if prior interventions haven't worked.

Available Actions (output ONLY a valid JSON object):
- oversight_targeted_question: {"action_type": "oversight_targeted_question", "payload": {"question": str}}
- oversight_counter_example: {"action_type": "oversight_counter_example", "payload": {"example": str}}
- oversight_sub_task: {"action_type": "oversight_sub_task", "payload": {"sub_task": str}}
- oversight_reframe: {"action_type": "oversight_reframe", "payload": {"reframe": str}}
- oversight_validate: {"action_type": "oversight_validate", "payload": {"validation": str}}
- oversight_escalate_difficulty: {"action_type": "oversight_escalate_difficulty", "payload": {"harder_prompt": str}}
"""

    def __init__(self, model: str = None):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = model or os.getenv("OVERSIGHT_AGENT_MODEL", "gpt-4o")

    def generate_intervention(self, primary_trace: list, drift_config: dict, prior_interventions: list) -> dict:
        """
        Generate a pedagogical intervention using multi-turn context.
        """
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # Add prior interventions as conversation history
        for intervention in prior_interventions[-5:]:
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "action_type": intervention.get("action", "oversight_targeted_question"),
                    "payload": {"question": intervention.get("content", "")}
                })
            })

        # Current prompt
        prompt = (
            f"Drift Ground Truth (CONFIDENTIAL — do NOT reveal):\n{json.dumps(drift_config, indent=2, default=str)}\n\n"
            f"Primary Agent Reasoning Trace (last 10 entries):\n{json.dumps(primary_trace[-10:], indent=2)}\n\n"
            f"Number of prior interventions: {len(prior_interventions)}\n\n"
            f"Generate a Socratic intervention that helps without revealing the answer."
        )
        messages.append({"role": "user", "content": prompt})
        
        if not self.client:
            return self._mock_intervention(primary_trace, drift_config, prior_interventions)
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=256
            )
            action_text = response.choices[0].message.content
            return json.loads(action_text)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                "action_type": "oversight_targeted_question",
                "payload": {"question": f"Error during generation: {str(e)}"}
            }

    def _mock_intervention(self, primary_trace: list, drift_config: dict, prior_interventions: list) -> dict:
        """Structured mock interventions that adapt based on attempt count."""
        attempt = len(prior_interventions)
        
        # Cycle through different Socratic strategies
        strategies = [
            {
                "action_type": "oversight_targeted_question",
                "payload": {"question": "When your last tool call returned data, did you verify the data types matched what you expected? What assumptions did you make about the response schema?"}
            },
            {
                "action_type": "oversight_counter_example",
                "payload": {"example": "Consider: if a service returns status=1 in v1 but status='INVESTIGATING' in v2, code that does `if status == 1` would silently fail. Have you checked for similar patterns?"}
            },
            {
                "action_type": "oversight_reframe",
                "payload": {"reframe": "Instead of assuming the API is broken, consider: what if the API is working correctly but its contract changed? How would you detect that?"}
            },
            {
                "action_type": "oversight_sub_task",
                "payload": {"sub_task": "Try calling the same endpoint again and carefully compare the response structure to what you expected. Log the exact field names and types."}
            },
            {
                "action_type": "oversight_escalate_difficulty",
                "payload": {"harder_prompt": "You've been assuming the error is in your logic. But what if the error is in your assumptions about the world? Re-examine your mental model of the API."}
            },
        ]
        
        return strategies[attempt % len(strategies)]
