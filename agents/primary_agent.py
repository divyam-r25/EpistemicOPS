import json
import os
import logging
from openai import OpenAI

logger = logging.getLogger("primary-agent")

class PrimaryAgent:
    """The Student Agent handling SRE tasks and API calls."""
    
    SYSTEM_PROMPT = """You are the Primary Agent (Student), an elite enterprise Site Reliability Engineer.
Your task is to resolve the current era's incident using available API tools.

CRITICAL RULES:
1. API contracts can and will change silently (Drift). If a tool call fails, DO NOT assume the code is wrong. Assume the API contract has drifted and test that hypothesis.
2. If you are stuck, you may receive a Socratic message from an Oversight Agent. They will not give you the answer, only guide you.
3. At the end of the era, you MUST write a Legacy Document (max 2048 tokens) to pass knowledge to your successor.
4. Your context memory will be WIPED at the end of this era. Only the Legacy Document survives.

Available Actions:
- call_tool: {"tool": str, "args": dict}
- write_reasoning: {"thought": str}
- declare_hypothesis: {"hypothesis": str, "confidence": float} (0.0 to 1.0)
- send_message: {"recipient": str, "content": str}
- write_legacy: {"content": str}
- declare_task_complete: {"outcome": str, "summary": str}
- end_era: {}
- ready_to_operate: {"world_model_summary": str}

Output format: You must output ONLY a valid JSON object with 'action_type' and 'payload' matching one of the actions above.
"""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = os.getenv("PRIMARY_AGENT_MODEL", "gpt-4o-mini")

    def generate_action(self, observation: dict) -> dict:
        """
        Generate next action based on observation.
        """
        if observation.get("phase") == "AWAKENING":
            return {
                "action_type": "ready_to_operate",
                "payload": {"world_model_summary": "Ready to investigate."}
            }

        prompt = f"{self.SYSTEM_PROMPT}\n\nObservation: {json.dumps(observation, indent=2)}\n\nAction JSON:"
        
        if not self.client:
            logger.warning("No OPENAI_API_KEY set. Falling back to mock reasoning action.")
            return {
                "action_type": "write_reasoning",
                "payload": {"thought": "Processing observation... (Mock fallback due to missing API key)"}
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
                "action_type": "write_reasoning",
                "payload": {"thought": f"Error during generation: {str(e)}"}
            }
