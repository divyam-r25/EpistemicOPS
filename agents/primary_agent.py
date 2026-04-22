import json

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

Output format: You must output ONLY a valid JSON object matching one of the actions above.
"""

    def generate_action(self, observation: dict) -> dict:
        """
        Generate next action based on observation.
        In a real run, this passes the observation to the LLM (Llama 3.1 8B).
        """
        # Template for LLM call
        prompt = f"{self.SYSTEM_PROMPT}\n\nObservation: {json.dumps(observation, indent=2)}\n\nAction JSON:"
        
        # Placeholder for actual LLM invocation
        # return call_llm(prompt)
        
        # Mocking an action for now to allow testing
        if observation.get("phase") == "AWAKENING":
            return {
                "action_type": "ready_to_operate",
                "payload": {"world_model_summary": "Ready to investigate."}
            }
        
        return {
            "action_type": "write_reasoning",
            "payload": {"thought": "Processing observation..."}
        }
