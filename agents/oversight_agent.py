import json

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

Output format: You must output ONLY a valid JSON object matching one of the actions above.
"""

    def generate_intervention(self, primary_trace: list, drift_config: dict, prior_interventions: list) -> dict:
        """
        Generate a pedagogical intervention.
        In a real run, this passes the context to the LLM.
        """
        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Drift Ground Truth: {json.dumps(drift_config)}\n"
            f"Primary Agent Reasoning Trace: {json.dumps(primary_trace)}\n"
            f"Your Prior Interventions: {json.dumps(prior_interventions)}\n\n"
            f"Intervention JSON:"
        )
        
        # Placeholder for actual LLM invocation
        # return call_llm(prompt)
        
        return {
            "action_type": "oversight_targeted_question",
            "payload": {"question": "What assumption did you make about the API response schema?"}
        }
