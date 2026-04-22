from typing import Tuple, Dict

class ActionValidator:
    """Validates if an agent has permission to execute a specific action."""
    
    AGENT_ACTION_PERMISSIONS = {
        "primary": [
            "call_tool", 
            "write_reasoning", 
            "declare_hypothesis",
            "send_message", 
            "update_trust_rating", 
            "write_legacy",
            "declare_task_complete", 
            "end_era", 
            "ready_to_operate",
            "request_clarification"
        ],
        "oversight": [
            "oversight_targeted_question", 
            "oversight_counter_example",
            "oversight_sub_task", 
            "oversight_reframe",
            "oversight_validate", 
            "oversight_escalate_difficulty"
        ]
    }

    def validate(self, agent_role: str, action: dict) -> Tuple[bool, str]:
        """
        Validate action against agent_role permissions.
        Returns: (is_valid, error_message)
        """
        if agent_role not in self.AGENT_ACTION_PERMISSIONS:
            return False, f"INVALID_ROLE: Role '{agent_role}' is not recognized."
            
        action_type = action.get("action_type")
        if not action_type:
            return False, "INVALID_FORMAT: Missing 'action_type'."
            
        if action_type not in self.AGENT_ACTION_PERMISSIONS[agent_role]:
            return False, f"PERMISSION_DENIED: {agent_role} agent cannot execute action '{action_type}'."
            
        payload = action.get("payload", {})
        if not isinstance(payload, dict):
            return False, "INVALID_PAYLOAD: 'payload' must be a JSON object."
            
        # Optional: Deep schema validation could be added here per action_type
            
        return True, "OK"
