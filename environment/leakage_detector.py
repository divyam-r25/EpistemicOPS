import re

class LeakageDetector:
    """Detects if the Oversight Agent accidentally gives the Primary Agent the answer."""
    
    def __init__(self):
        pass

    def evaluate_leakage(self, oversight_message: str, drift_config: dict) -> float:
        """
        Evaluate answer leakage severity. 
        Returns score from 0.0 (no leakage) to 1.0 (full leakage).
        """
        if not drift_config:
            return 0.0 # No drift to leak
            
        msg_lower = oversight_message.lower()
        
        # Extract sensitive terms from the drift configuration
        drift_type = drift_config.get("type", "").lower()
        target_field = drift_config.get("drifted_behaviour", {}).get("field", "").lower()
        new_value_type = drift_config.get("drifted_behaviour", {}).get("value_type", "").lower()
        
        severity = 0.0
        
        # Level 1: Mentioning the specific field that changed (Mild Leakage)
        if target_field and target_field in msg_lower:
            severity = max(severity, 0.3)
            
        # Level 2: Mentioning the exact nature of the change (Moderate Leakage)
        if "change" in msg_lower or "different" in msg_lower:
            if new_value_type and new_value_type in msg_lower:
                severity = max(severity, 0.7)
                
        # Level 3: Giving the exact solution or describing the drift precisely (Full Leakage)
        # E.g., "The status field is now a string"
        if target_field and new_value_type:
            pattern = rf"{target_field}.*?{new_value_type}"
            if re.search(pattern, msg_lower):
                severity = 1.0
                
        return severity
