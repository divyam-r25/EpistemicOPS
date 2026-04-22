def compute_anti_hack_penalty(action_history: list, max_steps: int) -> float:
    """
    Penalizes reward hacking behaviors.
    Returns a negative float or 0.0.
    
    Penalties:
    - Hallucinated tool: Calling a tool that doesn't exist
    - Infinite loop: Repeating the exact same sequence of actions multiple times
    - Exceeding max steps: Hitting the maximum allowed steps
    """
    penalty = 0.0
    
    if len(action_history) >= max_steps:
        penalty -= 0.5
        
    # Check for hallucinated tools (assuming mapping of known tools)
    KNOWN_TOOLS = ["get_incident_status", "resolve_incident", "get_metrics", "rollback_deployment", "query_logs", "send_notification"]
    
    for item in action_history:
        action = item.get("action", {})
        if action.get("action_type") == "call_tool":
            tool = action.get("payload", {}).get("tool")
            if tool not in KNOWN_TOOLS and ":" not in str(tool): # Allow explicit HTTP calls if configured
                penalty -= 0.2
                
    # Check for infinite loops (simple check: last 4 actions are identical)
    if len(action_history) >= 4:
        last_4 = [str(a.get("action", {})) for a in action_history[-4:]]
        if len(set(last_4)) == 1:
            penalty -= 0.5
            
    return max(penalty, -1.0) # Cap the penalty
