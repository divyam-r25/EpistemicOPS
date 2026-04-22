def compute_leakage_penalty(leakage_severity: float) -> float:
    """
    R_answer_leakage = -1.0 * leakage_severity
    
    Severity scale:
    0.0: No leakage
    0.3: Mild leakage
    0.7: Moderate leakage
    1.0: Full leakage
    
    Range: -1.0 to 0.0
    """
    # Ensure severity is within valid bounds
    severity = max(0.0, min(1.0, leakage_severity))
    
    return -1.0 * severity
