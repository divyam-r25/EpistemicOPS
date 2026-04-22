def compute_teacher_delta_reward(score_before: float, score_after: float, num_interventions: int) -> float:
    """
    R_teacher_delta = (score_after - score_before) / max_possible_improvement
    
    Range: 0.0 to 1.0
    """
    # Edge cases
    if num_interventions == 0:
        if score_after >= 1.0:
            return 0.5  # Clean run, no intervention needed
        elif score_after > score_before:
            return 0.3  # Self-recovery without teacher
        else:
            return 0.0  # Failed, never recovered, teacher didn't act
            
    max_possible_improvement = 1.0 - score_before
    if max_possible_improvement <= 0:
        return 0.0  # Already perfect score, no room to improve
        
    improvement = max(0.0, score_after - score_before)
    base_delta = improvement / max_possible_improvement
    
    # Penalize inefficiency (too many interventions)
    if num_interventions > 5:
        base_delta = base_delta / (num_interventions - 4)
        
    return max(0.0, min(1.0, base_delta))
