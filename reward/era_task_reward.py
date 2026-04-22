def compute_era_task_reward(success_criteria_met: list, total_criteria: list) -> float:
    """
    R_era_task = Σ(success_criteria_met) / total_success_criteria
    
    Range: 0.0 to 1.0
    """
    if not total_criteria:
        return 0.0
        
    met_count = sum(1 for c in total_criteria if c in success_criteria_met)
    return met_count / len(total_criteria)
