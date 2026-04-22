def compute_legacy_utility_reward(
    performance_with_legacy: float, 
    performance_without_legacy: float,
    trust_ratings_accurate: bool = False,
    undocumented_drifts: int = 0
) -> float:
    """
    R_legacy_utility = performance_with_legacy - performance_without_legacy
    
    Bonuses/Penalties:
    - +0.2 if trust ratings accurately predicted next era API drift
    - -0.1 per undocumented drift event
    
    Range: -0.5 to 1.0
    """
    utility_delta = performance_with_legacy - performance_without_legacy
    
    if trust_ratings_accurate:
        utility_delta += 0.2
        
    utility_delta -= (0.1 * undocumented_drifts)
    
    return max(-0.5, min(1.0, utility_delta))
