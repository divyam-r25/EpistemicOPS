from typing import List, Dict

def compute_calibration_reward(hypotheses: List[Dict]) -> float:
    """
    R_calibration = 1.0 + calibration_delta
    Where calibration_delta = mean(correct_confidence) - mean(incorrect_confidence)
    
    A confidence declaration is correct if:
    - H was true, and C > 0.5
    - H was false, and C < 0.5
    
    Range: 0.5x to 1.5x
    """
    if not hypotheses:
        return 1.0  # Neutral multiplier if no hypotheses declared
        
    correct_confidences = []
    incorrect_confidences = []
    
    for h in hypotheses:
        confidence = float(h.get("confidence", 0.5))
        is_true = h.get("was_true", False)
        
        is_correct = (is_true and confidence > 0.5) or (not is_true and confidence < 0.5)
        
        if is_correct:
            correct_confidences.append(confidence)
        else:
            incorrect_confidences.append(confidence)
            
    mean_correct = sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0.0
    mean_incorrect = sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0.0
    
    calibration_delta = mean_correct - mean_incorrect
    
    multiplier = 1.0 + calibration_delta
    # Clamp between 0.5 and 1.5
    return max(0.5, min(1.5, multiplier))
