import logging

logger = logging.getLogger("curriculum")

class CurriculumScheduler:
    """Manages the progressive difficulty scaling during training."""
    
    def __init__(self):
        self.schedule = [
            {"level": 1, "scenarios": ["cascading_incident"], "eras": 2, "until_reward": 0.50},
            {"level": 2, "scenarios": ["cascading_incident", "deployment_disaster"], "eras": 3, "until_reward": 0.65},
            {"level": 3, "scenarios": ["cascading_incident", "deployment_disaster"], "eras": 5, "until_reward": 0.75},
        ]
        self.current_level_idx = 0
        self.rolling_rewards = []
        self.window_size = 50  # Average over last 50 episodes
        
    def get_current_config(self) -> dict:
        """Get the scenario and era bounds for the current training level."""
        return self.schedule[self.current_level_idx]
        
    def log_episode_reward(self, normalized_reward: float) -> bool:
        """
        Log a reward and check if it's time to advance the curriculum.
        Returns True if the level advanced.
        """
        self.rolling_rewards.append(normalized_reward)
        if len(self.rolling_rewards) > self.window_size:
            self.rolling_rewards.pop(0)
            
        return self._check_advancement()
        
    def _check_advancement(self) -> bool:
        if self.current_level_idx >= len(self.schedule) - 1:
            return False # Max level reached
            
        if len(self.rolling_rewards) < self.window_size:
            return False # Not enough data
            
        current_avg = sum(self.rolling_rewards) / len(self.rolling_rewards)
        target = self.schedule[self.current_level_idx]["until_reward"]
        
        if current_avg >= target:
            self.current_level_idx += 1
            self.rolling_rewards = [] # Reset for new level
            logger.info(f"Curriculum advanced to Level {self.schedule[self.current_level_idx]['level']}!")
            return True
            
        return False
