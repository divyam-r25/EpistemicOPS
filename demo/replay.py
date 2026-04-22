import json

class EpisodeReplayer:
    """Loads a pre-recorded episode JSON and allows step-by-step playback for the UI."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.trajectory = []
        self.current_step = 0
        self._load()
        
    def _load(self):
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                self.trajectory = data.get("steps", [])
        except Exception as e:
            print(f"Failed to load trajectory: {e}")
            
    def get_next_step(self) -> dict:
        if self.current_step < len(self.trajectory):
            step_data = self.trajectory[self.current_step]
            self.current_step += 1
            return step_data
        return {"done": True}
        
    def reset(self):
        self.current_step = 0
