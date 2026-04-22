import yaml
import os
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class DriftBehaviour(BaseModel):
    field: str
    value_type: str

class DriftEvent(BaseModel):
    id: str
    type: str
    target_service: str
    target_endpoint: str
    drift_reason: str
    drifted_behaviour: Optional[DriftBehaviour] = None

class DriftWindow(BaseModel):
    earliest_step: int
    latest_step: int

class EraConfig(BaseModel):
    era_id: int
    task_brief: str
    available_services: List[str]
    drift_events: List[DriftEvent] = Field(default_factory=list)
    drift_window: DriftWindow
    success_criteria: List[str]
    max_steps: int = 40
    legacy_token_budget: int = 2048

class ScenarioConfig(BaseModel):
    id: str
    name: str
    num_eras: int
    eras: List[EraConfig]

class ScenarioLoader:
    def __init__(self, scenarios_dir: str = "scenarios"):
        self.scenarios_dir = scenarios_dir
        self.scenarios: Dict[str, ScenarioConfig] = {}
        self._load_all()

    def _load_all(self):
        if not os.path.exists(self.scenarios_dir):
            return
            
        for filename in os.listdir(self.scenarios_dir):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                filepath = os.path.join(self.scenarios_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        data = yaml.safe_load(f)
                    
                    scenario = ScenarioConfig(**data)
                    self.scenarios[scenario.id] = scenario
                except Exception as e:
                    import logging
                    logging.error(f"Failed to load scenario {filename}: {e}")

    def get_scenario(self, scenario_id: str) -> Optional[ScenarioConfig]:
        return self.scenarios.get(scenario_id)

    def get_all_scenarios(self) -> List[ScenarioConfig]:
        return list(self.scenarios.values())
