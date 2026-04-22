import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from environment.openenv_wrapper import EpistemicOpsEnv
from environment.scenario_loader import ScenarioLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openenv-server")

app = FastAPI(title="EpistemicOps OpenEnv API")

# Initialize global environment instance
env = EpistemicOpsEnv()
scenario_loader = ScenarioLoader()

class ResetRequest(BaseModel):
    scenario_id: str
    era_id: int = 1
    legacy_doc: Optional[str] = None

class ActionRequest(BaseModel):
    agent_role: str
    action: Dict[str, Any]

@app.post("/reset")
async def reset(request: ResetRequest):
    scenario_config = scenario_loader.get_scenario(request.scenario_id)
    if not scenario_config:
        raise HTTPException(status_code=404, detail=f"Scenario {request.scenario_id} not found")
        
    obs = env.reset(
        scenario_config=scenario_config.model_dump(), 
        era_id=request.era_id, 
        legacy_doc=request.legacy_doc
    )
    return {"observation": obs}

@app.post("/step")
async def step(request: ActionRequest):
    try:
        obs, reward, done, info = await env.step(request.agent_role, request.action)
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        logger.error(f"Error during step: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
async def state():
    return env.state()

@app.get("/scenarios")
async def get_scenarios():
    return [s.model_dump() for s in scenario_loader.get_all_scenarios()]
