import pytest
import asyncio
from environment.openenv_wrapper import EpistemicOpsEnv

@pytest.mark.asyncio
async def test_full_era_run():
    """Integration test simulating a very basic 1-era run."""
    env = EpistemicOpsEnv()
    
    scenario_config = {
        "id": "test_scenario",
        "eras": [
            {
                "era_id": 1, 
                "task_brief": "Test brief",
                "drift_window": {"earliest_step": 100, "latest_step": 200}
            }
        ]
    }
    
    obs = env.reset(scenario_config, era_id=1)
    assert obs["phase"] == "AWAKENING"
    
    # 1. Ready
    obs, r, done, info = await env.step("primary", {"action_type": "ready_to_operate"})
    assert info["phase"] == "OPERATION"
    
    # 2. Write legacy
    obs, r, done, info = await env.step("primary", {
        "action_type": "write_legacy", 
        "payload": {"content": "Test legacy doc"}
    })
    
    # 3. End era
    obs, r, done, info = await env.step("primary", {"action_type": "end_era"})
    
    assert done is True
    assert env.current_legacy_doc is not None
