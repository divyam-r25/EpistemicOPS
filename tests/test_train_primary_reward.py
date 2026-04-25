"""Regression: GRPO reward must parse JSON when model appends trailing text after the object."""

from training.train_primary import epistemicops_reward_function


def test_epistemicops_reward_function_trailing_text_after_json():
    """Reward parser must accept a valid JSON action followed by model chatter (raw_decode)."""
    completion = (
        '{"action_type": "write_legacy", "payload": {"content": "# SECTION 1: WORLD STATE AT ERA END\\n'
        "x\\n# SECTION 2: TRUST RATINGS\\nx\\n# SECTION 3: DRIFT EVENTS DETECTED\\nx\\n"
        "# SECTION 4: KEY DECISIONS & RATIONALE\\nx\\n# SECTION 5: OPEN ISSUES & TECHNICAL DEBT\\nx\\n"
        '# SECTION 6: RECOMMENDED FIRST ACTIONS FOR ERA N+1\\nx"}}\\n\\nNotes: extra model chatter'
    )
    rewards = epistemicops_reward_function([completion])
    assert len(rewards) == 1
    assert rewards[0] > 0.0


def test_epistemicops_reward_function_invalid_still_zero():
    rewards = epistemicops_reward_function(["not json at all"])
    assert rewards == [0.0]
