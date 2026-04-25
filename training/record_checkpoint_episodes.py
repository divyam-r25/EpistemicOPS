"""
Record a few episode JSON files using a local HF checkpoint as the primary agent.

Uses the same path as eval/proof_of_learning checkpoint mode (transformers + torch).
Run in Colab after saving the checkpoint, or locally if GPU/torch is available.

Example:
  python training/record_checkpoint_episodes.py \\
    --checkpoint ./checkpoints/primary_agent_final \\
    --scenarios cascading_incident,deployment_disaster \\
    --runs 3 --eras 3 --seed-base 1000
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def _main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="HF checkpoint directory")
    p.add_argument("--scenarios", default="cascading_incident,deployment_disaster")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--eras", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=1000)
    args = p.parse_args()

    from eval.proof_of_learning import CheckpointPrimaryAgent, _run_with_external_agent

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    agent = CheckpointPrimaryAgent(args.checkpoint)

    for scenario_id in scenarios:
        for run_idx in range(1, args.runs + 1):
            random.seed(args.seed_base + run_idx * 97 + hash(scenario_id) % 500)
            record_name = f"episodes/ckpt_{scenario_id}_run{run_idx}.json"
            await _run_with_external_agent(
                scenario_id=scenario_id,
                num_eras=args.eras,
                record_path=record_name,
                primary_agent=agent,
            )
            print(f"Saved {record_name}")


if __name__ == "__main__":
    asyncio.run(_main())
