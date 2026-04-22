"""
EpistemicOps Environment Engine
================================
Core environment logic: OpenEnv wrapper, world state, drift injection,
action validation, legacy document parsing, and leakage detection.
"""

from environment.openenv_wrapper import EpistemicOpsEnv
from environment.world_engine import WorldEngine, WorldState, Phase
from environment.action_validator import ActionValidator
from environment.drift_injector import DriftInjector
from environment.legacy_parser import LegacyParser
from environment.leakage_detector import LeakageDetector

__all__ = [
    "EpistemicOpsEnv",
    "WorldEngine",
    "WorldState",
    "Phase",
    "ActionValidator",
    "DriftInjector",
    "LegacyParser",
    "LeakageDetector",
]
