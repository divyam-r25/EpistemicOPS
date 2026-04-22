"""
EpistemicOps Agents
====================
Primary Agent (Student), Oversight Agent (Teacher), and LLM Judge.
"""

from agents.primary_agent import PrimaryAgent
from agents.oversight_agent import OversightAgent
from agents.llm_judge import LLMJudge

__all__ = ["PrimaryAgent", "OversightAgent", "LLMJudge"]
