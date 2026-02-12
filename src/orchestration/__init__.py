"""
Orchestration module for the Multi-Agent GitHub Issue Routing System.

This package contains the orchestration engine, moderator agent,
convergence detection, and workflow state management.
"""

from src.orchestration.convergence import (
    ConvergenceDetector,
    calculate_convergence_score,
    detect_rambling,
    measure_value_added,
)
from src.orchestration.deliberation import (
    DeliberationResult,
    MultiAgentDeliberationOrchestrator,
)
from src.orchestration.moderator import (
    AgentSelectionResult,
    ModeratorAgent,
    select_agents_by_keywords,
)
from src.orchestration.worker import Worker, WorkerStats

__all__ = [
    "AgentSelectionResult",
    "ConvergenceDetector",
    "DeliberationResult",
    "ModeratorAgent",
    "MultiAgentDeliberationOrchestrator",
    "Worker",
    "WorkerStats",
    "calculate_convergence_score",
    "detect_rambling",
    "measure_value_added",
    "select_agents_by_keywords",
]
