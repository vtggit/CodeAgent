"""
Convergence detection for the Multi-Agent GitHub Issue Routing System.

This module implements the convergence detection logic that determines
when a multi-agent deliberation should conclude. It uses multiple
heuristics and optional LLM analysis to measure:

1. Agreement/consensus level among agents
2. Rambling and repetition detection
3. Value-added measurement for each round
4. Topic diversity tracking

The convergence detector is used by the ModeratorAgent to decide
whether to continue to the next round or stop the deliberation.
"""

import json
import logging
import os
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Optional

import litellm

from src.models.workflow import Comment, ContinueDecision, WorkflowConfig

logger = logging.getLogger(__name__)


# ==================
# Text Analysis Helpers
# ==================

# Agreement indicators (patterns that suggest consensus)
AGREEMENT_PATTERNS = [
    r"\bagree\b",
    r"\bsecond\s+that\b",
    r"\bsupport\b",
    r"\bconcur\b",
    r"\bgood\s+point\b",
    r"\bexactly\b",
    r"\byes\b",
    r"\balign\b",
    r"\bconsensus\b",
    r"\bbuilding\s+on\b",
    r"\bin\s+line\s+with\b",
    r"\breinforce\b",
    r"\bconsistent\s+with\b",
    r"\becho\b",
    r"\b\+1\b",
]

# Disagreement indicators
DISAGREEMENT_PATTERNS = [
    r"\bdisagree\b",
    r"\bhowever\b",
    r"\bbut\b",
    r"\bon\s+the\s+other\s+hand\b",
    r"\balternatively\b",
    r"\bconcern\b",
    r"\brisk\b",
    r"\bcaution\b",
    r"\bcontrad\w+\b",
    r"\boppose\b",
    r"\binstead\b",
    r"\brather\b",
    r"\bpushback\b",
]

# Value indicators (new information, solutions, etc.)
VALUE_PATTERNS = [
    r"\brecommend\b",
    r"\bpropose\b",
    r"\bsuggest\b",
    r"\bconsider\b",
    r"\bimportant\b",
    r"\bcritical\b",
    r"\bsolution\b",
    r"\bapproach\b",
    r"\bimplement\b",
    r"\barchitect\b",
    r"\bstrategy\b",
    r"\bpattern\b",
    r"\bedge\s+case\b",
    r"\btrade-?off\b",
    r"\bconstraint\b",
    r"\brequirement\b",
]

# Rambling indicators
RAMBLING_PATTERNS = [
    r"\bas\s+(?:i|we)\s+(?:said|mentioned|noted)\s+(?:before|earlier|previously)\b",
    r"\bto\s+reiterate\b",
    r"\bagain\b",
    r"\brepeating\b",
    r"\bas\s+previously\s+(?:stated|mentioned|discussed)\b",
    r"\bonce\s+more\b",
]


def count_pattern_matches(text: str, patterns: list[str]) -> int:
    """Count matches of regex patterns in text."""
    count = 0
    text_lower = text.lower()
    for pattern in patterns:
        count += len(re.findall(pattern, text_lower, re.IGNORECASE))
    return count


def text_similarity(text1: str, text2: str) -> float:
    """
    Calculate text similarity ratio using SequenceMatcher.

    Returns a float between 0.0 (completely different) and 1.0 (identical).
    """
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def extract_unique_words(text: str) -> set[str]:
    """Extract unique meaningful words from text (stopwords removed)."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "just", "because", "but", "and", "or",
        "if", "while", "about", "it", "its", "this", "that", "these", "those",
        "i", "we", "you", "he", "she", "they", "me", "us", "him", "her",
        "them", "my", "our", "your", "his", "their",
    }
    words = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
    return words - stopwords


# ==================
# Convergence Scoring
# ==================


def measure_agreement_level(comments: list[Comment]) -> float:
    """
    Measure the level of agreement among agents.

    Analyzes the ratio of agreement vs disagreement indicators
    in the recent comments.

    Args:
        comments: List of comments to analyze

    Returns:
        Float between 0.0 (all disagreement) and 1.0 (all agreement)
    """
    if not comments:
        return 0.0

    total_agree = 0
    total_disagree = 0

    for c in comments:
        total_agree += count_pattern_matches(c.comment, AGREEMENT_PATTERNS)
        total_disagree += count_pattern_matches(c.comment, DISAGREEMENT_PATTERNS)

    total = total_agree + total_disagree
    if total == 0:
        return 0.5  # Neutral

    return total_agree / total


def measure_cross_reference_density(comments: list[Comment]) -> float:
    """
    Measure how much agents reference each other.

    Higher cross-referencing indicates agents are building on each
    other's ideas rather than talking past each other.

    Args:
        comments: List of comments to analyze

    Returns:
        Float between 0.0 (no references) and 1.0 (high cross-referencing)
    """
    if not comments:
        return 0.0

    total_refs = sum(len(c.references) for c in comments)
    # Normalize: expect roughly 1 reference per comment for good discussion
    return min(total_refs / max(len(comments), 1), 1.0)


def measure_topic_convergence(
    recent_comments: list[Comment],
    earlier_comments: list[Comment],
) -> float:
    """
    Measure whether the discussion is narrowing in topic scope.

    Compares unique topics between recent and earlier comments.
    High topic convergence means fewer new topics are being introduced.

    Args:
        recent_comments: Comments from the latest round(s)
        earlier_comments: Comments from earlier rounds

    Returns:
        Float between 0.0 (many new topics) and 1.0 (no new topics)
    """
    if not recent_comments:
        return 1.0
    if not earlier_comments:
        return 0.0

    recent_words = set()
    for c in recent_comments:
        recent_words |= extract_unique_words(c.comment)

    earlier_words = set()
    for c in earlier_comments:
        earlier_words |= extract_unique_words(c.comment)

    if not recent_words:
        return 1.0

    # How many words in recent are new (not seen in earlier)?
    new_words = recent_words - earlier_words
    new_ratio = len(new_words) / len(recent_words)

    # Convert to convergence score (fewer new words = higher convergence)
    return 1.0 - min(new_ratio, 1.0)


def calculate_convergence_score(
    conversation_history: list[Comment],
    current_round: int,
) -> float:
    """
    Multi-factor convergence scoring.

    Combines multiple signals to estimate how close the agents
    are to reaching consensus:

    1. Agreement level in text (40% weight)
    2. Topic convergence (20% weight)
    3. Cross-reference density (20% weight)
    4. Participation decline (20% weight)

    Args:
        conversation_history: Full conversation history
        current_round: Current round number

    Returns:
        Float between 0.0 (no convergence) and 1.0 (full convergence)
    """
    if len(conversation_history) < 2:
        return 0.0

    # Get recent vs earlier comments
    recent = [c for c in conversation_history if c.round == current_round]
    earlier = [c for c in conversation_history if c.round < current_round]

    if not recent:
        # No comments in current round - strong convergence signal
        return 0.95

    # Factor 1: Agreement level (40%)
    all_recent = conversation_history[-10:]  # Last 10 comments
    agreement_score = measure_agreement_level(all_recent)

    # Factor 2: Topic convergence (20%)
    topic_score = measure_topic_convergence(recent, earlier)

    # Factor 3: Cross-reference density (20%)
    ref_score = measure_cross_reference_density(recent)

    # Factor 4: Participation decline (20%)
    # Fewer agents commenting each round suggests natural conclusion
    if current_round >= 2 and earlier:
        prev_round_comments = [c for c in earlier if c.round == current_round - 1]
        prev_agents = len(set(c.agent for c in prev_round_comments))
        curr_agents = len(set(c.agent for c in recent))
        if prev_agents > 0:
            decline_ratio = 1.0 - (curr_agents / prev_agents)
            participation_score = max(0.0, min(decline_ratio, 1.0))
        else:
            participation_score = 0.5
    else:
        participation_score = 0.0

    # Weighted combination
    convergence = (
        0.4 * agreement_score
        + 0.2 * topic_score
        + 0.2 * ref_score
        + 0.2 * participation_score
    )

    return min(max(convergence, 0.0), 1.0)


# ==================
# Rambling Detection
# ==================


def detect_text_repetition(
    round_comments: list[Comment],
    history: list[Comment],
    similarity_threshold: float = 0.7,
) -> bool:
    """
    Detect if new comments are too similar to previous ones.

    Uses SequenceMatcher to compare text similarity between
    current round comments and earlier comments.

    Args:
        round_comments: Comments from the current round
        history: Full conversation history
        similarity_threshold: Threshold above which comments are considered repetitive

    Returns:
        True if repetitive comments are detected
    """
    if not round_comments or len(history) < 2:
        return False

    earlier = [c for c in history if c not in round_comments]
    if not earlier:
        return False

    repetition_count = 0
    for new_comment in round_comments:
        for old_comment in earlier[-15:]:  # Check against recent 15 comments
            sim = text_similarity(new_comment.comment, old_comment.comment)
            if sim > similarity_threshold:
                repetition_count += 1
                logger.debug(
                    "Repetition detected: %s (round %d) similar to %s (round %d) "
                    "with score %.2f",
                    new_comment.agent, new_comment.round,
                    old_comment.agent, old_comment.round,
                    sim,
                )
                break  # One match per new comment is enough

    # If more than 40% of new comments are repetitive
    return repetition_count > len(round_comments) * 0.4


def detect_rambling_patterns(round_comments: list[Comment]) -> bool:
    """
    Detect explicit rambling indicators in the text.

    Looks for phrases like "as I mentioned before", "to reiterate", etc.

    Args:
        round_comments: Comments from the current round

    Returns:
        True if rambling patterns are detected
    """
    if not round_comments:
        return False

    rambling_count = 0
    for c in round_comments:
        matches = count_pattern_matches(c.comment, RAMBLING_PATTERNS)
        if matches >= 2:
            rambling_count += 1

    # If more than 30% of comments show rambling patterns
    return rambling_count > len(round_comments) * 0.3


def detect_self_repetition(
    round_comments: list[Comment],
    history: list[Comment],
) -> bool:
    """
    Detect if individual agents are repeating themselves.

    Checks if any agent's current comment is very similar to their
    own previous comments (self-repetition).

    Args:
        round_comments: Comments from the current round
        history: Full conversation history

    Returns:
        True if self-repetition is detected
    """
    if not round_comments or len(history) < 2:
        return False

    self_repeat_count = 0
    for new_comment in round_comments:
        # Get this agent's previous comments
        agent_history = [
            c for c in history
            if c.agent == new_comment.agent and c != new_comment
        ]
        for old in agent_history:
            if text_similarity(new_comment.comment, old.comment) > 0.6:
                self_repeat_count += 1
                break

    # If more than 50% of agents are repeating themselves
    return self_repeat_count > len(round_comments) * 0.5


def detect_rambling(
    round_comments: list[Comment],
    full_history: list[Comment],
) -> bool:
    """
    Comprehensive rambling detection using multiple heuristics.

    Checks for:
    1. High text similarity with previous comments
    2. Explicit rambling patterns in text
    3. Agent self-repetition

    Args:
        round_comments: Comments from the current round
        full_history: Full conversation history

    Returns:
        True if rambling is detected by any heuristic
    """
    if not round_comments:
        return False

    # Heuristic 1: Text repetition
    if detect_text_repetition(round_comments, full_history):
        logger.info("Rambling detected: High text repetition")
        return True

    # Heuristic 2: Explicit rambling patterns
    if detect_rambling_patterns(round_comments):
        logger.info("Rambling detected: Explicit rambling phrases")
        return True

    # Heuristic 3: Self-repetition
    if detect_self_repetition(round_comments, full_history):
        logger.info("Rambling detected: Agent self-repetition")
        return True

    return False


# ==================
# Value-Added Measurement
# ==================


def measure_value_added(
    round_comments: list[Comment],
    full_history: list[Comment],
) -> float:
    """
    Score the value of new information in the current round.

    Combines multiple indicators:
    - New valuable content (recommendations, solutions, constraints)
    - Novelty compared to previous rounds
    - Number of new topics introduced

    Args:
        round_comments: Comments from the current round
        full_history: Full conversation history

    Returns:
        Float between 0.0 (no value) and 1.0 (high value)
    """
    if not round_comments:
        return 0.0

    # Indicator 1: Value-indicating patterns (0-1)
    total_value_matches = 0
    total_words = 0
    for c in round_comments:
        total_value_matches += count_pattern_matches(c.comment, VALUE_PATTERNS)
        total_words += len(c.comment.split())

    # Normalize by word count (value density)
    if total_words > 0:
        value_density = min(total_value_matches / (total_words / 50), 1.0)
    else:
        value_density = 0.0

    # Indicator 2: Novelty (how different from previous)
    earlier = [c for c in full_history if c not in round_comments]
    if earlier:
        current_words = set()
        for c in round_comments:
            current_words |= extract_unique_words(c.comment)

        earlier_words = set()
        for c in earlier:
            earlier_words |= extract_unique_words(c.comment)

        if current_words:
            new_words = current_words - earlier_words
            novelty = len(new_words) / len(current_words)
        else:
            novelty = 0.0
    else:
        novelty = 1.0  # First round is all new

    # Indicator 3: Cross-references (agents engaging with each other)
    engagement = measure_cross_reference_density(round_comments)

    # Weighted combination
    value = (
        0.4 * value_density
        + 0.4 * novelty
        + 0.2 * engagement
    )

    return min(max(value, 0.0), 1.0)


# ==================
# Convergence Detector
# ==================


class ConvergenceDetector:
    """
    Detects convergence in multi-agent deliberation.

    Implements the should_continue_deliberation() logic with multiple
    stopping criteria:

    1. No new comments (natural conclusion)
    2. High convergence score (consensus reached)
    3. Rambling detected (unproductive discussion)
    4. Diminishing returns (low value added)
    5. Safety limit (max rounds reached)

    Supports optional LLM-enhanced analysis when API keys are available.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        use_llm: bool = True,
    ) -> None:
        """
        Initialize the ConvergenceDetector.

        Args:
            model: LLM model identifier for enhanced analysis
            use_llm: Whether to attempt LLM-based analysis
        """
        self.model = model or os.getenv(
            "ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"
        )
        self.use_llm = use_llm
        self._logger = logging.getLogger(f"{__name__}.ConvergenceDetector")

    async def should_continue(
        self,
        round_num: int,
        round_comments: list[Comment],
        full_history: list[Comment],
        config: Optional[WorkflowConfig] = None,
    ) -> ContinueDecision:
        """
        Decide whether the deliberation should continue.

        Evaluates multiple stopping criteria in order of priority.

        Args:
            round_num: Current round number (1-based)
            round_comments: Comments from the current round
            full_history: Complete conversation history
            config: Workflow configuration with thresholds

        Returns:
            ContinueDecision with reasoning and metrics
        """
        if config is None:
            config = WorkflowConfig()

        self._logger.info(
            "Evaluating convergence: round=%d, round_comments=%d, total_history=%d",
            round_num, len(round_comments), len(full_history),
        )

        # Criterion 1: No comments = natural conclusion
        if len(round_comments) == 0:
            self._logger.info("Stop: No agents commented - natural conclusion")
            return ContinueDecision(
                should_continue=False,
                reason="No agents commented - discussion naturally concluded",
                convergence_score=1.0,
                rambling_detected=False,
                value_trend="converged",
            )

        # Criterion 2: Convergence detection
        convergence_score = calculate_convergence_score(
            full_history, round_num
        )

        if convergence_score >= config.convergence_threshold:
            self._logger.info(
                "Stop: High convergence (%.2f >= %.2f)",
                convergence_score, config.convergence_threshold,
            )
            return ContinueDecision(
                should_continue=False,
                reason=f"High convergence detected ({convergence_score:.2f})",
                convergence_score=convergence_score,
                rambling_detected=False,
                value_trend="converged",
            )

        # Criterion 3: Rambling detection
        is_rambling = detect_rambling(round_comments, full_history)
        if is_rambling:
            self._logger.info("Stop: Rambling detected")
            return ContinueDecision(
                should_continue=False,
                reason="Rambling/repetition detected - terminating discussion",
                convergence_score=convergence_score,
                rambling_detected=True,
                value_trend="declining",
            )

        # Criterion 4: Diminishing returns
        value_added = measure_value_added(round_comments, full_history)
        if value_added < config.min_value_threshold:
            self._logger.info(
                "Stop: Low value added (%.2f < %.2f)",
                value_added, config.min_value_threshold,
            )
            return ContinueDecision(
                should_continue=False,
                reason=f"Low value added ({value_added:.2f}) - discussion plateaued",
                convergence_score=convergence_score,
                rambling_detected=False,
                value_trend="plateaued",
            )

        # Criterion 5: Safety limit
        if round_num >= config.max_rounds:
            self._logger.info(
                "Stop: Max rounds reached (%d)", config.max_rounds
            )
            return ContinueDecision(
                should_continue=False,
                reason=f"Maximum rounds ({config.max_rounds}) reached",
                convergence_score=convergence_score,
                rambling_detected=False,
                value_trend="limited",
            )

        # Determine value trend
        if round_num >= 2:
            prev_round_comments = [
                c for c in full_history if c.round == round_num - 1
            ]
            prev_value = measure_value_added(
                prev_round_comments,
                [c for c in full_history if c.round < round_num - 1],
            )
            if value_added > prev_value + 0.1:
                trend = "increasing"
            elif value_added < prev_value - 0.1:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "productive"

        # Continue deliberation
        self._logger.info(
            "Continue: convergence=%.2f, value=%.2f, trend=%s",
            convergence_score, value_added, trend,
        )
        return ContinueDecision(
            should_continue=True,
            reason=f"Discussion is productive (convergence={convergence_score:.2f}, value={value_added:.2f})",
            convergence_score=convergence_score,
            rambling_detected=False,
            value_trend=trend,
        )

    async def measure_convergence(
        self,
        conversation_history: list[Comment],
        current_round: int,
    ) -> float:
        """
        Public method to measure convergence score.

        Args:
            conversation_history: Full conversation history
            current_round: Current round number

        Returns:
            Convergence score between 0.0 and 1.0
        """
        return calculate_convergence_score(conversation_history, current_round)

    async def get_round_metrics(
        self,
        round_num: int,
        round_comments: list[Comment],
        full_history: list[Comment],
    ) -> dict[str, Any]:
        """
        Get detailed metrics for a specific round.

        Args:
            round_num: Round number
            round_comments: Comments from this round
            full_history: Full conversation history

        Returns:
            Dictionary with convergence, value, rambling, and participation metrics
        """
        convergence = calculate_convergence_score(full_history, round_num)
        value = measure_value_added(round_comments, full_history)
        is_rambling = detect_rambling(round_comments, full_history)
        agreement = measure_agreement_level(round_comments)

        participating_agents = list(set(c.agent for c in round_comments))

        return {
            "round": round_num,
            "convergence_score": round(convergence, 3),
            "value_added": round(value, 3),
            "agreement_level": round(agreement, 3),
            "rambling_detected": is_rambling,
            "comments_count": len(round_comments),
            "participating_agents": participating_agents,
            "agents_count": len(participating_agents),
        }
