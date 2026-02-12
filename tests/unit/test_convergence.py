"""
Tests for the convergence detection logic.

Tests cover:
- Convergence score calculation
- Agreement level measurement
- Rambling and repetition detection
- Value-added measurement
- ConvergenceDetector.should_continue() with all stop conditions
- Topic convergence tracking
- Cross-reference density measurement
"""

from datetime import datetime

import pytest

from src.models.workflow import Comment, ContinueDecision, WorkflowConfig
from src.orchestration.convergence import (
    ConvergenceDetector,
    calculate_convergence_score,
    count_pattern_matches,
    detect_rambling,
    detect_rambling_patterns,
    detect_self_repetition,
    detect_text_repetition,
    extract_unique_words,
    measure_agreement_level,
    measure_cross_reference_density,
    measure_topic_convergence,
    measure_value_added,
    text_similarity,
    AGREEMENT_PATTERNS,
    DISAGREEMENT_PATTERNS,
)


# ==================
# Fixtures
# ==================


def make_comment(
    round: int,
    agent: str,
    comment: str,
    references: list[str] | None = None,
) -> Comment:
    """Helper to create Comment objects for testing."""
    return Comment(
        round=round,
        agent=agent,
        comment=comment,
        references=references or [],
        timestamp=datetime.utcnow(),
    )


@pytest.fixture
def high_agreement_history():
    """Conversation history with strong consensus."""
    return [
        make_comment(1, "agent_a", "I recommend using a component-based architecture."),
        make_comment(1, "agent_b", "I agree with the component approach. We should also add testing."),
        make_comment(1, "agent_c", "I support this plan. Let's proceed with components.", references=["agent_a"]),
        make_comment(2, "agent_a", "Building on agent_c's point, I concur we need tests too.", references=["agent_c"]),
        make_comment(2, "agent_b", "Yes, exactly. The consensus is clear - components with tests.", references=["agent_a", "agent_c"]),
        make_comment(2, "agent_c", "I second that. We're all aligned on this approach.", references=["agent_a", "agent_b"]),
    ]


@pytest.fixture
def low_agreement_history():
    """Conversation history with significant disagreement."""
    return [
        make_comment(1, "agent_a", "We should use microservices for this feature."),
        make_comment(1, "agent_b", "However, I disagree. A monolith would be simpler and faster."),
        make_comment(1, "agent_c", "I have concerns about both approaches. Alternatively, we could try a hybrid."),
        make_comment(2, "agent_a", "But the monolith will cause scaling issues. I oppose that idea."),
        make_comment(2, "agent_b", "On the other hand, microservices add complexity. I push back on that."),
        make_comment(2, "agent_c", "Instead of either extreme, the risk with microservices is too high."),
    ]


@pytest.fixture
def rambling_history():
    """Conversation history with repetitive content."""
    base = "We should implement a caching layer with Redis for the user session data."
    return [
        make_comment(1, "agent_a", base),
        make_comment(1, "agent_b", "Good idea about the caching layer."),
        make_comment(2, "agent_a", f"As I said before, {base} To reiterate, Redis is the best choice."),
        make_comment(2, "agent_b", f"As previously mentioned, {base}"),
        make_comment(3, "agent_a", f"Again, {base} As I mentioned earlier, this is critical."),
        make_comment(3, "agent_b", f"Once more, {base} As we previously discussed."),
    ]


@pytest.fixture
def productive_history():
    """Conversation history with productive new information each round."""
    return [
        make_comment(1, "agent_a", "I recommend implementing a REST API with OpenAPI documentation."),
        make_comment(1, "agent_b", "We should consider authentication using JWT tokens for security."),
        make_comment(1, "agent_c", "The database schema needs to support multi-tenancy with proper isolation."),
        make_comment(2, "agent_a", "Building on the security concern, we need RBAC for authorization.", references=["agent_b"]),
        make_comment(2, "agent_b", "I propose rate limiting at the API gateway level to prevent abuse.", references=["agent_a"]),
        make_comment(2, "agent_c", "Consider implementing event sourcing for audit trail requirements.", references=["agent_a", "agent_b"]),
    ]


@pytest.fixture
def detector():
    """Create a ConvergenceDetector with LLM disabled."""
    return ConvergenceDetector(use_llm=False)


@pytest.fixture
def default_config():
    """Default workflow configuration."""
    return WorkflowConfig()


# ==================
# Text Analysis Helper Tests
# ==================


class TestTextAnalysisHelpers:
    """Tests for text analysis utility functions."""

    def test_text_similarity_identical(self):
        """Identical texts have similarity 1.0."""
        assert text_similarity("hello world", "hello world") == 1.0

    def test_text_similarity_different(self):
        """Very different texts have low similarity."""
        sim = text_similarity("hello world", "goodbye universe completely different")
        assert sim < 0.5

    def test_text_similarity_empty(self):
        """Empty strings have similarity 0.0."""
        assert text_similarity("", "hello") == 0.0
        assert text_similarity("hello", "") == 0.0

    def test_text_similarity_partial(self):
        """Partially similar texts have medium similarity."""
        sim = text_similarity("I recommend caching", "I recommend indexing")
        assert 0.3 < sim < 0.9

    def test_extract_unique_words(self):
        """Extract meaningful words excluding stopwords."""
        words = extract_unique_words("The quick brown fox jumps over the lazy dog")
        assert "quick" in words
        assert "brown" in words
        assert "the" not in words
        assert "over" not in words

    def test_count_agreement_patterns(self):
        """Count agreement patterns in text."""
        text = "I agree with this approach. I second that idea. Good point."
        count = count_pattern_matches(text, AGREEMENT_PATTERNS)
        assert count >= 2

    def test_count_disagreement_patterns(self):
        """Count disagreement patterns in text."""
        text = "However, I disagree with this. There's a concern about scalability."
        count = count_pattern_matches(text, DISAGREEMENT_PATTERNS)
        assert count >= 2


# ==================
# Agreement Level Tests
# ==================


class TestAgreementLevel:
    """Tests for agreement level measurement."""

    def test_high_agreement(self, high_agreement_history):
        """High agreement history scores high."""
        score = measure_agreement_level(high_agreement_history)
        assert score > 0.6

    def test_low_agreement(self, low_agreement_history):
        """Low agreement history scores low."""
        score = measure_agreement_level(low_agreement_history)
        assert score < 0.5

    def test_empty_comments(self):
        """Empty comment list returns 0.0."""
        assert measure_agreement_level([]) == 0.0

    def test_neutral_comments(self):
        """Comments without clear agreement/disagreement return 0.5."""
        comments = [
            make_comment(1, "agent_a", "The system uses a database."),
            make_comment(1, "agent_b", "There are 42 endpoints in total."),
        ]
        score = measure_agreement_level(comments)
        assert score == 0.5


# ==================
# Cross-Reference Density Tests
# ==================


class TestCrossReferenceDensity:
    """Tests for cross-reference density measurement."""

    def test_high_cross_references(self):
        """Comments with many references score high."""
        comments = [
            make_comment(2, "a", "Building on b's point", references=["b", "c"]),
            make_comment(2, "b", "Responding to a", references=["a"]),
            make_comment(2, "c", "Regarding a and b", references=["a", "b"]),
        ]
        score = measure_cross_reference_density(comments)
        assert score > 0.8

    def test_no_cross_references(self):
        """Comments without references score 0."""
        comments = [
            make_comment(1, "a", "My independent opinion."),
            make_comment(1, "b", "My own separate view."),
        ]
        score = measure_cross_reference_density(comments)
        assert score == 0.0

    def test_empty_comments(self):
        """Empty list returns 0.0."""
        assert measure_cross_reference_density([]) == 0.0


# ==================
# Topic Convergence Tests
# ==================


class TestTopicConvergence:
    """Tests for topic convergence measurement."""

    def test_converging_topics(self):
        """When recent topics mostly reuse earlier words, convergence is higher."""
        earlier = [
            make_comment(1, "a", "We need database indexing and caching strategies for performance."),
            make_comment(1, "b", "Authentication and authorization are critical for security."),
            make_comment(1, "c", "The caching layer should handle database query results."),
        ]
        recent = [
            make_comment(2, "a", "The database indexing and caching should be implemented together."),
        ]
        score = measure_topic_convergence(recent, earlier)
        assert score > 0.3  # Some convergence since words overlap

    def test_diverging_topics(self):
        """When recent introduces many new topics, convergence is low."""
        earlier = [
            make_comment(1, "a", "We should implement a simple REST API."),
        ]
        recent = [
            make_comment(2, "a", "Actually let's consider blockchain technology with smart contracts, "
                        "NFT marketplaces, and decentralized finance protocols."),
        ]
        score = measure_topic_convergence(recent, earlier)
        assert score < 0.7

    def test_empty_recent(self):
        """No recent comments means full convergence."""
        earlier = [make_comment(1, "a", "Some comment.")]
        assert measure_topic_convergence([], earlier) == 1.0

    def test_no_earlier(self):
        """No earlier comments means no convergence possible."""
        recent = [make_comment(2, "a", "Some comment.")]
        assert measure_topic_convergence(recent, []) == 0.0


# ==================
# Convergence Score Tests
# ==================


class TestConvergenceScore:
    """Tests for overall convergence score calculation."""

    def test_high_convergence_conversation(self, high_agreement_history):
        """High agreement history scores high convergence."""
        score = calculate_convergence_score(high_agreement_history, 2)
        assert score > 0.4

    def test_low_convergence_conversation(self, low_agreement_history):
        """Low agreement history scores low convergence."""
        score = calculate_convergence_score(low_agreement_history, 2)
        assert score < 0.5

    def test_empty_history(self):
        """Empty history returns 0.0."""
        assert calculate_convergence_score([], 1) == 0.0

    def test_single_comment(self):
        """Single comment returns 0.0 (not enough data)."""
        history = [make_comment(1, "a", "Hello world")]
        assert calculate_convergence_score(history, 1) == 0.0

    def test_no_comments_in_current_round(self):
        """No comments in current round indicates strong convergence."""
        history = [
            make_comment(1, "a", "Initial comment"),
            make_comment(1, "b", "Another comment"),
        ]
        score = calculate_convergence_score(history, 2)
        assert score > 0.9

    def test_score_bounded(self, productive_history):
        """Score is always between 0.0 and 1.0."""
        score = calculate_convergence_score(productive_history, 2)
        assert 0.0 <= score <= 1.0


# ==================
# Rambling Detection Tests
# ==================


class TestRamblingDetection:
    """Tests for rambling and repetition detection."""

    def test_detect_text_repetition(self):
        """Detects when comments repeat earlier content."""
        base = "We need to implement comprehensive error handling with retry logic and circuit breakers."
        history = [
            make_comment(1, "a", base),
            make_comment(1, "b", "Good architecture proposal."),
        ]
        round_comments = [
            make_comment(2, "a", base),  # Exact repeat
            make_comment(2, "b", base),  # Exact repeat
        ]
        assert detect_text_repetition(round_comments, history + round_comments)

    def test_no_text_repetition(self, productive_history):
        """Productive comments are not flagged as repetitive."""
        round_comments = [c for c in productive_history if c.round == 2]
        assert not detect_text_repetition(round_comments, productive_history)

    def test_detect_rambling_patterns(self):
        """Detects explicit rambling phrases."""
        comments = [
            make_comment(2, "a", "As I said before, we need caching. To reiterate, Redis is essential."),
            make_comment(2, "b", "As previously mentioned, as I mentioned earlier, this was discussed."),
        ]
        assert detect_rambling_patterns(comments)

    def test_no_rambling_patterns(self, productive_history):
        """Productive comments don't have rambling patterns."""
        round_comments = [c for c in productive_history if c.round == 2]
        assert not detect_rambling_patterns(round_comments)

    def test_detect_self_repetition(self):
        """Detects when agents repeat their own previous comments."""
        history = [
            make_comment(1, "agent_a", "We need to implement database indexing for query performance."),
            make_comment(1, "agent_b", "Authentication should use OAuth 2.0 with PKCE flow."),
        ]
        round_comments = [
            make_comment(2, "agent_a", "We need to implement database indexing for query performance."),
            make_comment(2, "agent_b", "Authentication should use OAuth 2.0 with PKCE flow."),
        ]
        assert detect_self_repetition(round_comments, history + round_comments)

    def test_no_self_repetition(self, productive_history):
        """Different comments from same agent are not self-repetition."""
        round_comments = [c for c in productive_history if c.round == 2]
        assert not detect_self_repetition(round_comments, productive_history)

    def test_detect_rambling_comprehensive(self, rambling_history):
        """Comprehensive rambling detection catches repetitive discussion."""
        round3 = [c for c in rambling_history if c.round == 3]
        assert detect_rambling(round3, rambling_history)

    def test_no_rambling_productive(self, productive_history):
        """Productive discussion is not flagged as rambling."""
        round2 = [c for c in productive_history if c.round == 2]
        assert not detect_rambling(round2, productive_history)

    def test_empty_round_no_rambling(self):
        """Empty round is not rambling."""
        assert not detect_rambling([], [make_comment(1, "a", "test")])


# ==================
# Value-Added Measurement Tests
# ==================


class TestValueAdded:
    """Tests for value-added measurement."""

    def test_high_value_comments(self):
        """Comments with recommendations and solutions score high."""
        round_comments = [
            make_comment(2, "a", "I recommend implementing a caching strategy with Redis. "
                        "This approach would solve the performance constraint. "
                        "Consider implementing circuit breakers for resilience."),
        ]
        history = [
            make_comment(1, "b", "The system is experiencing slow queries."),
        ] + round_comments
        value = measure_value_added(round_comments, history)
        assert value > 0.3

    def test_low_value_repetitive_comments(self):
        """Repetitive comments with no new info score low."""
        base_text = "The system needs to handle user authentication properly."
        history = [
            make_comment(1, "a", base_text),
            make_comment(1, "b", base_text),
        ]
        round_comments = [
            make_comment(2, "a", base_text),
        ]
        value = measure_value_added(round_comments, history + round_comments)
        assert value < 0.5

    def test_first_round_high_value(self):
        """First round comments are inherently high value (all new)."""
        round_comments = [
            make_comment(1, "a", "We should implement microservices architecture."),
        ]
        value = measure_value_added(round_comments, round_comments)
        assert value > 0.3

    def test_empty_round_zero_value(self):
        """Empty round has zero value."""
        assert measure_value_added([], []) == 0.0


# ==================
# ConvergenceDetector Tests
# ==================


class TestConvergenceDetector:
    """Tests for the ConvergenceDetector class."""

    @pytest.mark.asyncio
    async def test_stop_on_no_comments(self, detector, default_config):
        """Should stop when no agents comment."""
        decision = await detector.should_continue(
            round_num=3,
            round_comments=[],
            full_history=[make_comment(1, "a", "test"), make_comment(2, "b", "test")],
            config=default_config,
        )
        assert decision.should_continue is False
        assert "naturally concluded" in decision.reason.lower()
        assert decision.convergence_score == 1.0

    @pytest.mark.asyncio
    async def test_stop_on_high_convergence(self, detector):
        """Should stop when convergence threshold is reached."""
        config = WorkflowConfig(convergence_threshold=0.3)  # Low threshold for testing
        history = [
            make_comment(1, "a", "I agree we should proceed with the plan.", references=["b"]),
            make_comment(1, "b", "I concur, let's align on this approach.", references=["a"]),
            make_comment(1, "c", "Yes, I support this. Exactly right.", references=["a", "b"]),
        ]
        # Round 2: no comments triggers high convergence
        decision = await detector.should_continue(
            round_num=2,
            round_comments=[],
            full_history=history,
            config=config,
        )
        assert decision.should_continue is False

    @pytest.mark.asyncio
    async def test_stop_on_rambling(self, detector, default_config, rambling_history):
        """Should stop when rambling is detected."""
        round3 = [c for c in rambling_history if c.round == 3]
        decision = await detector.should_continue(
            round_num=3,
            round_comments=round3,
            full_history=rambling_history,
            config=default_config,
        )
        assert decision.should_continue is False
        assert decision.rambling_detected is True

    @pytest.mark.asyncio
    async def test_stop_on_max_rounds(self, detector):
        """Should stop when max rounds is reached."""
        config = WorkflowConfig(max_rounds=2)
        history = [
            make_comment(1, "a", "First recommendation about the architecture."),
            make_comment(1, "b", "Second concern about security implications."),
        ]
        round_comments = [
            make_comment(2, "a", "New proposal for implementing the feature with a different strategy."),
        ]
        decision = await detector.should_continue(
            round_num=2,
            round_comments=round_comments,
            full_history=history + round_comments,
            config=config,
        )
        assert decision.should_continue is False
        assert "maximum rounds" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_continue_productive_discussion(self, detector, default_config, productive_history):
        """Should continue when discussion is productive."""
        round2 = [c for c in productive_history if c.round == 2]
        decision = await detector.should_continue(
            round_num=2,
            round_comments=round2,
            full_history=productive_history,
            config=default_config,
        )
        assert decision.should_continue is True
        assert decision.rambling_detected is False
        assert 0.0 <= decision.convergence_score <= 1.0

    @pytest.mark.asyncio
    async def test_decision_has_valid_fields(self, detector, default_config, productive_history):
        """ContinueDecision has all required fields."""
        round2 = [c for c in productive_history if c.round == 2]
        decision = await detector.should_continue(
            round_num=2,
            round_comments=round2,
            full_history=productive_history,
            config=default_config,
        )
        assert isinstance(decision, ContinueDecision)
        assert isinstance(decision.should_continue, bool)
        assert isinstance(decision.reason, str)
        assert 0.0 <= decision.convergence_score <= 1.0
        assert isinstance(decision.rambling_detected, bool)
        assert decision.value_trend in {
            "increasing", "decreasing", "stable", "unknown",
            "converged", "declining", "plateaued", "limited", "productive",
        }

    @pytest.mark.asyncio
    async def test_default_config_used(self, detector, productive_history):
        """Default config is used when none is provided."""
        round2 = [c for c in productive_history if c.round == 2]
        decision = await detector.should_continue(
            round_num=2,
            round_comments=round2,
            full_history=productive_history,
        )
        assert isinstance(decision, ContinueDecision)


class TestConvergenceDetectorMetrics:
    """Tests for ConvergenceDetector metric methods."""

    @pytest.mark.asyncio
    async def test_measure_convergence(self, detector, productive_history):
        """measure_convergence returns a valid score."""
        score = await detector.measure_convergence(productive_history, 2)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_get_round_metrics(self, detector, productive_history):
        """get_round_metrics returns detailed metrics."""
        round2 = [c for c in productive_history if c.round == 2]
        metrics = await detector.get_round_metrics(2, round2, productive_history)

        assert "round" in metrics
        assert metrics["round"] == 2
        assert "convergence_score" in metrics
        assert "value_added" in metrics
        assert "agreement_level" in metrics
        assert "rambling_detected" in metrics
        assert "comments_count" in metrics
        assert "participating_agents" in metrics
        assert "agents_count" in metrics
        assert metrics["comments_count"] == 3
        assert metrics["agents_count"] == 3


class TestConvergenceDetectorInit:
    """Tests for ConvergenceDetector initialization."""

    def test_default_init(self):
        """Default initialization."""
        detector = ConvergenceDetector()
        assert detector.model == "claude-sonnet-4-5-20250929"
        assert detector.use_llm is True

    def test_custom_init(self):
        """Custom initialization."""
        detector = ConvergenceDetector(model="custom-model", use_llm=False)
        assert detector.model == "custom-model"
        assert detector.use_llm is False
