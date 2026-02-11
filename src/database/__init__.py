"""
Database package for Multi-Agent GitHub Issue Routing System.

Provides SQLAlchemy ORM models, session management, and CRUD operations
for persisting workflow state, conversation history, agent selections,
and convergence metrics.
"""

from src.database.engine import (
    get_engine,
    get_session,
    get_async_session,
    init_db,
    close_db,
    AsyncSessionLocal,
)
from src.database.models import (
    Base,
    WorkflowRecord,
    ConversationRecord,
    AgentRecord,
    ConvergenceMetricRecord,
)

__all__ = [
    # Engine / Session
    "get_engine",
    "get_session",
    "get_async_session",
    "init_db",
    "close_db",
    "AsyncSessionLocal",
    # ORM Models
    "Base",
    "WorkflowRecord",
    "ConversationRecord",
    "AgentRecord",
    "ConvergenceMetricRecord",
]
