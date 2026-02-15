"""
Pydantic models for GitHub webhook payloads and issue data.

These models provide type-safe parsing and validation of incoming
GitHub webhook events and issue information.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from src.utils.validation import (
    strip_control_characters,
    MAX_ISSUE_TITLE_LENGTH,
    MAX_ISSUE_BODY_LENGTH,
    MAX_LABEL_COUNT,
    MAX_LABEL_LENGTH,
)


class GitHubLabel(BaseModel):
    """A GitHub issue label."""

    id: Optional[int] = None
    name: str = Field(..., max_length=MAX_LABEL_LENGTH)
    color: Optional[str] = None
    description: Optional[str] = None


class GitHubUser(BaseModel):
    """A GitHub user (issue author, sender, etc.)."""

    login: str = Field(..., max_length=256)
    id: Optional[int] = None
    avatar_url: Optional[str] = None
    html_url: Optional[str] = None
    type: Optional[str] = None


class GitHubRepository(BaseModel):
    """A GitHub repository reference."""

    id: Optional[int] = None
    name: str = Field(..., max_length=256)
    full_name: str = Field(..., max_length=512)
    owner: Optional[GitHubUser] = None
    html_url: Optional[str] = None
    description: Optional[str] = None
    private: bool = False


class GitHubIssue(BaseModel):
    """
    A GitHub issue from a webhook payload.

    Represents the core issue data extracted from the GitHub API,
    including title, body, labels, and metadata.
    """

    number: int = Field(..., ge=1, description="Issue number")
    title: str = Field(..., max_length=MAX_ISSUE_TITLE_LENGTH, description="Issue title")
    body: Optional[str] = Field(None, max_length=MAX_ISSUE_BODY_LENGTH, description="Issue body/description")
    state: Optional[str] = Field("open", description="Issue state (open/closed)")
    labels: list[str] = Field(
        default_factory=list,
        max_length=MAX_LABEL_COUNT,
        description="List of label names",
    )
    url: Optional[str] = Field(None, description="HTML URL to the issue")
    created_at: Optional[datetime] = Field(None, description="Issue creation time")
    updated_at: Optional[datetime] = Field(None, description="Last update time")

    @field_validator("title")
    @classmethod
    def clean_title(cls, v: str) -> str:
        return strip_control_characters(v.strip())

    @field_validator("body")
    @classmethod
    def clean_body(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return strip_control_characters(v)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "number": 42,
                    "title": "Add dark mode toggle to settings",
                    "body": "Users want ability to switch between light and dark themes.",
                    "state": "open",
                    "labels": ["enhancement", "ui"],
                    "url": "https://github.com/myorg/myrepo/issues/42",
                }
            ]
        }
    }


class GitHubWebhookPayload(BaseModel):
    """
    Parsed GitHub webhook payload for issue events.

    Contains the action, issue data, repository, and sender information
    as received from a GitHub webhook POST.
    """

    action: str = Field(..., description="Webhook action (opened, labeled, edited, etc.)")
    issue: GitHubIssue = Field(..., description="The issue data")
    repository: Optional[str] = Field(
        None,
        description="Repository full name (owner/repo)",
    )
    sender: Optional[str] = Field(None, description="GitHub username who triggered the event")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "action": "opened",
                    "issue": {
                        "number": 42,
                        "title": "Add dark mode toggle",
                        "body": "Need dark mode support",
                        "labels": ["enhancement"],
                    },
                    "repository": "myorg/myrepo",
                    "sender": "octocat",
                }
            ]
        }
    }

    @classmethod
    def from_raw_webhook(cls, payload: dict) -> "GitHubWebhookPayload":
        """
        Parse a raw GitHub webhook payload dictionary into a validated model.

        Handles the nested GitHub API format, extracting label names and
        flattening repository/sender info.

        Args:
            payload: Raw webhook payload dictionary.

        Returns:
            GitHubWebhookPayload: Validated payload model.

        Raises:
            ValueError: If payload is missing required 'issue' data.
        """
        if "issue" not in payload:
            raise ValueError("Webhook payload missing 'issue' field")

        raw_issue = payload["issue"]
        labels = [
            label.get("name", "") if isinstance(label, dict) else str(label)
            for label in raw_issue.get("labels", [])
        ]

        issue = GitHubIssue(
            number=raw_issue["number"],
            title=raw_issue.get("title", ""),
            body=raw_issue.get("body"),
            state=raw_issue.get("state", "open"),
            labels=labels,
            url=raw_issue.get("html_url"),
            created_at=raw_issue.get("created_at"),
            updated_at=raw_issue.get("updated_at"),
        )

        return cls(
            action=payload.get("action", "unknown"),
            issue=issue,
            repository=payload.get("repository", {}).get("full_name")
            if isinstance(payload.get("repository"), dict)
            else payload.get("repository"),
            sender=payload.get("sender", {}).get("login")
            if isinstance(payload.get("sender"), dict)
            else payload.get("sender"),
        )
