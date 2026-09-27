"""Read-only GitHub tools for a local AI agent.

Configure these environment variables before use:
    GITHUB_TOKEN - optional for public repositories, recommended for rate limits
    GITHUB_API_URL - optional, defaults to https://api.github.com

The functions return JSON-serializable dictionaries and raise GitHubAPIError
for HTTP/API failures so an agent can report errors clearly.
"""

from __future__ import annotations

import base64
import os
from typing import Any

import requests


GITHUB_API_URL = os.getenv("GITHUB_API_URL", "https://api.github.com").rstrip("/")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
_TIMEOUT_SECONDS = 20


class GitHubAPIError(RuntimeError):
    """Raised when the GitHub API cannot fulfill a request."""


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "local-agentic-ai",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def _request(path: str, params: dict[str, Any] | None = None) -> Any:
    """Perform a GET request and return decoded JSON."""
    url = f"{GITHUB_API_URL}/{path.lstrip('/')}"
    try:
        response = requests.get(
            url,
            headers=_headers(),
            params=params,
            timeout=_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise GitHubAPIError(f"Could not connect to GitHub: {exc}") from exc

    if not response.ok:
        try:
            details = response.json().get("message", response.text)
        except ValueError:
            details = response.text
        raise GitHubAPIError(f"GitHub API error ({response.status_code}): {details}")

    try:
        return response.json()
    except ValueError as exc:
        raise GitHubAPIError("GitHub returned an invalid JSON response") from exc


def _repo_path(owner: str, repo: str) -> str:
    owner = owner.strip()
    repo = repo.strip()
    if not owner or not repo or "/" in owner or "/" in repo:
        raise ValueError("owner and repo must be non-empty names without '/'")
    return f"repos/{owner}/{repo}"


def get_repository(owner: str, repo: str) -> dict[str, Any]:
    """Get basic metadata for a repository."""
    data = _request(_repo_path(owner, repo))
    return {
        "name": data["full_name"],
        "description": data.get("description"),
        "private": data.get("private", False),
        "default_branch": data.get("default_branch"),
        "stars": data.get("stargazers_count", 0),
        "forks": data.get("forks_count", 0),
        "open_issues": data.get("open_issues_count", 0),
        "language": data.get("language"),
        "url": data.get("html_url"),
    }


def list_issues(owner: str, repo: str, state: str = "open", limit: int = 10) -> list[dict[str, Any]]:
    """List issues, excluding pull requests, with a small safety limit."""
    if state not in {"open", "closed", "all"}:
        raise ValueError("state must be open, closed, or all")
    limit = max(1, min(int(limit), 50))
    data = _request(
        f"{_repo_path(owner, repo)}/issues",
        {"state": state, "per_page": limit, "page": 1},
    )
    return [
        {
            "number": item["number"],
            "title": item["title"],
            "state": item["state"],
            "author": item.get("user", {}).get("login"),
            "labels": [label["name"] for label in item.get("labels", [])],
            "comments": item.get("comments", 0),
            "url": item["html_url"],
        }
        for item in data
        if "pull_request" not in item
    ]


def list_pull_requests(
    owner: str,
    repo: str,
    state: str = "open",
    limit: int = 10,
) -> list[dict[str, Any]]:
    """List pull requests."""
    if state not in {"open", "closed", "all"}:
        raise ValueError("state must be open, closed, or all")
    limit = max(1, min(int(limit), 50))
    data = _request(
        f"{_repo_path(owner, repo)}/pulls",
        {"state": state, "per_page": limit, "page": 1, "sort": "updated"},
    )
    return [
        {
            "number": item["number"],
            "title": item["title"],
            "state": item["state"],
            "draft": item.get("draft", False),
            "author": item.get("user", {}).get("login"),
            "head": item.get("head", {}).get("ref"),
            "base": item.get("base", {}).get("ref"),
            "url": item["html_url"],
        }
        for item in data
    ]


def get_issue(owner: str, repo: str, number: int) -> dict[str, Any]:
    """Get an issue and its comments."""
    if int(number) < 1:
        raise ValueError("number must be positive")
    issue = _request(f"{_repo_path(owner, repo)}/issues/{int(number)}")
    comments = _request(
        f"{_repo_path(owner, repo)}/issues/{int(number)}/comments",
        {"per_page": 50, "page": 1},
    )
    return {
        "number": issue["number"],
        "title": issue["title"],
        "body": issue.get("body"),
        "state": issue["state"],
        "author": issue.get("user", {}).get("login"),
        "labels": [label["name"] for label in issue.get("labels", [])],
        "url": issue["html_url"],
        "comments": [
            {
                "author": comment.get("user", {}).get("login"),
                "body": comment.get("body"),
                "created_at": comment.get("created_at"),
            }
            for comment in comments
        ],
    }


def get_file(owner: str, repo: str, path: str, ref: str | None = None) -> dict[str, Any]:
    """Read a text file from a repository."""
    if not path.strip():
        raise ValueError("path cannot be empty")
    params = {"ref": ref} if ref else None
    data = _request(f"{_repo_path(owner, repo)}/contents/{path.lstrip('/')}", params)
    if isinstance(data, list) or data.get("type") != "file":
        raise GitHubAPIError(f"'{path}' is not a file")

    try:
        content = base64.b64decode(data.get("content", "")).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise GitHubAPIError(f"Could not decode '{path}' as UTF-8 text") from exc
    return {"path": data["path"], "sha": data["sha"], "content": content, "url": data["html_url"]}


TOOLS = {
    "get_repository": get_repository,
    "list_issues": list_issues,
    "list_pull_requests": list_pull_requests,
    "get_issue": get_issue,
    "get_file": get_file,
}

