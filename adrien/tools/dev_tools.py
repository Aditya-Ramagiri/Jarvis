"""Coding and development tools (spec 7.2).

Git commands run against a *resolved* repository: the LLM may name a repo
loosely ("the jarvis one") so `resolve_repo` looks under the configured
workspace roots rather than demanding a full path. Everything reports
structured success/failure, because "did the push actually land" is exactly
the kind of thing Adrien must not get cheerfully wrong (spec 7.9).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from adrien.config import env_str
from adrien.logging_setup import get_logger
from adrien.tools._shell import run
from adrien.tools.registry import ToolResult, tool

log = get_logger(__name__)

# Where to look for repositories named loosely. Overridable via ADRIEN_REPO_DIRS
# (colon-separated), because everyone keeps their code somewhere different.
def _workspace_roots() -> list[Path]:
    configured = env_str("ADRIEN_REPO_DIRS")
    if configured:
        return [Path(part).expanduser() for part in configured.split(":") if part]
    home = Path.home()
    return [home / "code", home / "dev", home / "Developer", home / "projects", home / "src", home]


def resolve_repo(repo: str = "") -> tuple[Path | None, str]:
    """Turn a loose repo name into a path. Returns (path, error)."""
    if not repo:
        cwd = Path(os.getcwd())
        return (cwd, "") if (cwd / ".git").exists() else (
            None, "no repository named, and Adrien is not running inside one"
        )

    candidate = Path(repo).expanduser()
    if candidate.is_dir() and (candidate / ".git").exists():
        return candidate, ""

    name = candidate.name.lower()
    for root in _workspace_roots():
        if not root.is_dir():
            continue
        try:
            for child in root.iterdir():
                if child.is_dir() and child.name.lower() == name and (child / ".git").exists():
                    return child, ""
        except PermissionError:
            continue
    return None, f"could not find a git repository called {repo}"


def _git(repo: str, *args: str, timeout: float = 30.0):
    path, error = resolve_repo(repo)
    if path is None:
        return None, ToolResult.failure(error)
    return run(["git", *args], cwd=path, timeout=timeout), None


@tool(category="dev")
def git_status(repo: str = "") -> ToolResult:
    """Check whether a git repository has uncommitted changes and how it sits
    against its remote.

    Args:
        repo: Repository name or path. Defaults to the current directory.
    """
    result, failure = _git(repo, "status", "--porcelain=v1", "--branch")
    if failure:
        return failure
    if not result.ok:
        return ToolResult.failure(f"git status failed: {result.output}")

    lines = result.stdout.splitlines()
    branch_line = lines[0] if lines and lines[0].startswith("##") else ""
    changes = [line for line in lines[1:] if line.strip()]
    branch = branch_line[3:].split("...")[0] if branch_line else "unknown"

    ahead = behind = 0
    if "[" in branch_line:
        marker = branch_line[branch_line.index("[") + 1: branch_line.rindex("]")]
        for part in marker.split(","):
            part = part.strip()
            if part.startswith("ahead "):
                ahead = int(part.split()[1])
            elif part.startswith("behind "):
                behind = int(part.split()[1])

    return ToolResult.success(
        {
            "branch": branch,
            "changed_files": len(changes),
            "files": [line[3:] for line in changes[:20]],
            "ahead": ahead,
            "behind": behind,
            "clean": not changes,
        },
        speak=(
            f"{branch} is clean" if not changes
            else f"{len(changes)} changed file{'s' if len(changes) != 1 else ''} on {branch}"
        ),
    )


@tool(
    category="dev",
    destructive=True,
    confirm="Commit everything staged with the message: {message}?",
)
def git_commit(message: str, repo: str = "", add_all: bool = True) -> ToolResult:
    """Commit changes in a git repository.

    Args:
        message: The commit message.
        repo: Repository name or path. Defaults to the current directory.
        add_all: Stage every modified and new file before committing.
    """
    path, error = resolve_repo(repo)
    if path is None:
        return ToolResult.failure(error)

    if add_all:
        staged = run(["git", "add", "-A"], cwd=path)
        if not staged.ok:
            return ToolResult.failure(f"could not stage changes: {staged.output}")

    result = run(["git", "commit", "-m", message], cwd=path)
    if not result.ok:
        if "nothing to commit" in result.output.lower():
            return ToolResult.failure("there is nothing to commit")
        return ToolResult.failure(f"commit failed: {result.output}")

    sha = run(["git", "rev-parse", "--short", "HEAD"], cwd=path).stdout.strip()
    return ToolResult.success({"sha": sha, "message": message}, speak=f"committed as {sha}")


@tool(category="dev", destructive=True, confirm="Push {repo} to its remote?")
def git_push(repo: str = "", remote: str = "origin", branch: str = "") -> ToolResult:
    """Push commits to a remote. Reports the real outcome, including rejections.

    Args:
        repo: Repository name or path.
        remote: Remote name. Defaults to origin.
        branch: Branch to push. Defaults to the current branch.
    """
    path, error = resolve_repo(repo)
    if path is None:
        return ToolResult.failure(error)

    if not branch:
        head = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=path)
        if not head.ok:
            return ToolResult.failure(f"could not work out the current branch: {head.output}")
        branch = head.stdout.strip()

    result = run(["git", "push", "-u", remote, branch], cwd=path, timeout=120)
    if not result.ok:
        output = result.output
        if "rejected" in output and "non-fast-forward" in output:
            return ToolResult.failure(
                f"{branch} was rejected - the remote has commits you do not have. Pull first."
            )
        if "Permission denied" in output or "Authentication failed" in output:
            return ToolResult.failure("the remote refused the credentials")
        return ToolResult.failure(f"push failed: {output[:300]}")
    return ToolResult.success({"branch": branch, "remote": remote},
                              speak=f"pushed {branch} to {remote}")


@tool(category="dev")
def git_pull(repo: str = "", remote: str = "origin", branch: str = "") -> ToolResult:
    """Pull the latest commits from a remote.

    Args:
        repo: Repository name or path.
        remote: Remote name. Defaults to origin.
        branch: Branch to pull. Defaults to the current branch.
    """
    path, error = resolve_repo(repo)
    if path is None:
        return ToolResult.failure(error)

    command = ["git", "pull", remote] + ([branch] if branch else [])
    result = run(command, cwd=path, timeout=120)
    if not result.ok:
        if "conflict" in result.output.lower():
            return ToolResult.failure("the pull hit merge conflicts that need resolving by hand")
        return ToolResult.failure(f"pull failed: {result.output[:300]}")
    if "Already up to date" in result.stdout:
        return ToolResult.success({"updated": False}, speak="already up to date")
    return ToolResult.success({"updated": True, "detail": result.stdout[-400:]}, speak="pulled")


@tool(
    category="dev",
    irreversible=True,
    confirm="Run the script at {path}?",
    timeout=120.0,
)
def run_script(path: str, args: str = "", cwd: str = "") -> ToolResult:
    """Execute a script and report whether it succeeded, with its output.

    Args:
        path: Path to the script.
        args: Space-separated arguments to pass to it.
        cwd: Directory to run it in. Defaults to the script's own directory.
    """
    import shlex

    script = Path(path).expanduser()
    if not script.exists():
        return ToolResult.failure(f"there is no script at {script}")

    interpreter: list[str] = []
    if script.suffix == ".py":
        interpreter = ["python3"]
    elif script.suffix in (".sh", ".bash"):
        interpreter = ["bash"]
    elif not os.access(script, os.X_OK):
        return ToolResult.failure(f"{script.name} is not executable and Adrien cannot tell how to run it")

    command = interpreter + [str(script)] + shlex.split(args)
    result = run(command, cwd=cwd or script.parent, timeout=110)

    tail = (result.stdout or result.stderr).strip().splitlines()[-15:]
    if not result.ok:
        return ToolResult.failure(
            f"{script.name} exited with code {result.code}",
            data={"output": "\n".join(tail)},
        )
    return ToolResult.success(
        {"exit_code": 0, "output": "\n".join(tail)},
        speak=f"{script.name} finished cleanly",
    )


# --------------------------------------------------------------------------
# GitHub
# --------------------------------------------------------------------------
def _github_client():
    """Authenticated GitHub client, or (None, error)."""
    token = env_str("GITHUB_TOKEN")
    if not token:
        return None, "GITHUB_TOKEN is not set in .env"
    try:
        from github import Auth, Github

        return Github(auth=Auth.Token(token)), ""
    except ImportError:
        return None, "PyGithub is not installed"


@tool(category="dev", requires_env=["GITHUB_TOKEN"])
def check_github_notifications(limit: int = 10) -> ToolResult:
    """List unread GitHub notifications.

    Args:
        limit: How many to return at most.
    """
    client, error = _github_client()
    if client is None:
        return ToolResult.failure(error)

    try:
        items: list[dict[str, Any]] = []
        for notification in client.get_user().get_notifications(all=False)[:limit]:
            items.append({
                "repo": notification.repository.full_name,
                "title": notification.subject.title,
                "type": notification.subject.type,
                "reason": notification.reason,
            })
    except Exception as exc:
        return ToolResult.failure(f"GitHub rejected the request: {exc}")

    if not items:
        return ToolResult.success({"count": 0, "items": []}, speak="no unread notifications")
    return ToolResult.success(
        {"count": len(items), "items": items},
        speak=f"{len(items)} unread notification{'s' if len(items) != 1 else ''}",
    )


@tool(category="dev", requires_env=["GITHUB_TOKEN"])
def check_github_prs(repo: str, state: str = "open", limit: int = 10) -> ToolResult:
    """List pull requests on a repository with their review and merge status.

    Args:
        repo: Repository as owner/name.
        state: open, closed or all.
        limit: How many to return at most.
    """
    client, error = _github_client()
    if client is None:
        return ToolResult.failure(error)

    try:
        repository = client.get_repo(repo)
        items = []
        for pull in repository.get_pulls(state=state, sort="updated", direction="desc")[:limit]:
            items.append({
                "number": pull.number,
                "title": pull.title,
                "author": pull.user.login if pull.user else "unknown",
                "draft": pull.draft,
                "mergeable_state": pull.mergeable_state,
                "url": pull.html_url,
            })
    except Exception as exc:
        return ToolResult.failure(f"could not read pull requests for {repo}: {exc}")

    return ToolResult.success(
        {"repo": repo, "count": len(items), "pull_requests": items},
        speak=f"{len(items)} {state} pull request{'s' if len(items) != 1 else ''} on {repo}",
    )


@tool(category="dev", requires_env=["GITHUB_TOKEN"])
def check_github_actions_status(repo: str, branch: str = "") -> ToolResult:
    """Check the most recent CI run on a repository.

    Args:
        repo: Repository as owner/name.
        branch: Restrict to one branch. Defaults to all branches.
    """
    client, error = _github_client()
    if client is None:
        return ToolResult.failure(error)

    try:
        repository = client.get_repo(repo)
        runs = repository.get_workflow_runs(branch=branch) if branch else repository.get_workflow_runs()
        latest = runs[0] if runs.totalCount else None
    except Exception as exc:
        return ToolResult.failure(f"could not read CI runs for {repo}: {exc}")

    if latest is None:
        return ToolResult.success({"runs": 0}, speak=f"no CI runs found on {repo}")

    conclusion = latest.conclusion or latest.status
    return ToolResult.success(
        {
            "workflow": latest.name,
            "branch": latest.head_branch,
            "status": latest.status,
            "conclusion": conclusion,
            "url": latest.html_url,
        },
        speak=f"the latest run of {latest.name} on {latest.head_branch} is {conclusion}",
    )


@tool(category="dev", timeout=45.0)
async def explain_error_log(log_text: str, context: str = "") -> ToolResult:
    """Explain an error log or stack trace in plain language, and suggest a fix.

    Args:
        log_text: The error output to explain.
        context: What the user was doing when it happened, if known.
    """
    from adrien.core.llm_router import LLMRouter

    if not log_text.strip():
        return ToolResult.failure("there was no log text to look at")

    # A separate LLM call rather than inline reasoning: this keeps the long,
    # noisy log out of the conversation context, where it would crowd out
    # everything else and slow every later turn.
    router = LLMRouter()
    prompt = (
        "Explain this error to someone listening, not reading. Two or three "
        "sentences: what broke, why, and the single most likely fix. No code "
        "blocks, no lists.\n\n"
        + (f"What they were doing: {context}\n\n" if context else "")
        + f"Log:\n{log_text[:6000]}"
    )
    try:
        explanation = await router.complete(prompt, tier="smart", max_tokens=300)
    except Exception as exc:
        return ToolResult.failure(f"could not analyse the log: {exc}")
    return ToolResult.success({"explanation": explanation}, speak=explanation)
