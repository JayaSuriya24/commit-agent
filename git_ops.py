#!/usr/bin/env python3
"""Git index/commit plumbing used to land one chunk at a time."""
import re
import subprocess
from typing import List, Optional

# feature/PROJ-104-login -> PROJ-104 ; bugfix/ab-12/x -> AB-12
TICKET_PATTERN = re.compile(r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9]{1,9})[-_](\d{1,6})(?![0-9])")

_NON_TICKET_PREFIXES = {"feature", "feat", "fix", "bugfix", "hotfix", "chore", "release", "v"}


class GitError(RuntimeError):
    """A git plumbing command failed."""


def _run(args: List[str], stdin: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, input=stdin, capture_output=True, text=True)


def in_repo() -> bool:
    """True when cwd is inside a git work tree."""
    return _run(["git", "rev-parse", "--git-dir"]).returncode == 0


def git_dir() -> str:
    """Absolute path to the .git directory.

    Resolved via git, not hardcoded: a literal ".git/..." only works when the
    tool is invoked from the repo root, and worktrees put it elsewhere entirely.
    """
    result = _run(["git", "rev-parse", "--absolute-git-dir"])
    if result.returncode != 0:
        raise GitError("not inside a git repository")
    return result.stdout.strip()


def get_current_branch() -> Optional[str]:
    result = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    return branch or None


def extract_ticket_id(branch: Optional[str]) -> Optional[str]:
    """Pull a ticket ID out of a branch name (feature/PROJ-104-login -> PROJ-104)."""
    if not branch:
        return None
    for match in TICKET_PATTERN.finditer(branch):
        word = match.group(1)
        if word.lower() in _NON_TICKET_PREFIXES:
            continue
        return f"{word.upper()}-{match.group(2)}"
    return None


def build_commit_body(ticket: Optional[str]) -> Optional[str]:
    return f"Refs: {ticket}" if ticket else None


def get_binary_safe_diff() -> str:
    """Full staged diff including binary payloads — used as a restore point."""
    result = _run(["git", "diff", "--cached", "--binary"])
    if result.returncode != 0:
        raise GitError(result.stderr.strip())
    return result.stdout


def staged_paths() -> List[str]:
    result = _run(["git", "diff", "--cached", "--name-only"])
    return [p for p in result.stdout.splitlines() if p]


def has_staged_changes() -> bool:
    return _run(["git", "diff", "--cached", "--quiet"]).returncode != 0


def unstage_all() -> None:
    """Clear the index back to HEAD, leaving the working tree untouched."""
    result = _run(["git", "reset", "--quiet"])
    if result.returncode != 0:
        raise GitError(f"git reset failed: {result.stderr.strip()}")


def apply_patch_to_index(
    patch: str, check_only: bool = False, three_way: bool = False
) -> None:
    """Stage `patch` via `git apply --cached` (or just validate it).

    `three_way` is for recovery: a backup written just before a crash can name
    changes that already landed in a commit, and a strict apply rejects the
    whole patch over them. A 3-way merge treats those as no-ops and restores
    the rest. Keep it off for normal chunk staging, where a conflict is a bug.
    """
    if not patch.endswith("\n"):
        patch += "\n"
    args = ["git", "apply", "--cached"]
    if three_way:
        args.append("--3way")
    if check_only:
        args.append("--check")
    args.append("-")
    result = _run(args, stdin=patch)
    if result.returncode != 0:
        raise GitError(result.stderr.strip() or "git apply failed")


def commit(message: str) -> None:
    result = _run(["git", "commit", "-m", message])
    if result.returncode != 0:
        raise GitError(result.stderr.strip() or result.stdout.strip())


def head_subject() -> Optional[str]:
    result = _run(["git", "log", "-1", "--pretty=%s"])
    return result.stdout.strip() if result.returncode == 0 else None
