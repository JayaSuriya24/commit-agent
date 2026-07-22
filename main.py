#!/usr/bin/python3
"""Agentic commit tool: split staged changes into atomic, per-domain commits.

Shebang is pinned to the system interpreter on purpose. `env python3` would
pick up whatever virtualenv happens to be active in the caller's shell, and
this runs from any repo on the machine. The tool is stdlib-only, so the system
python is sufficient and no venv path is baked into a committed file.
"""
import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import git_ops
from domain_chunker import chunk_by_domain
from git_diff_parser import build_combined_patch, format_for_llm_prompt, get_raw_git_diff, parse_git_diff
from llm_client import DEFAULT_MODEL, OllamaError, generate_commit_message

GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
DIM = "\033[2m"
RESET = "\033[0m"

BACKUP_PATCH_NAME = "ai-commit-backup.patch"


def backup_patch_path() -> Optional[str]:
    """Where the emergency backup lives, or None outside a git repo."""
    try:
        return os.path.join(git_ops.git_dir(), BACKUP_PATCH_NAME)
    except git_ops.GitError:
        return None


def create_emergency_backup(patch: Optional[str] = None) -> None:
    """Persist staged changes to disk before we reset the index.

    Uses the --binary diff so binary blobs survive the round trip; a plain
    `git diff --cached` would restore a corrupted index.
    """
    path = backup_patch_path()
    if path is None:
        return
    if patch is None:
        patch = git_ops.get_binary_safe_diff()
    if not patch.strip():
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(patch)


def refresh_backup(remaining: List[Dict[str, Any]]) -> None:
    """Shrink the backup to the chunks that are still uncommitted.

    After a chunk lands, HEAD has moved, so the original full-diff backup no
    longer applies. Only what is left is still restorable.
    """
    if not remaining:
        cleanup_backup()
        return
    create_emergency_backup(build_combined_patch([f for c in remaining for f in c["files"]]))


def check_for_recovery() -> None:
    """Offer to restore staging left behind by an interrupted run."""
    path = backup_patch_path()
    if path is None or not os.path.exists(path):
        return

    print(f"⚠️  Found a backup patch from an interrupted run:\n   {path}")
    choice = (
        input("Restore staged changes? (y = restore / d = discard / anything else = decide later): ")
        .lower()
        .strip()
    )

    if choice == "y":
        with open(path, encoding="utf-8", newline="") as f:
            patch = f.read()
        try:
            git_ops.apply_patch_to_index(patch, three_way=True)
        except git_ops.GitError as e:
            # Do NOT delete the backup here: it is the only copy.
            print(f"❌ Could not restore: {e}", file=sys.stderr)
            print(f"   Backup kept at {path} — apply it by hand once resolved.", file=sys.stderr)
            print("   Your file contents are safe; only staging was affected.", file=sys.stderr)
            return
        os.remove(path)
        restored = git_ops.staged_paths()
        print(f"✅ Staging restored ({len(restored)} file(s)): {', '.join(restored)}")
    elif choice == "d":
        os.remove(path)
        print("🗑️  Backup discarded.")
    else:
        print(f"↩︎  Left in place at {path}")


def cleanup_backup() -> None:
    """Remove the emergency backup after clean completion."""
    path = backup_patch_path()
    if path and os.path.exists(path):
        os.remove(path)


def _print_chunk(index: int, total: int, chunk: Dict[str, Any]) -> None:
    print(f"\n{YELLOW}[{index}/{total}] {chunk['label']}{RESET}  ({chunk['domain']})")
    for path in chunk["paths"]:
        print(f"  {DIM}•{RESET} {path}")


def _prompt_choice(message: str) -> str:
    print("\n----------------------------------------")
    print(f"Suggested Commit Message:\n{GREEN}{message}{RESET}")
    print("----------------------------------------")
    return (
        input("Commit this chunk? (y/e/n/q) [yes / edit / skip / quit]: ").lower().strip()
    )


def _resolve_message(
    chunk: Dict[str, Any], body: Optional[str], opts: Any
) -> Optional[str]:
    """Generate a message for one chunk, or None if it should be skipped."""
    payload = format_for_llm_prompt(chunk["files"])
    print(f"⚡ Generating message for {chunk['domain']}...")

    try:
        message = generate_commit_message(
            payload,
            model=opts.model,
            scope_hint=chunk["scope_hint"],
            type_hint=chunk["type_hint"],
            body=body,
        )
    except OllamaError as e:
        # Bug #1: never let an error string become a commit message.
        print(f"❌ Inference failed: {e}", file=sys.stderr)
        if opts.yes or opts.dry_run:
            return None  # non-interactive: skip rather than hang on input()
        choice = input("Write this chunk's message by hand? (y/N): ").lower().strip()
        if choice != "y":
            return None
        message = ""

    # Both modes are non-interactive; prompting here would hang with no tty.
    if opts.yes or opts.dry_run:
        return message or None

    while True:
        choice = _prompt_choice(message) if message else "e"
        if choice in ("y", ""):
            return message
        if choice == "e":
            custom = input("Type your custom commit message: ").strip()
            if custom:
                message = f"{custom}\n\n{body}" if body else custom
            elif not message:
                return None  # nothing generated and nothing typed
            continue
        if choice == "q":
            raise KeyboardInterrupt
        return None


def _restage(chunks: List[Dict[str, Any]]) -> None:
    for chunk in chunks:
        try:
            git_ops.apply_patch_to_index(chunk["patch"])
        except git_ops.GitError as e:
            print(f"⚠️  Could not re-stage {', '.join(chunk['paths'])}: {e}", file=sys.stderr)


def _commit_single(chunk: Dict[str, Any], body: Optional[str], opts) -> None:
    """Commit the index exactly as staged — no reset, no patch surgery."""
    message = _resolve_message(chunk, body, opts)
    if message is None:
        print("❌ Commit cancelled.")
        sys.exit(0)
    if opts.dry_run:
        print("🔍 Dry run — nothing committed.")
        return
    git_ops.commit(message)
    print("✅ Committed successfully!")


def _dry_run(chunks: List[Dict[str, Any]], body: Optional[str], opts) -> None:
    """Show the split and the messages without touching the index."""
    for i, chunk in enumerate(chunks, 1):
        _print_chunk(i, len(chunks), chunk)
        message = _resolve_message(chunk, body, opts)
        if message:
            print("\n" + "\n".join(f"    {line}" for line in message.splitlines()))
    print(f"\n🔍 Dry run — {len(chunks)} commit(s) planned, nothing committed.")


def _commit_chunks(chunks: List[Dict[str, Any]], body: Optional[str], opts) -> None:
    """Stage and commit one chunk at a time, restoring anything left over."""
    # On disk, not just in memory: a SIGKILL between the reset below and the
    # first successful apply would otherwise lose the user's staging for good.
    create_emergency_backup()

    # Chunks whose changes are not yet in any commit. Committed chunks drop out
    # because they are safe in history and no longer belong in the backup.
    pending = list(chunks)
    committed = 0

    git_ops.unstage_all()
    try:
        for i, chunk in enumerate(chunks, 1):
            _print_chunk(i, len(chunks), chunk)
            try:
                message = _resolve_message(chunk, body, opts)
            except KeyboardInterrupt:
                print("\n⏹  Stopping early.")
                break

            if message is None:
                print("↩︎  Skipped.")
                continue

            git_ops.apply_patch_to_index(chunk["patch"])
            git_ops.commit(message)
            pending = [c for c in pending if c is not chunk]
            refresh_backup(pending)
            committed += 1
            print(f"✅ Committed: {message.splitlines()[0]}")
    except Exception as e:
        print(f"\n❌ {e}", file=sys.stderr)
        print("Restoring uncommitted changes to the index...", file=sys.stderr)
        git_ops.unstage_all()
        _restage(pending)
        print(f"Index restored. {committed} commit(s) were made.", file=sys.stderr)
        sys.exit(1)

    # Skipped and unreached chunks go back on the index, never silently dropped.
    _restage(pending)
    cleanup_backup()
    print(f"\n🏁 {committed} commit(s) created, {len(pending)} chunk(s) left staged.")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai-commit",
        description="Split staged changes into atomic, per-domain commits using a local LLM.",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="approve every chunk without prompting"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the plan and generated messages; commit nothing",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--suggest",
        metavar="FILE",
        help=(
            "write ONE message for all staged changes to FILE and exit; "
            "makes no commits. For prepare-commit-msg hooks."
        ),
    )
    return parser.parse_args(argv)


def _suggest_only(parsed: List[Dict[str, Any]], body: Optional[str], opts) -> int:
    """Hook mode: write a message to a file. Never commits, never resets."""
    payload = format_for_llm_prompt(parsed)
    try:
        message = generate_commit_message(
            payload, model=opts.model, body=body
        )
    except OllamaError as e:
        # A hook must not block the commit: leave the file untouched so git
        # falls back to the editor with its default template.
        print(f"⚠️  ai-commit: {e}", file=sys.stderr)
        return 0

    try:
        with open(opts.suggest, "r", encoding="utf-8") as f:
            existing = f.read()
    except OSError:
        existing = ""

    # Only fill in an empty/template message. If git already put real content
    # there (merge, squash, -m, an amend), leave the user's text alone.
    if any(line.strip() and not line.lstrip().startswith("#") for line in existing.splitlines()):
        print("ℹ️  ai-commit: message already present, leaving it untouched.", file=sys.stderr)
        return 0

    with open(opts.suggest, "w", encoding="utf-8") as f:
        f.write(f"{message}\n{existing}")  # keep git's comment block below
    return 0


def main(argv: Optional[List[str]] = None) -> None:
    opts = _parse_args(argv)

    if not git_ops.in_repo():
        print("❌ Not a git repository.", file=sys.stderr)
        sys.exit(1)

    # Before anything else: a previous run may have died mid-split.
    if not opts.suggest:
        check_for_recovery()

    raw_diff = get_raw_git_diff(staged=True)
    if not raw_diff.strip():
        print("❌ No staged changes found. Please run 'git add <files>' first.")
        sys.exit(0)

    parsed = parse_git_diff(raw_diff)
    if not parsed:
        print("❌ Could not parse the staged diff.", file=sys.stderr)
        sys.exit(1)

    branch = git_ops.get_current_branch()
    ticket = git_ops.extract_ticket_id(branch)
    body = git_ops.build_commit_body(ticket)
    if ticket and not opts.suggest:
        print(f"🔖 Ticket {ticket} detected on branch '{branch}'.")

    # Hook mode: one message, written to a file, no commits and no index reset.
    if opts.suggest:
        sys.exit(_suggest_only(parsed, body, opts))

    # Binary files have no re-applyable text hunks, so the split-and-restage
    # dance would lose them. Fall back to a single commit.
    if any(f.get("is_binary") for f in parsed):
        print("ℹ️  Binary changes staged — committing everything as one chunk.")
        chunks = [
            {
                "domain": "all",
                "label": "all staged changes",
                "scope_hint": None,
                "type_hint": None,
                "order": 0,
                "files": parsed,
                "paths": [f.get("new_path") or f.get("old_path") for f in parsed],
                "patch": "",
            }
        ]
        _commit_single(chunks[0], body, opts)
        return

    chunks = chunk_by_domain(parsed)
    print(f"🧩 {len(parsed)} file(s) grouped into {len(chunks)} atomic commit(s).")

    if len(chunks) == 1:
        # Nothing to split: commit the index exactly as the user staged it.
        _print_chunk(1, 1, chunks[0])
        _commit_single(chunks[0], body, opts)
        return

    if opts.dry_run:
        _dry_run(chunks, body, opts)
        return

    _commit_chunks(chunks, body, opts)


if __name__ == "__main__":
    main()
