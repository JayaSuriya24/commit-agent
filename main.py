#!/usr/bin/env python3
"""Agentic commit tool: split staged changes into atomic, per-domain commits."""
import sys
from typing import Any, Dict, List, Optional

import git_ops
from domain_chunker import chunk_by_domain
from git_diff_parser import format_for_llm_prompt, get_raw_git_diff, parse_git_diff
from llm_client import OllamaError, generate_commit_message

GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
DIM = "\033[2m"
RESET = "\033[0m"


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


def _resolve_message(chunk: Dict[str, Any], body: Optional[str]) -> Optional[str]:
    """Generate a message for one chunk, or None if the user skips it."""
    payload = format_for_llm_prompt(chunk["files"])
    print(f"⚡ Generating message for {chunk['domain']}...")

    try:
        message = generate_commit_message(
            payload,
            scope_hint=chunk["scope_hint"],
            type_hint=chunk["type_hint"],
            body=body,
        )
    except OllamaError as e:
        # Bug #1: never let an error string become a commit message.
        print(f"❌ Inference failed: {e}", file=sys.stderr)
        choice = input("Write this chunk's message by hand? (y/N): ").lower().strip()
        if choice != "y":
            return None
        message = ""

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


def _commit_chunks(chunks: List[Dict[str, Any]], body: Optional[str]) -> None:
    """Stage and commit one chunk at a time, restoring anything left over."""
    backup = git_ops.get_binary_safe_diff()
    committed, leftover = 0, []

    git_ops.unstage_all()
    try:
        for i, chunk in enumerate(chunks, 1):
            _print_chunk(i, len(chunks), chunk)
            try:
                message = _resolve_message(chunk, body)
            except KeyboardInterrupt:
                print("\n⏹  Stopping early.")
                leftover.extend(chunks[i - 1 :])
                break

            if message is None:
                print("↩︎  Skipped.")
                leftover.append(chunk)
                continue

            git_ops.apply_patch_to_index(chunk["patch"])
            git_ops.commit(message)
            committed += 1
            print(f"✅ Committed: {message.splitlines()[0]}")
    except Exception as e:
        print(f"\n❌ {e}", file=sys.stderr)
        print("Restoring the original staged state...", file=sys.stderr)
        git_ops.unstage_all()
        git_ops.apply_patch_to_index(backup)
        print("Index restored. No further commits were made.", file=sys.stderr)
        sys.exit(1)

    # Anything skipped goes back on the index so it is never silently dropped.
    for chunk in leftover:
        try:
            git_ops.apply_patch_to_index(chunk["patch"])
        except git_ops.GitError as e:
            print(f"⚠️  Could not re-stage {', '.join(chunk['paths'])}: {e}", file=sys.stderr)

    print(f"\n🏁 {committed} commit(s) created, {len(leftover)} chunk(s) left staged.")


def main() -> None:
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
    if ticket:
        print(f"🔖 Ticket {ticket} detected on branch '{branch}'.")

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
        message = _resolve_message(chunks[0], body)
        if message is None:
            print("❌ Commit cancelled.")
            sys.exit(0)
        git_ops.commit(message)
        print("✅ Committed successfully!")
        return

    chunks = chunk_by_domain(parsed)
    print(f"🧩 {len(parsed)} file(s) grouped into {len(chunks)} atomic commit(s).")

    if len(chunks) == 1:
        # Nothing to split: commit the index exactly as the user staged it.
        _print_chunk(1, 1, chunks[0])
        message = _resolve_message(chunks[0], body)
        if message is None:
            print("❌ Commit cancelled.")
            sys.exit(0)
        git_ops.commit(message)
        print("✅ Committed successfully!")
        return

    _commit_chunks(chunks, body)


if __name__ == "__main__":
    main()
