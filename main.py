#!/usr/bin/env python3
import subprocess
import sys

from git_diff_parser import (
    format_for_llm_prompt,
    get_raw_git_diff,
    parse_git_diff,
)
from llm_client import generate_commit_message


def main():
    # 1. Fetch staged git changes
    raw_diff = get_raw_git_diff(staged=True)
    if not raw_diff.strip():
        print("❌ No staged changes found. Please run 'git add <files>' first.")
        sys.exit(0)

    # 2. Parse and format diff
    structured_diff = parse_git_diff(raw_diff)
    prompt_payload = format_for_llm_prompt(structured_diff)

    print("⚡ Generating conventional commit message from local AI...")

    # 3. Call local LLM
    commit_msg = generate_commit_message(prompt_payload)

    print("\n----------------------------------------")
    print(f"Suggested Commit Message:\n\033[1;32m{commit_msg}\033[0m")
    print("----------------------------------------\n")

    # 4. Interactive user choice
    choice = (
        input(
            "Do you want to commit with this message? (y/e/n) [yes / edit / no]: "
        )
        .lower()
        .strip()
    )

    if choice == "y":
        subprocess.run(["git", "commit", "-m", commit_msg])
        print("✅ Committed successfully!")
    elif choice == "e":
        custom_msg = input("Type your custom commit message: ")
        if custom_msg.strip():
            subprocess.run(["git", "commit", "-m", custom_msg.strip()])
            print("✅ Committed with custom message!")
    else:
        print("❌ Commit cancelled.")


if __name__ == "__main__":
    main()