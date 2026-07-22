#!/usr/bin/env python3
import json
import re
import subprocess
import sys
from typing import Any, Dict, List


def get_raw_git_diff(staged: bool = True) -> str:
    """Executes git diff and retrieves raw output."""
    cmd = ["git", "diff", "--cached"] if staged else ["git", "diff"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff command: {e}", file=sys.stderr)
        sys.exit(1)


def parse_git_diff(diff_text: str) -> List[Dict[str, Any]]:
    """Parses raw git diff string into structured file blocks and hunks."""
    files = []
    file_diffs = re.split(r"^diff --git ", diff_text, flags=re.MULTILINE)

    for file_diff in file_diffs:
        if not file_diff.strip():
            continue

        lines = file_diff.splitlines()
        header_info = lines[0] if lines else ""

        path_match = re.search(r"a/(.+?)\s+b/(.+)", header_info)
        old_path = path_match.group(1) if path_match else None
        new_path = path_match.group(2) if path_match else None

        current_file = {
            "old_path": old_path,
            "new_path": new_path,
            "hunks": [],
        }

        hunk_blocks = re.split(
            r"^(@@\s+-[0-9,]+\s+\+[0-9,]+\s+@@.*)$",
            file_diff,
            flags=re.MULTILINE,
        )

        for i in range(1, len(hunk_blocks), 2):
            hunk_header = hunk_blocks[i]
            hunk_content = (
                hunk_blocks[i + 1] if i + 1 < len(hunk_blocks) else ""
            )

            added_lines = []
            deleted_lines = []
            raw_lines = []

            for line in hunk_content.splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    added_lines.append(line[1:])
                    raw_lines.append(line)
                elif line.startswith("-") and not line.startswith("---"):
                    deleted_lines.append(line[1:])
                    raw_lines.append(line)
                elif line.startswith(" "):
                    raw_lines.append(line)

            current_file["hunks"].append(
                {
                    "header": hunk_header,
                    "added_lines": added_lines,
                    "deleted_lines": deleted_lines,
                    "raw_patch": "\n".join(raw_lines).strip(),
                }
            )

        if current_file["hunks"]:
            files.append(current_file)

    return files


def format_for_llm_prompt(parsed_diffs: List[Dict[str, Any]]) -> str:
    """Formats the structured diff into a token-efficient prompt string."""
    prompt_blocks = []

    for file in parsed_diffs:
        file_path = file["new_path"] or file["old_path"]
        prompt_blocks.append(f"### File: {file_path}")

        for idx, hunk in enumerate(file["hunks"], 1):
            prompt_blocks.append(f"#### Hunk {idx}: {hunk['header']}")
            prompt_blocks.append("```diff")
            prompt_blocks.append(hunk["raw_patch"])
            prompt_blocks.append("```\n")

    return "\n".join(prompt_blocks)