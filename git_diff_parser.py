#!/usr/bin/env python3
"""Parse a unified git diff into structured, re-applyable file/hunk records."""
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

DIFF_HEADER = "diff --git "


def get_raw_git_diff(staged: bool = True) -> str:
    """Executes git diff and retrieves raw output."""
    cmd = ["git", "diff", "--cached"] if staged else ["git", "diff"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff command: {e}", file=sys.stderr)
        sys.exit(1)


def _unquote_path(raw: str) -> str:
    """Undo git's C-style quoting (used when a path has odd bytes)."""
    inner = raw[1:-1]
    out = bytearray()
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch != "\\":
            out.extend(ch.encode("utf-8"))
            i += 1
            continue
        nxt = inner[i + 1] if i + 1 < len(inner) else ""
        simple = {"n": 10, "t": 9, "r": 13, "\\": 92, '"': 34, "a": 7, "b": 8, "f": 12, "v": 11}
        if nxt in simple:
            out.append(simple[nxt])
            i += 2
        elif nxt.isdigit():
            octal = inner[i + 1 : i + 4]
            out.append(int(octal, 8))
            i += 1 + len(octal)
        else:
            out.extend(nxt.encode("utf-8"))
            i += 2
    return out.decode("utf-8", errors="replace")


def _clean_path(raw: str) -> Optional[str]:
    """Turn an `a/some path.py` token from a ---/+++ line into a bare path.

    Handles spaces (the old `a/(.+?)\\s+b/(.+)` regex did not), git's C-style
    quoting, and the /dev/null sentinel used for adds and deletes.
    """
    raw = raw.rstrip("\r\n")
    # git appends a tab (plus optional timestamp) when the path needs one
    if "\t" in raw:
        raw = raw.split("\t", 1)[0]
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        raw = _unquote_path(raw)
    if raw == "/dev/null":
        return None
    if len(raw) > 2 and raw[1] == "/" and raw[0] in ("a", "b"):
        return raw[2:]
    return raw or None


def _paths_from_diff_line(line: str) -> Tuple[Optional[str], Optional[str]]:
    """Fallback path extraction from `diff --git a/X b/Y` (no ---/+++ present).

    Prefers the split where both halves are identical, so paths containing
    " b/" or spaces still resolve correctly for non-renames.
    """
    rest = line[len(DIFF_HEADER) :].rstrip("\r\n")
    if rest.startswith('"'):
        # quoted form: "a/x" "b/y"
        parts = re.findall(r'"(?:[^"\\]|\\.)*"|\S+', rest)
        if len(parts) == 2:
            return _clean_path(parts[0]), _clean_path(parts[1])
    candidates = [m.start() for m in re.finditer(r" b/", rest)]
    for idx in candidates:
        left, right = rest[:idx], rest[idx + 1 :]
        lp, rp = _clean_path(left), _clean_path(right)
        if lp is not None and lp == rp:
            return lp, rp
    if candidates:
        idx = candidates[0]
        return _clean_path(rest[:idx]), _clean_path(rest[idx + 1 :])
    return None, None


def _build_hunk(header: str, body: List[str]) -> Dict[str, Any]:
    """Build a hunk record; `full_patch` stays byte-faithful to the input."""
    added_lines: List[str] = []
    deleted_lines: List[str] = []
    prompt_lines: List[str] = []

    for line in body:
        if line.startswith("+"):
            added_lines.append(line[1:])
            prompt_lines.append(line)
        elif line.startswith("-"):
            deleted_lines.append(line[1:])
            prompt_lines.append(line)
        elif line.startswith(" ") or line == "":
            prompt_lines.append(line)
        # "\ No newline at end of file" is kept in full_patch only

    return {
        "header": header,
        "added_lines": added_lines,
        "deleted_lines": deleted_lines,
        # Condensed view for the LLM prompt. Only newlines are trimmed so the
        # +/-/space marker column survives.
        "raw_patch": "\n".join(prompt_lines).strip("\n"),
        # Untouched hunk, @@ header included: safe to feed to `git apply`.
        "full_patch": "\n".join([header] + body),
    }


def parse_git_diff(diff_text: str) -> List[Dict[str, Any]]:
    """Parses a raw git diff into structured file blocks and hunks.

    Each file carries `header_lines` (the `diff --git`/`index`/`---`/`+++`
    preamble) and a `full_patch` that can be handed straight to `git apply`.
    """
    files: List[Dict[str, Any]] = []
    lines = diff_text.splitlines()
    i, n = 0, len(lines)

    while i < n:
        if not lines[i].startswith(DIFF_HEADER):
            i += 1
            continue

        diff_line = lines[i]
        header_lines = [diff_line]
        i += 1

        # Preamble: index / mode / similarity / --- / +++ / binary notices.
        while i < n and not lines[i].startswith("@@") and not lines[i].startswith(DIFF_HEADER):
            header_lines.append(lines[i])
            i += 1

        old_path = new_path = None
        for line in header_lines:
            if line.startswith("--- ") and old_path is None:
                old_path = _clean_path(line[4:])
            elif line.startswith("+++ ") and new_path is None:
                new_path = _clean_path(line[4:])
        if old_path is None and new_path is None:
            old_path, new_path = _paths_from_diff_line(diff_line)

        hunks = []
        while i < n and lines[i].startswith("@@"):
            hunk_header = lines[i]
            i += 1
            body: List[str] = []
            while i < n and not lines[i].startswith("@@") and not lines[i].startswith(DIFF_HEADER):
                body.append(lines[i])
                i += 1
            hunks.append(_build_hunk(hunk_header, body))

        is_binary = any(
            line.startswith("Binary files ") or line.startswith("GIT binary patch")
            for line in header_lines
        )

        files.append(
            {
                "old_path": old_path,
                "new_path": new_path,
                "header_lines": header_lines,
                "is_binary": is_binary,
                "hunks": hunks,
                "full_patch": build_file_patch_from_parts(header_lines, hunks),
            }
        )

    return files


def build_file_patch_from_parts(header_lines: List[str], hunks: List[Dict[str, Any]]) -> str:
    """Reassemble one file's diff from its preamble and hunks."""
    parts = list(header_lines)
    for hunk in hunks:
        parts.extend(hunk["full_patch"].split("\n"))
    return "\n".join(parts) + "\n"


def build_combined_patch(files: List[Dict[str, Any]]) -> str:
    """Concatenate several parsed files into one applyable patch."""
    return "".join(f["full_patch"] for f in files)


def file_path_of(file: Dict[str, Any]) -> str:
    """Best display/routing path for a parsed file (new path wins)."""
    return file.get("new_path") or file.get("old_path") or "<unknown>"


def format_for_llm_prompt(parsed_diffs: List[Dict[str, Any]]) -> str:
    """Formats the structured diff into a token-efficient prompt string."""
    prompt_blocks = []

    for file in parsed_diffs:
        prompt_blocks.append(f"### File: {file_path_of(file)}")

        if file.get("is_binary"):
            prompt_blocks.append("(binary file)\n")
            continue

        for idx, hunk in enumerate(file["hunks"], 1):
            prompt_blocks.append(f"#### Hunk {idx}: {hunk['header']}")
            prompt_blocks.append("```diff")
            prompt_blocks.append(hunk["raw_patch"])
            prompt_blocks.append("```\n")

    return "\n".join(prompt_blocks)
