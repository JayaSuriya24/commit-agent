#!/usr/bin/env python3
"""Local Ollama client that returns a validated Conventional Commit message."""
import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

OLLAMA_URL = "http://localhost:11434/api/generate"

VALID_TYPES = ("feat", "fix", "docs", "style", "refactor", "test", "chore")

SUBJECT_MAX = 72

SYSTEM_PROMPT = """You are an expert Git commit assistant.
Given a git diff payload, describe the change as a Conventional Commit.

Respond with JSON only, matching this shape:
{"type": "<type>", "scope": "<scope>", "subject": "<short summary>"}

Rules:
1. "type" MUST be exactly one of: feat, fix, docs, style, refactor, test, chore.
2. "scope" is a single lowercase word naming the touched area (no parentheses).
3. "subject" is imperative mood, lowercase, no trailing period, under 60 chars.
4. Describe what the change does, not which lines moved.
"""


class OllamaError(RuntimeError):
    """Raised when inference fails or returns something uncommittable.

    Callers must abort rather than commit: an error must never be able to
    masquerade as a commit message.
    """


def _extract_json(raw: str) -> Dict[str, Any]:
    """Parse the model's response, tolerating stray prose or markdown fences."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError as e:
            raise OllamaError(f"Model returned invalid JSON: {raw[:200]}") from e
    raise OllamaError(f"Model returned no JSON object: {raw[:200]}")


def build_commit_message(
    parsed: Dict[str, Any], body: Optional[str] = None
) -> str:
    """Assemble and validate the commit string from the model's JSON fields."""
    if not isinstance(parsed, dict):
        raise OllamaError(f"Expected a JSON object, got {type(parsed).__name__}")

    commit_type = str(parsed.get("type", "")).strip().lower()
    if commit_type not in VALID_TYPES:
        raise OllamaError(
            f"Invalid commit type {commit_type!r}; expected one of {', '.join(VALID_TYPES)}"
        )

    subject = " ".join(str(parsed.get("subject", "")).split()).rstrip(".")
    if not subject:
        raise OllamaError("Model returned an empty subject")

    scope = str(parsed.get("scope") or "").strip().strip("()").lower()
    scope = "".join(ch for ch in scope if ch.isalnum() or ch in "-_/.")

    header = f"{commit_type}({scope}): {subject}" if scope else f"{commit_type}: {subject}"
    if len(header) > SUBJECT_MAX:
        keep = SUBJECT_MAX - (len(header) - len(subject))
        header = header.replace(subject, subject[: max(keep, 0)].rstrip())

    return f"{header}\n\n{body.strip()}" if body and body.strip() else header


def generate_commit_message(
    diff_payload: str,
    model: str = "qwen2.5-coder:1.5b",
    scope_hint: Optional[str] = None,
    type_hint: Optional[str] = None,
    body: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """Ask the local Ollama model for a commit message for `diff_payload`.

    Raises OllamaError on any failure so the caller aborts instead of
    committing an error string.
    """
    hints = []
    if scope_hint:
        hints.append(f'Prefer the scope "{scope_hint}" unless the diff clearly contradicts it.')
    if type_hint:
        hints.append(f'These files usually warrant type "{type_hint}".')
    hint_block = ("\nHints:\n" + "\n".join(hints) + "\n") if hints else ""

    prompt = f"{SYSTEM_PROMPT}{hint_block}\nDiff Payload:\n{diff_payload}"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Constrain decoding to valid JSON instead of regex-scrubbing prose.
        "format": "json",
        "options": {"temperature": 0.2},
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise OllamaError(
            f"Could not reach Ollama at {OLLAMA_URL}. "
            f"Ensure Ollama is installed and running. Details: {e}"
        ) from e
    except json.JSONDecodeError as e:
        raise OllamaError(f"Ollama returned a malformed response: {e}") from e
    except OSError as e:  # timeouts, connection resets
        raise OllamaError(f"Ollama request failed: {e}") from e

    if "error" in result:
        raise OllamaError(f"Ollama error: {result['error']}")

    return build_commit_message(_extract_json(result.get("response", "")), body=body)
