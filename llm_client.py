#!/usr/bin/env python3
import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """You are an expert Git commit assistant.
Given a git diff payload, generate a concise Conventional Commit message.

Rules:
1. Follow the format: <type>(<scope>): <short summary>
2. Types must be one of: feat, fix, docs, style, refactor, test, chore.
3. Keep the entire summary on a single line under 72 characters.
4. Output ONLY the raw commit text. Do NOT wrap in markdown fences, backticks, or quotes.
"""


def generate_commit_message(
    diff_payload: str, model: str = "qwen2.5-coder:1.5b"
) -> str:
    """Sends the diff payload to local Ollama API and cleans the returned response."""
    prompt = f"{SYSTEM_PROMPT}\n\nDiff Payload:\n{diff_payload}"

    payload = {"model": model, "prompt": prompt, "stream": False}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            raw_msg = result.get("response", "").strip()

            # Clean markdown code fences (e.g. ```text ... ```)
            if raw_msg.startswith("```"):
                lines = raw_msg.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_msg = "\n".join(lines).strip()

            # Strip leading/trailing backticks or quotation marks
            clean_msg = raw_msg.strip("`'\"")

            return clean_msg

    except urllib.error.URLError as e:
        return (
            f"Error connecting to Ollama at {OLLAMA_URL}.\n"
            f"Please ensure Ollama is installed and running.\nDetails: {e}"
        )