import json
import subprocess

import pytest

import git_ops
from domain_chunker import chunk_by_domain, classify_path
from git_diff_parser import (
    build_combined_patch,
    format_for_llm_prompt,
    parse_git_diff,
)
from llm_client import OllamaError, build_commit_message, generate_commit_message

# Sample raw diff payload for mocking
SAMPLE_RAW_DIFF = """diff --git a/app.py b/app.py
index e69de29..b835010 100644
--- a/app.py
+++ b/app.py
@@ -0,0 +1 @@
+print('Hello world from AI CLI')
"""

MULTI_DOMAIN_DIFF = """diff --git a/prisma/schema.prisma b/prisma/schema.prisma
index 1111111..2222222 100644
--- a/prisma/schema.prisma
+++ b/prisma/schema.prisma
@@ -1,3 +1,4 @@
 model User {
   id Int @id
+  avatarUrl String?
 }
diff --git a/src/components/UserCard.tsx b/src/components/UserCard.tsx
index 3333333..4444444 100644
--- a/src/components/UserCard.tsx
+++ b/src/components/UserCard.tsx
@@ -1,2 +1,3 @@
 export const UserCard = () => {
+  return <img src={user.avatarUrl} />
 }
"""


def _mock_ollama(monkeypatch, response_text):
    """Patch urlopen so no real network request is made."""

    class MockResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"response": response_text}).encode("utf-8")

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout=60: MockResponse()
    )


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def test_parse_git_diff():
    """Verify raw diff string parses correctly into file objects and hunks."""
    parsed = parse_git_diff(SAMPLE_RAW_DIFF)

    assert len(parsed) == 1
    assert parsed[0]["new_path"] == "app.py"
    assert len(parsed[0]["hunks"]) == 1
    assert "print('Hello world from AI CLI')" in parsed[0]["hunks"][0]["added_lines"]


def test_format_for_llm_prompt():
    """Verify parsed diff formats cleanly into a markdown prompt block."""
    parsed = parse_git_diff(SAMPLE_RAW_DIFF)
    formatted_prompt = format_for_llm_prompt(parsed)

    assert "### File: app.py" in formatted_prompt
    assert "```diff" in formatted_prompt


def test_full_patch_keeps_headers_and_marker_column():
    """Bug #2: full_patch must stay a valid unified diff, unlike raw_patch."""
    diff = (
        "diff --git a/app.py b/app.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,2 +1,3 @@\n"
        " import os\n"
        "+import sys\n"
        " print(os.getcwd())\n"
    )
    parsed = parse_git_diff(diff)
    full = parsed[0]["full_patch"]

    assert full.startswith("diff --git a/app.py b/app.py\n")
    assert "--- a/app.py\n" in full
    assert "+++ b/app.py\n" in full
    assert "@@ -1,2 +1,3 @@\n" in full
    # the leading space marker of the first context line survives
    assert "\n import os\n" in full
    assert full.endswith("\n")
    # the hunk itself is also independently applyable
    assert parsed[0]["hunks"][0]["full_patch"].startswith("@@ -1,2 +1,3 @@\n import os")


def test_parses_paths_containing_spaces():
    """Bug #5: the old a/(.+?)\\s+b/(.+) regex truncated paths with spaces."""
    diff = (
        "diff --git a/my docs/read me.md b/my docs/read me.md\n"
        "index 1111111..2222222 100644\n"
        "--- a/my docs/read me.md\n"
        "+++ b/my docs/read me.md\n"
        "@@ -0,0 +1 @@\n"
        "+hello\n"
    )
    parsed = parse_git_diff(diff)

    assert parsed[0]["old_path"] == "my docs/read me.md"
    assert parsed[0]["new_path"] == "my docs/read me.md"


def test_parses_new_and_deleted_files():
    diff = (
        "diff --git a/gone.py b/gone.py\n"
        "deleted file mode 100644\n"
        "index 1111111..0000000\n"
        "--- a/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-print('bye')\n"
    )
    parsed = parse_git_diff(diff)

    assert parsed[0]["old_path"] == "gone.py"
    assert parsed[0]["new_path"] is None
    assert parsed[0]["hunks"][0]["deleted_lines"] == ["print('bye')"]


def test_parses_multiple_files():
    parsed = parse_git_diff(MULTI_DOMAIN_DIFF)

    assert [f["new_path"] for f in parsed] == [
        "prisma/schema.prisma",
        "src/components/UserCard.tsx",
    ]


# --------------------------------------------------------------------------
# Round-trip: stored patches must survive `git apply --cached --check`
# --------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path):
    """A throwaway repo with one committed file."""

    def git(*args, **kwargs):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, **kwargs
        )

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("config", "commit.gpgsign", "false")
    (tmp_path / "app.py").write_text("import os\nprint(os.getcwd())\n")
    (tmp_path / "notes.md").write_text("# Notes\n")
    git("add", "-A")
    git("commit", "-q", "-m", "initial")
    return tmp_path, git


def test_stored_patch_round_trips_through_git_apply(git_repo):
    """Bug #2 regression: a stored full_patch re-applies cleanly to the index."""
    repo, git = git_repo
    (repo / "app.py").write_text("import os\nimport sys\nprint(os.getcwd())\n")
    (repo / "notes.md").write_text("# Notes\n\nMore notes.\n")
    git("add", "-A")

    raw_diff = git("diff", "--cached").stdout
    parsed = parse_git_diff(raw_diff)
    assert len(parsed) == 2

    # Reset the index; the stored patches must be able to rebuild it.
    git("reset", "-q")
    assert git("diff", "--cached", "--quiet").returncode == 0

    for file in parsed:
        check = git("apply", "--cached", "--check", "-", input=file["full_patch"])
        assert check.returncode == 0, f"--check failed: {check.stderr}"

    combined = build_combined_patch(parsed)
    applied = git("apply", "--cached", "-", input=combined)
    assert applied.returncode == 0, applied.stderr
    assert git("diff", "--cached").stdout == raw_diff


def test_chunk_patches_commit_independently(git_repo):
    """Each chunk's patch stages only its own files."""
    repo, git = git_repo
    (repo / "app.py").write_text("import os\nimport sys\nprint(os.getcwd())\n")
    (repo / "notes.md").write_text("# Notes\n\nMore notes.\n")
    git("add", "-A")

    parsed = parse_git_diff(git("diff", "--cached").stdout)
    chunks = chunk_by_domain(parsed)
    git("reset", "-q")

    assert len(chunks) == 2
    for chunk in chunks:
        assert git("apply", "--cached", "-", input=chunk["patch"]).returncode == 0
        staged = git("diff", "--cached", "--name-only").stdout.split()
        assert staged == chunk["paths"]
        git("commit", "-q", "-m", f"chunk: {chunk['domain']}")

    assert git("diff", "--cached", "--quiet").returncode == 0


def test_patch_with_spaces_in_path_applies(git_repo):
    """Bug #5 regression: path handling holds up against real git apply."""
    repo, git = git_repo
    (repo / "my docs").mkdir()
    (repo / "my docs" / "read me.md").write_text("hello\n")
    git("add", "-A")

    parsed = parse_git_diff(git("diff", "--cached").stdout)
    assert parsed[0]["new_path"] == "my docs/read me.md"

    git("reset", "-q")
    applied = git("apply", "--cached", "-", input=parsed[0]["full_patch"])
    assert applied.returncode == 0, applied.stderr
    assert git("diff", "--cached", "--name-only").stdout.strip() == "my docs/read me.md"


# --------------------------------------------------------------------------
# Domain chunker
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        ("prisma/schema.prisma", "db"),
        ("db/migrations/0001_init.sql", "db"),
        ("src/components/UserCard.tsx", "ui"),
        ("src/styles/main.scss", "ui"),
        ("test_git_ai.py", "test"),
        ("tests/test_thing.py", "test"),
        ("src/components/Button.test.tsx", "test"),
        ("README.md", "docs"),
        ("docs/guide.rst", "docs"),
        (".github/workflows/ci.yml", "ci"),
        ("package-lock.json", "deps"),
        ("pyproject.toml", "config"),
        ("src/api/users.py", "api"),
        ("llm_client.py", "core"),
    ],
)
def test_classify_path(path, expected):
    assert classify_path(path) == expected


def test_test_files_beat_ui_and_config_rules():
    """Ordering matters: a .test.tsx is a test, not UI."""
    assert classify_path("src/components/Modal.test.tsx") == "test"
    assert classify_path("tests/fixtures/config.yaml") == "test"


def test_chunker_splits_schema_from_ui():
    """The headline case: schema.prisma + UserCard.tsx become two commits."""
    chunks = chunk_by_domain(parse_git_diff(MULTI_DOMAIN_DIFF))

    assert [c["domain"] for c in chunks] == ["db", "ui"]  # db ordered first
    assert chunks[0]["paths"] == ["prisma/schema.prisma"]
    assert chunks[0]["scope_hint"] == "schema"
    assert chunks[1]["paths"] == ["src/components/UserCard.tsx"]
    assert chunks[1]["scope_hint"] == "ui"
    assert "schema.prisma" in chunks[0]["patch"]
    assert "UserCard.tsx" not in chunks[0]["patch"]


def test_chunker_keeps_single_domain_together():
    chunks = chunk_by_domain(parse_git_diff(SAMPLE_RAW_DIFF))

    assert len(chunks) == 1
    assert chunks[0]["domain"] == "core"
    assert chunks[0]["scope_hint"] == "app"  # inferred from the filename stem


# --------------------------------------------------------------------------
# LLM client
# --------------------------------------------------------------------------


def test_generate_commit_message_from_json(monkeypatch):
    """JSON-constrained output is assembled into a Conventional Commit."""
    _mock_ollama(
        monkeypatch,
        '{"type": "feat", "scope": "app", "subject": "add print statement"}',
    )

    assert generate_commit_message("dummy payload") == "feat(app): add print statement"


def test_generate_commit_message_tolerates_fenced_json(monkeypatch):
    _mock_ollama(
        monkeypatch,
        'Sure!\n```json\n{"type": "fix", "scope": "ui", "subject": "correct label"}\n```',
    )

    assert generate_commit_message("dummy payload") == "fix(ui): correct label"


def test_generate_commit_message_appends_body(monkeypatch):
    _mock_ollama(
        monkeypatch, '{"type": "feat", "scope": "ui", "subject": "add avatar"}'
    )

    result = generate_commit_message("dummy payload", body="Refs: PROJ-104")
    assert result == "feat(ui): add avatar\n\nRefs: PROJ-104"


def test_connection_error_raises_instead_of_returning_a_message(monkeypatch):
    """Bug #1: a failure must never be committable as a message."""
    import urllib.error

    def boom(req, timeout=60):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)

    with pytest.raises(OllamaError):
        generate_commit_message("dummy payload")


def test_invalid_type_is_rejected():
    """Bug #4: only the seven Conventional Commit types are accepted."""
    with pytest.raises(OllamaError):
        build_commit_message({"type": "database", "scope": "x", "subject": "y"})


def test_empty_subject_is_rejected():
    with pytest.raises(OllamaError):
        build_commit_message({"type": "feat", "scope": "x", "subject": "   "})


def test_non_json_response_raises(monkeypatch):
    _mock_ollama(monkeypatch, "I could not read the diff, sorry.")

    with pytest.raises(OllamaError):
        generate_commit_message("dummy payload")


def test_subject_is_truncated_to_72_chars():
    message = build_commit_message(
        {"type": "feat", "scope": "ui", "subject": "x" * 200}
    )

    assert len(message) <= 72
    assert message.startswith("feat(ui): ")


def test_scope_is_normalised():
    message = build_commit_message(
        {"type": "feat", "scope": "(UI)", "subject": "add thing."}
    )

    assert message == "feat(ui): add thing"


# --------------------------------------------------------------------------
# Branch / ticket context
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "branch,expected",
    [
        ("feature/PROJ-104-login", "PROJ-104"),
        ("bugfix/AB-12", "AB-12"),
        ("PROJ-9", "PROJ-9"),
        ("feature/proj-104-login", "PROJ-104"),
        ("main", None),
        ("feature/no-ticket-here", None),
        (None, None),
    ],
)
def test_extract_ticket_id(branch, expected):
    assert git_ops.extract_ticket_id(branch) == expected


def test_build_commit_body():
    assert git_ops.build_commit_body("PROJ-104") == "Refs: PROJ-104"
    assert git_ops.build_commit_body(None) is None
