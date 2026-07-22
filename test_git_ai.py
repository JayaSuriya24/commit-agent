import json
import os
import subprocess

import pytest

import git_ops
import main
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


# --------------------------------------------------------------------------
# Crash protection
# --------------------------------------------------------------------------


def test_backup_survives_simulated_crash_and_restores(git_repo, monkeypatch):
    """Kill the process after the reset: staging must be recoverable."""
    repo, git = git_repo
    monkeypatch.chdir(repo)
    (repo / "app.py").write_text("import os\nimport sys\nprint(os.getcwd())\n")
    git("add", "-A")
    original = git("diff", "--cached").stdout

    main.create_emergency_backup()
    assert os.path.exists(main.backup_patch_path())

    git("reset", "-q")  # simulate the crash window: index cleared, then death
    assert git("diff", "--cached", "--quiet").returncode == 0

    monkeypatch.setattr("builtins.input", lambda _: "y")
    main.check_for_recovery()

    assert git("diff", "--cached").stdout == original
    assert not os.path.exists(main.backup_patch_path())  # cleaned after success


def test_backup_is_binary_safe(git_repo, monkeypatch):
    """A plain `git diff --cached` would drop binary content from the backup."""
    repo, git = git_repo
    monkeypatch.chdir(repo)
    (repo / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x01\x02\x03blob")
    git("add", "-A")

    main.create_emergency_backup()
    patch = open(main.backup_patch_path(), encoding="utf-8").read()
    assert "GIT binary patch" in patch

    git("reset", "-q")
    monkeypatch.setattr("builtins.input", lambda _: "y")
    main.check_for_recovery()

    assert git("diff", "--cached", "--name-only").stdout.strip() == "logo.png"
    assert (repo / "logo.png").read_bytes().startswith(b"\x89PNG")


def test_failed_restore_keeps_the_backup(git_repo, monkeypatch):
    """The backup is the only copy: never delete it on a failed apply."""
    repo, git = git_repo
    monkeypatch.chdir(repo)
    path = os.path.join(git_ops.git_dir(), main.BACKUP_PATCH_NAME)
    with open(path, "w") as f:
        f.write("this is not a valid patch\n")

    monkeypatch.setattr("builtins.input", lambda _: "y")
    main.check_for_recovery()

    assert os.path.exists(path), "backup was deleted despite the restore failing"


def test_recovery_declined_keeps_backup(git_repo, monkeypatch):
    """Answering anything but y/d leaves the file for a later decision."""
    repo, git = git_repo
    monkeypatch.chdir(repo)
    (repo / "app.py").write_text("changed\n")
    git("add", "-A")
    main.create_emergency_backup()

    monkeypatch.setattr("builtins.input", lambda _: "n")
    main.check_for_recovery()
    assert os.path.exists(main.backup_patch_path())

    monkeypatch.setattr("builtins.input", lambda _: "d")
    main.check_for_recovery()
    assert not os.path.exists(main.backup_patch_path())


def test_refresh_backup_drops_committed_chunks(git_repo, monkeypatch):
    """After a chunk lands, HEAD moved: the backup must shrink to what's left."""
    repo, git = git_repo
    monkeypatch.chdir(repo)
    (repo / "app.py").write_text("import os\nimport sys\nprint(os.getcwd())\n")
    (repo / "notes.md").write_text("# Notes\n\nMore.\n")
    git("add", "-A")

    chunks = chunk_by_domain(parse_git_diff(git("diff", "--cached").stdout))
    main.create_emergency_backup()

    main.refresh_backup(chunks[1:])  # pretend chunks[0] was committed
    patch = open(main.backup_patch_path(), encoding="utf-8").read()
    assert chunks[1]["paths"][0] in patch
    assert chunks[0]["paths"][0] not in patch

    main.refresh_backup([])  # nothing left -> backup removed
    assert not os.path.exists(main.backup_patch_path())


def test_stale_backup_recovers_after_a_chunk_already_committed(git_repo, monkeypatch):
    """The real crash window: killed between `git commit` and the backup refresh.

    The backup then names a change that is already in HEAD. A strict apply
    rejects the whole patch; the 3-way restore must skip the committed file and
    re-stage only what is genuinely uncommitted.
    """
    repo, git = git_repo
    monkeypatch.chdir(repo)
    (repo / "app.py").write_text("import os\nimport sys\nprint(os.getcwd())\n")
    (repo / "notes.md").write_text("# Notes\n\nMore.\n")
    git("add", "-A")

    chunks = chunk_by_domain(parse_git_diff(git("diff", "--cached").stdout))
    main.create_emergency_backup()  # backup covers BOTH files
    git("reset", "-q")

    # One chunk lands, then the process dies before refresh_backup() runs.
    git("apply", "--cached", "-", input=chunks[0]["patch"])
    git("commit", "-q", "-m", "docs: notes")
    committed_path = chunks[0]["paths"][0]

    backup = open(main.backup_patch_path(), encoding="utf-8").read()
    assert committed_path in backup, "precondition: backup is stale"

    monkeypatch.setattr("builtins.input", lambda _: "y")
    main.check_for_recovery()

    staged = git_ops.staged_paths()
    assert staged == chunks[1]["paths"], f"expected only the uncommitted chunk, got {staged}"
    assert not os.path.exists(main.backup_patch_path())


def test_backup_path_resolves_from_a_subdirectory(git_repo, monkeypatch):
    """A hardcoded '.git/...' would crash or write a stray file from a subdir."""
    repo, git = git_repo
    subdir = repo / "src" / "deep"
    subdir.mkdir(parents=True)
    monkeypatch.chdir(subdir)

    path = main.backup_patch_path()
    assert os.path.isabs(path)
    assert os.path.dirname(path) == str(repo / ".git")


# --------------------------------------------------------------------------
# CLI flags and hook (--suggest) mode
# --------------------------------------------------------------------------


def test_args_default_to_interactive_committing():
    opts = main._parse_args([])
    assert opts.yes is False
    assert opts.dry_run is False
    assert opts.suggest is None
    assert opts.model  # a default model is set


def test_args_parse_flags():
    opts = main._parse_args(["--yes", "--model", "llama3", "--suggest", "MSG"])
    assert opts.yes and opts.model == "llama3" and opts.suggest == "MSG"


def _stub_llm(monkeypatch, text="feat(x): generated"):
    monkeypatch.setattr(main, "generate_commit_message", lambda *a, **k: text)


def test_suggest_fills_empty_template_and_keeps_comments(tmp_path, monkeypatch):
    _stub_llm(monkeypatch)
    msg = tmp_path / "COMMIT_EDITMSG"
    msg.write_text("\n# Please enter a commit message.\n# Lines with # are ignored.\n")
    opts = main._parse_args(["--suggest", str(msg)])

    parsed = parse_git_diff(SAMPLE_RAW_DIFF)
    assert main._suggest_only(parsed, "Refs: X-1", opts) == 0

    content = msg.read_text()
    assert content.startswith("feat(x): generated\n")
    assert "# Please enter a commit message." in content  # git's block preserved


def test_suggest_never_clobbers_a_real_message(tmp_path, monkeypatch):
    """A -m/merge/amend message already in the file must survive untouched."""
    _stub_llm(monkeypatch)
    msg = tmp_path / "COMMIT_EDITMSG"
    msg.write_text("my hand-written message\n\n# a comment\n")
    opts = main._parse_args(["--suggest", str(msg)])

    main._suggest_only(parse_git_diff(SAMPLE_RAW_DIFF), None, opts)
    assert msg.read_text() == "my hand-written message\n\n# a comment\n"


def test_suggest_leaves_file_alone_when_inference_fails(tmp_path, monkeypatch):
    """A hook must never block the commit: on LLM failure, don't touch the file."""

    def boom(*a, **k):
        raise OllamaError("ollama down")

    monkeypatch.setattr(main, "generate_commit_message", boom)
    msg = tmp_path / "COMMIT_EDITMSG"
    msg.write_text("\n# template\n")
    opts = main._parse_args(["--suggest", str(msg)])

    assert main._suggest_only(parse_git_diff(SAMPLE_RAW_DIFF), None, opts) == 0
    assert msg.read_text() == "\n# template\n"  # unchanged, git falls back


def test_dry_run_and_yes_skip_on_llm_failure_without_prompting(monkeypatch):
    """Non-interactive modes must not call input() when generation fails."""

    def boom(*a, **k):
        raise OllamaError("down")

    monkeypatch.setattr(main, "generate_commit_message", boom)

    def no_input(_):
        raise AssertionError("input() must not be called in non-interactive mode")

    monkeypatch.setattr("builtins.input", no_input)
    chunk = {"files": [], "domain": "x", "scope_hint": None, "type_hint": None}

    assert main._resolve_message(chunk, None, main._parse_args(["--yes"])) is None
    assert main._resolve_message(chunk, None, main._parse_args(["--dry-run"])) is None
