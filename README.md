# ai-commit

An **agentic** git commit tool. It reads your staged diff, splits unrelated
changes into atomic per-domain commits, and writes a
[Conventional Commit](https://www.conventionalcommits.org/) message for each —
using a **local** LLM (Ollama). Nothing leaves your machine.

Staging `schema.prisma` + `UserCard.tsx` together produces two commits
(`feat(schema): …` and `feat(ui): …`), not one.

- **Zero dependencies** — standard library only, runs on the system Python.
- **Local inference** — Ollama on `localhost:11434`, default `qwen2.5-coder:1.5b`.
- **Crash-safe** — staging is backed up before any index surgery and restored
  on the next run if the process dies mid-split.

## Requirements

- Python 3.9+ (stdlib only — no `pip install`, no virtualenv needed to run).
- [Ollama](https://ollama.com) running locally with a code model pulled:
  ```bash
  ollama pull qwen2.5-coder:1.5b
  ```

## Install

The tool is stdlib-only, so symlink it onto your `PATH` — no venv, no `sudo`:

```bash
chmod +x /path/to/git-ai-commit/main.py
mkdir -p ~/.local/bin
ln -sf /path/to/git-ai-commit/main.py ~/.local/bin/ai-commit
```

Make sure `~/.local/bin` is on your `PATH` (add to `~/.zshrc` if missing):

```bash
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *)
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc ;;
esac
```

Now run `ai-commit` from any git repository.

> The shebang is pinned to `/usr/bin/python3` (the system interpreter) on
> purpose. `#!/usr/bin/env python3` would pick up whatever virtualenv is active
> in the calling shell — wrong for a tool you run from arbitrary repos.

## Usage

Stage your changes, then:

```bash
git add .
ai-commit
```

For each domain chunk you get an approve / edit / skip / quit prompt:

```
🧩 3 file(s) grouped into 2 atomic commit(s).

[1/2] database & migrations  (db)
  • prisma/schema.prisma
⚡ Generating message for db...

----------------------------------------
Suggested Commit Message:
feat(schema): add avatarUrl field to User model

Refs: PROJ-104
----------------------------------------
Commit this chunk? (y/e/n/q) [yes / edit / skip / quit]:
```

Skipped chunks are left staged so nothing is silently dropped.

### Flags

| Flag | Effect |
| --- | --- |
| `-y`, `--yes` | Approve every chunk without prompting (non-interactive). |
| `--dry-run` | Show the split and generated messages; commit nothing, touch nothing. |
| `--model MODEL` | Use a different Ollama model (default `qwen2.5-coder:1.5b`). |
| `--suggest FILE` | Write **one** message for all staged changes to `FILE` and exit. For git hooks — see below. |

```bash
ai-commit --dry-run          # preview the plan
ai-commit --yes              # commit every chunk unattended
ai-commit --model qwen2.5-coder:7b
```

## Ticket context

If the current branch name contains a ticket ID, it's appended to every commit
body:

```
feature/PROJ-104-avatars  ->  Refs: PROJ-104
```

Case-insensitive; prefixes like `feature/`, `release/`, `v2` are not mistaken
for tickets.

## Optional: git hook

To pre-fill the commit message every time you run a plain `git commit`, install
a `prepare-commit-msg` hook. It uses `--suggest`, so git still creates the
commit — the hook only fills in the message.

```bash
cat > .git/hooks/prepare-commit-msg <<'EOF'
#!/usr/bin/env bash
# Only fill an empty message: skip -m, merges, squashes, amends.
case "$2" in message|merge|squash|commit) exit 0 ;; esac
exec ~/.local/bin/ai-commit --suggest "$1" </dev/null
EOF
chmod +x .git/hooks/prepare-commit-msg
```

Notes:

- **Do not** write a hook that calls `git commit` itself — that re-triggers the
  hook and recurses infinitely.
- The hook produces a **single** message for all staged changes. Per-domain
  splitting only happens in the interactive `ai-commit` flow, because a hook
  must not reset your index.
- If Ollama is unreachable the hook leaves the message empty and the commit
  proceeds normally.

## How it works

| Module | Responsibility |
| --- | --- |
| `git_diff_parser.py` | Parse the unified diff into files/hunks, keeping a byte-faithful `full_patch` that re-applies cleanly with `git apply`. |
| `domain_chunker.py` | Map each file to a domain (`db`, `ui`, `api`, `test`, `docs`, `ci`, `config`, `deps`, `core`) by path/extension and group into commit-sized chunks. |
| `llm_client.py` | Call Ollama with JSON-constrained decoding, validate the type, and assemble the commit string in code. |
| `git_ops.py` | Index/commit plumbing and branch/ticket extraction. |
| `main.py` | Per-chunk approve/edit/skip loop, crash-safe backup/restore, CLI. |

### Crash safety

Committing per chunk means resetting the index and re-applying one patch at a
time. Before that starts, the full staged diff (`--binary`) is written to
`.git/ai-commit-backup.patch`. After each commit the backup shrinks to the
still-uncommitted chunks. If the process is killed mid-split, the next run
offers to restore your staging (via a 3-way apply, so already-committed changes
are skipped). Your **file contents are never at risk** — only the staging state.

## Development

```bash
python3 -m venv venv && ./venv/bin/pip install pytest
./venv/bin/pytest
```

The suite covers diff parsing (incl. paths with spaces, renames, binary),
`git apply` round-trips, domain classification, JSON parsing/validation,
ticket extraction, crash backup/restore, and the CLI/hook modes.
