#!/usr/bin/env python3
"""Heuristic, path-based grouping of changed files into commit-sized domains.

No AST here on purpose: a file's path and extension already predict its domain
well enough to split "schema.prisma + UserCard.tsx" into two atomic commits.
"""
import os
import re
from typing import Any, Dict, List, NamedTuple, Optional

from git_diff_parser import build_combined_patch, file_path_of


class Domain(NamedTuple):
    key: str
    scope: Optional[str]  # scope hint for the commit message
    type_hint: str  # suggested Conventional Commit type
    order: int  # commit ordering: dependencies first
    label: str


DOMAINS: Dict[str, Domain] = {
    "db": Domain("db", "schema", "feat", 10, "database & migrations"),
    "api": Domain("api", "api", "feat", 20, "API surface"),
    "core": Domain("core", None, "feat", 30, "core logic"),
    "ui": Domain("ui", "ui", "feat", 40, "user interface"),
    "test": Domain("test", "tests", "test", 50, "tests"),
    "docs": Domain("docs", "docs", "docs", 60, "documentation"),
    "ci": Domain("ci", "ci", "chore", 70, "CI pipelines"),
    "config": Domain("config", "config", "chore", 80, "configuration"),
    "deps": Domain("deps", "deps", "chore", 90, "dependencies"),
}

# Ordered: first match wins. Tests must precede ui so Button.test.tsx is a test.
RULES: List[tuple] = [
    (
        "deps",
        r"(^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|"
        r"Cargo\.lock|go\.sum|Gemfile\.lock|Pipfile\.lock|requirements[^/]*\.txt)$",
    ),
    (
        "ci",
        r"(^|/)\.github/workflows/|(^|/)\.circleci/|"
        r"(^|/)(\.gitlab-ci\.yml|\.travis\.yml|azure-pipelines\.yml|Jenkinsfile)$",
    ),
    (
        "test",
        r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]+\.[a-z]+$|"
        r"(^|/)[^/]+_test\.[a-z]+$|\.(test|spec)\.[jt]sx?$|(^|/)conftest\.py$",
    ),
    (
        "docs",
        r"\.(md|mdx|rst|adoc)$|(^|/)docs?/|"
        r"(^|/)(README|LICENSE|CHANGELOG|CONTRIBUTING|AUTHORS)(\.[^/]*)?$",
    ),
    (
        "db",
        r"\.(prisma|sql)$|(^|/)(migrations?|alembic)/|(^|/)models?/|"
        r"(^|/)(schema\.rb|schema\.sql)$",
    ),
    (
        "ui",
        r"\.(tsx|jsx|vue|svelte|css|scss|sass|less|styl)$|"
        r"(^|/)(components?|pages|views|ui|templates)/",
    ),
    (
        "api",
        r"(^|/)(api|routes?|controllers?|handlers?|endpoints?|serializers?)/|\.proto$",
    ),
    (
        "config",
        r"\.(toml|ya?ml|ini|cfg|conf)$|"
        r"(^|/)(package\.json|tsconfig\.json|setup\.py|pyproject\.toml|Makefile|"
        r"Dockerfile[^/]*|docker-compose\.ya?ml)$|(^|/)\.env|(^|/)\.[^/]+rc$|"
        r"(^|/)\.gitignore$",
    ),
]

COMPILED = [(key, re.compile(pattern, re.IGNORECASE)) for key, pattern in RULES]


def classify_path(path: str) -> str:
    """Map one file path to a domain key. Unmatched paths fall back to 'core'."""
    normalized = path.replace(os.sep, "/")
    while normalized.startswith("./"):  # not lstrip("./"): that eats .github's dot
        normalized = normalized[2:]
    for key, pattern in COMPILED:
        if pattern.search(normalized):
            return key
    return "core"


def _derive_scope(domain: Domain, files: List[Dict[str, Any]]) -> Optional[str]:
    """Use the domain's scope, or infer one from the paths for 'core'."""
    if domain.scope:
        return domain.scope

    paths = [file_path_of(f).replace(os.sep, "/") for f in files]
    dirs = {os.path.dirname(p) for p in paths}
    if len(dirs) == 1:
        only = dirs.pop()
        if only:
            return os.path.basename(only)
    if len(paths) == 1:
        stem = os.path.basename(paths[0]).split(".")[0]
        return stem or None
    return None


def chunk_by_domain(parsed_diffs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group parsed files into one chunk per domain, in commit order.

    Each chunk is ready to commit: it carries the files, a combined applyable
    patch, and the scope/type hints handed to the LLM.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for file in parsed_diffs:
        buckets.setdefault(classify_path(file_path_of(file)), []).append(file)

    chunks = []
    for key, files in buckets.items():
        domain = DOMAINS[key]
        chunks.append(
            {
                "domain": key,
                "label": domain.label,
                "scope_hint": _derive_scope(domain, files),
                "type_hint": domain.type_hint,
                "order": domain.order,
                "files": files,
                "paths": [file_path_of(f) for f in files],
                "patch": build_combined_patch(files),
            }
        )

    chunks.sort(key=lambda c: (c["order"], c["domain"]))
    return chunks
