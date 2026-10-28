#!/usr/bin/env python3
"""The one place a repository is declared.

Five repositories used to be spelled out by hand in eight files: the API's
repo list, the two dataset builders, two feature pipelines, the retraining
DAG's default arguments and two test modules. Adding a sixth was a memory
test -- miss one list and the API served five repositories while the
pipeline mined six, with no error to say which was right.

``registry/repos.yaml`` is now the only place a repository is declared, and
every one of those call sites asks this module instead. Adding a repository
is one YAML entry; ``tests/test_registry.py`` fails if a module goes back to
writing the list out by hand.

Usage::

    from registry import get, repo_names, repos, urls

    repo_names()             # ('django', 'react', ...) in declaration order
    repo_names(role="train") # just the training corpus
    repos()                  # (Repo(...), ...)
    get("django").path       # <project>/repos/django
    urls()                   # clone URLs, in declaration order

The file can be relocated with ``GATEKEEPER_REGISTRY=/path/to/repos.yaml``,
which the tests use and a deployment that mounts its registry elsewhere
would too.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

# registry/__init__.py -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = PROJECT_ROOT / "registry" / "repos.yaml"
ENV_VAR = "GATEKEEPER_REGISTRY"

# A repository's part in the project. `train` rows build the model, `validate`
# rows are scored but never trained on, and `holdout` rows are touched only by
# the final evaluation. Anything else is a typo caught at load time.
ROLES = ("train", "validate", "holdout")

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_ALLOWED_KEYS = {"name", "url", "clone_dir", "role", "description"}


class RegistryError(ValueError):
    """The registry file is missing, malformed or internally inconsistent."""


@dataclass(frozen=True)
class Repo:
    """One declared repository.

    ``clone_dir`` is stored the way it is written in YAML (a POSIX path
    relative to the project root); ``path`` resolves it, so callers never
    rebuild the path by hand.
    """

    name: str
    url: str
    clone_dir: str
    role: str = "train"
    description: str = ""

    @property
    def path(self) -> Path:
        return PROJECT_ROOT / self.clone_dir

    def as_mapping(self) -> dict[str, str]:
        """The shape older call sites expect (name/url/clone_dir)."""
        return {"name": self.name, "url": self.url, "clone_dir": self.clone_dir}


_cache: dict[tuple[str, int], tuple[Repo, ...]] = {}


def _path(override: str | Path | None) -> Path:
    if override is not None:
        return Path(override)
    return Path(os.environ.get(ENV_VAR) or DEFAULT_PATH)


def _fail(path: Path, where: str, message: str) -> RegistryError:
    return RegistryError(f"{path}: {where}: {message}")


def _parse_entry(path: Path, index: int, entry: object) -> Repo:
    where = f"repos[{index}]"
    if not isinstance(entry, dict):
        raise _fail(path, where, f"expected a mapping, got {type(entry).__name__}")

    unknown = sorted(set(entry) - _ALLOWED_KEYS)
    if unknown:
        raise _fail(
            path,
            where,
            f"unknown key(s) {', '.join(unknown)}; allowed: {', '.join(sorted(_ALLOWED_KEYS))}",
        )

    name = entry.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise _fail(path, where, f"name must match {_NAME_RE.pattern!r}, got {name!r}")

    url = entry.get("url")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise _fail(path, where, f"{name}: url must be an https:// clone URL, got {url!r}")

    clone_dir = entry.get("clone_dir", f"repos/{name}")
    if not isinstance(clone_dir, str) or not clone_dir:
        raise _fail(path, where, f"{name}: clone_dir must be a non-empty string")
    parts = PurePosixPath(clone_dir).parts
    if PurePosixPath(clone_dir).is_absolute() or ".." in parts:
        raise _fail(
            path,
            where,
            f"{name}: clone_dir must stay inside the project, got {clone_dir!r}",
        )
    if "\\" in clone_dir:
        raise _fail(path, where, f"{name}: clone_dir must use forward slashes, got {clone_dir!r}")

    role = entry.get("role", "train")
    if role not in ROLES:
        raise _fail(path, where, f"{name}: role must be one of {', '.join(ROLES)}, got {role!r}")

    description = entry.get("description", "")
    if not isinstance(description, str):
        raise _fail(path, where, f"{name}: description must be a string")

    return Repo(
        name=name,
        url=url,
        clone_dir=str(PurePosixPath(clone_dir)),
        role=role,
        description=description,
    )


def load(path: str | Path | None = None) -> tuple[Repo, ...]:
    """Every declared repository, in declaration order.

    Results are cached against the file's modification time, so a long-lived
    API process does not re-read the YAML on every request while an edit on
    disk is still picked up on the next call.
    """
    registry_path = _path(path)
    try:
        stat = registry_path.stat()
    except FileNotFoundError:
        raise RegistryError(
            f"{registry_path}: registry file not found; "
            f"restore it or point {ENV_VAR} at one"
        ) from None

    key = (str(registry_path.resolve()), stat.st_mtime_ns)
    if key in _cache:
        return _cache[key]

    try:
        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise _fail(registry_path, "top level", f"not valid YAML: {exc}") from None

    if not isinstance(raw, dict):
        raise _fail(registry_path, "top level", "expected a mapping with a 'repos' list")

    entries = raw.get("repos")
    if not isinstance(entries, list) or not entries:
        raise _fail(registry_path, "repos", "expected a non-empty list of repositories")

    parsed = tuple(_parse_entry(registry_path, i, entry) for i, entry in enumerate(entries))

    seen: dict[str, str] = {}
    for repo in parsed:
        if repo.name in seen:
            raise _fail(
                registry_path,
                "repos",
                f"duplicate name {repo.name!r} (also declared with clone_dir {seen[repo.name]!r})",
            )
        seen[repo.name] = repo.clone_dir

    # One clone directory shared by two repositories would make the second
    # clone a no-op that silently trains on the first repository's history.
    clones: dict[str, str] = {}
    for repo in parsed:
        if repo.clone_dir in clones:
            raise _fail(
                registry_path,
                "repos",
                f"{repo.name!r} and {clones[repo.clone_dir]!r} share clone_dir {repo.clone_dir!r}",
            )
        clones[repo.clone_dir] = repo.name

    _cache[key] = parsed
    return parsed


def repos(role: str | None = None, path: str | Path | None = None) -> tuple[Repo, ...]:
    """Declared repositories, optionally filtered to one role."""
    if role is not None and role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}, got {role!r}")
    declared = load(path)
    if role is None:
        return declared
    return tuple(repo for repo in declared if repo.role == role)


def repo_names(role: str | None = None, path: str | Path | None = None) -> tuple[str, ...]:
    """Just the names, in declaration order."""
    return tuple(repo.name for repo in repos(role=role, path=path))


def urls(role: str | None = None, path: str | Path | None = None) -> tuple[str, ...]:
    """Clone URLs, in declaration order."""
    return tuple(repo.url for repo in repos(role=role, path=path))


def get(name: str, path: str | Path | None = None) -> Repo:
    """One repository by name; a helpful error if it is not declared."""
    for repo in load(path):
        if repo.name == name:
            return repo
    declared = ", ".join(repo_names(path=path))
    raise KeyError(f"repository {name!r} is not in the registry; declared: {declared}")


def clone_dir(name: str, path: str | Path | None = None) -> Path:
    """The local checkout path of a declared repository."""
    return get(name, path=path).path


def clear_cache() -> None:
    """Drop cached parses (tests that rewrite the file use this)."""
    _cache.clear()


__all__ = [
    "DEFAULT_PATH",
    "ENV_VAR",
    "PROJECT_ROOT",
    "ROLES",
    "RegistryError",
    "Repo",
    "clear_cache",
    "clone_dir",
    "get",
    "load",
    "repo_names",
    "repos",
    "urls",
]
