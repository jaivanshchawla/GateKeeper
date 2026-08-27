#!/usr/bin/env python3
"""
Shared M.1 feature computation — the SINGLE source of truth.

Both bulk extraction (m1_compute_features.py) and single-commit extraction
(single_commit_features.py) call compute_m1_features() from this module.
Two implementations of one contract is the root cause of train/serve skew;
this module eliminates that by providing exactly one implementation.
"""

import subprocess
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Module-level identity resolver (initialized per-repo)
_identity_map: dict[str, str] = {}  # variant -> canonical
_identity_loaded_for: str = ""


def _load_identity_map(repo_path: str):
    """Load .mailmap-based identity map for a repo. Cached per process."""
    global _identity_map, _identity_loaded_for
    rp = str(Path(repo_path).resolve())
    if rp == _identity_loaded_for and _identity_map:
        return
    try:
        from policy.identity import build_identity_map
        _identity_map = build_identity_map(repo_path)
    except ImportError:
        _identity_map = {}
    _identity_loaded_for = rp


def resolve_author(repo_path: str, email: str) -> str:
    """Resolve an email to canonical form using .mailmap + heuristics."""
    _load_identity_map(repo_path)
    e = email.lower().strip()
    return _identity_map.get(e, e)


def normalize_author_id(identifier: str) -> str:
    """Normalize author identifier (email preferred, name fallback).

    Uses NFKD normalization, strips combining marks, casefolds.
    Applied identically in both bulk and single-commit paths.
    Keyed on email, not display name — emails are stable across
    git config changes and noreply addresses.
    """
    if not identifier:
        return ""
    # NFKD normalize, strip combining marks (accents), casefold
    nfkd = unicodedata.normalize("NFKD", identifier)
    stripped = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    return stripped.casefold()


def build_graph(repo_path: str, since: str, until: str) -> dict:
    """Build commit graph from git log — same format used by both paths.

    Returns: {hash: {date: datetime(naive UTC), files: [str], subject: str,
                      author: str, is_merge: bool}}
    """
    # Use %aE (email) instead of %aN (name) — emails are stable,
    # display names are not ("Esteban Küber" vs "Esteban Kuber",
    # "Lauren Tan" vs "lauren").
    fmt = "%H|%ct|%aE|%s"
    result = subprocess.run(
        ["git", "log", f"--since={since}", f"--until={until}",
         f"--pretty=format:{fmt}", "--name-only", "--no-merges", "HEAD"],
        cwd=repo_path, capture_output=True, text=True, timeout=600, check=False,
        encoding="utf-8", errors="replace",
    )

    graph: dict[str, dict] = {}
    ch = None
    cf: list[str] = []
    ct = 0
    ca = ""
    cs = ""

    for line in result.stdout.split("\n"):
        line = line.rstrip()
        if not line:
            continue
        parts = line.split("|", 3)
        if len(parts) == 4 and len(parts[0]) == 40:
            if ch is not None:
                graph[ch] = {
                    "date": datetime.fromtimestamp(ct, tz=timezone.utc).replace(tzinfo=None),
                    "files": cf, "subject": cs, "author": ca, "is_merge": False,
                }
            raw_email = parts[2]
            ch, ct, ca, cs = parts[0], int(parts[1]), resolve_author(repo_path, normalize_author_id(raw_email)), parts[3]
            cf = []
        else:
            cf.append(line)
    if ch is not None:
        graph[ch] = {
            "date": datetime.fromtimestamp(ct, tz=timezone.utc).replace(tzinfo=None),
            "files": cf, "subject": cs, "author": ca, "is_merge": False,
        }

    # Add merge commits (same as m1_compute_features.py)
    merge_result = subprocess.run(
        ["git", "log", f"--since={since}", f"--until={until}",
         f"--pretty=format:{fmt}", "--merges", "HEAD"],
        cwd=repo_path, capture_output=True, text=True, timeout=600, check=False,
        encoding="utf-8", errors="replace",
    )
    for line in merge_result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|", 3)
        if len(parts) == 4 and len(parts[0]) == 40:
            raw_email = parts[2]
            mh, mt, ma, ms = parts[0], int(parts[1]), resolve_author(repo_path, normalize_author_id(raw_email)), parts[3]
            if mh in graph:
                graph[mh]["is_merge"] = True
            else:
                graph[mh] = {
                    "date": datetime.fromtimestamp(mt, tz=timezone.utc).replace(tzinfo=None),
                    "files": [], "subject": ms, "author": ma, "is_merge": True,
                }
    return graph


def compute_risky_hashes(graph: dict, label_window_days: int = 7) -> set[str]:
    """Compute risky hashes from a complete graph.

    Matches rebuild_b2.py's label_from_graph logic exactly.
    """
    risky: set[str] = set()

    # Criterion 1: revert in subject
    for h, info in graph.items():
        if "revert" in info["subject"].lower():
            risky.add(h)

    # Criterion 2: file re-touch within window
    file_touches: dict[str, list[tuple[str, datetime]]] = defaultdict(list)
    for h, info in graph.items():
        cd = info["date"]
        for fp in info["files"]:
            file_touches[fp].append((h, cd))

    for touches in file_touches.values():
        if len(touches) < 2:
            continue
        touches.sort(key=lambda x: x[1])
        for i, (h_i, d_i) in enumerate(touches):
            if h_i in risky:
                continue
            for j in range(i + 1, len(touches)):
                h_j, d_j = touches[j]
                if (d_j - d_i).days <= label_window_days:
                    risky.add(h_i)
                    break
                else:
                    break
    return risky


def count_authors_before(repo_path: str, before_date: str) -> dict[str, int]:
    """Count commits per author (normalized email) before a given date.

    Used by both bulk (before window start) and SC (before commit date).
    Accepts either a date string (YYYY-MM-DD) or ISO-8601 timestamp.
    Uses --before (not --until) so ISO-8601 timestamps preserve the
    exact time, avoiding midnight truncation.
    Keyed on normalized email (%aE), not display name (%aN).
    """
    result = subprocess.run(
        ["git", "log", f"--before={before_date}", "--format=%aE",
         "--no-merges", "HEAD"],
        cwd=repo_path, capture_output=True, text=True, timeout=300,
        encoding="utf-8", errors="replace",
    )
    counts: dict[str, int] = defaultdict(int)
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line:
            canonical = resolve_author(repo_path, normalize_author_id(line))
            counts[canonical] += 1
    return dict(counts)


def walk_graph_to_state(graph: dict, risky_hashes: set[str],
                        stop_hash: str | None = None,
                        stop_date: datetime | None = None,
                        sorted_graph: list | None = None,
                        start_index: int = 0,
                        start_state: dict | None = None):
    """Walk graph chronologically, building running state.

    Stops BEFORE stop_hash (exclusive) or before stop_date (exclusive).
    Returns (state_dict, target_info) where target_info is the commit at the stop point.

    This is the SAME state-building logic used by m1_compute_features.py's
    compute_features_incremental, extracted for reuse by both paths.

    Args:
        sorted_graph: Optional pre-sorted list of (hash, info) tuples.
            If provided, skips the O(n log n) sort on every call.
        start_index: Index in sorted_graph to start from (for snapshot resume).
        start_state: Pre-built state dict to resume from (for snapshot resume).
    """
    if sorted_graph is None:
        sorted_graph = sorted(graph.items(), key=lambda x: x[1]["date"])

    if start_state:
        # Resume from snapshot
        file_change_count = defaultdict(int, start_state.get("file_change_count", {}))
        file_risky_count = defaultdict(int, start_state.get("file_risky_count", {}))
        file_revert_count = defaultdict(int, start_state.get("file_revert_count", {}))
        file_first_seen = dict(start_state.get("file_first_seen", {}))
        file_last_touch_hash = dict(start_state.get("file_last_touch_hash", {}))
        file_authors = defaultdict(set, {k: set(v) for k, v in start_state.get("file_authors", {}).items()})
        author_state_raw = start_state.get("author_state", {})
        author_state = defaultdict(
            lambda: {"files": defaultdict(int), "dirs": defaultdict(int), "last_date": None, "total": 0}
        )
        for author, ast in author_state_raw.items():
            author_state[author] = {
                "files": defaultdict(int, ast.get("files", {})),
                "dirs": defaultdict(int, ast.get("dirs", {})),
                "last_date": ast.get("last_date"),
                "total": ast.get("total", 0),
            }
        co_change = defaultdict(int, {tuple(k): v for k, v in start_state.get("co_change", {}).items()})
    else:
        # Running state — same variables as compute_features_incremental
        file_change_count: dict[str, int] = defaultdict(int)
        file_risky_count: dict[str, int] = defaultdict(int)
        file_revert_count: dict[str, int] = defaultdict(int)
        file_first_seen: dict[str, datetime] = {}
        file_last_touch_hash: dict[str, str] = {}
        file_authors: dict[str, set] = defaultdict(set)
        author_state: dict[str, dict] = defaultdict(
            lambda: {"files": defaultdict(int), "dirs": defaultdict(int), "last_date": None}
        )
        co_change: dict[tuple, int] = defaultdict(int)

    target_info = None

    for idx in range(start_index, len(sorted_graph)):
        h, v = sorted_graph[idx]
        files_in_this = set(v.get("files", []))
        author = v.get("author", "")
        is_risky = h in risky_hashes
        subj = v.get("subject", "")
        is_revert = "revert" in subj.lower()
        v_date = v["date"]

        # Check if this is the stop point
        should_stop = False
        if stop_hash and h == stop_hash:
            should_stop = True
            target_info = {
                "hash": h, "date": v_date, "author": author,
                "files": files_in_this, "is_merge": 1 if v.get("is_merge", False) else 0,
                "subject": subj,
            }
        elif stop_date and v_date >= stop_date:
            should_stop = True
            target_info = {
                "hash": h, "date": v_date, "author": author,
                "files": files_in_this, "is_merge": 1 if v.get("is_merge", False) else 0,
                "subject": v.get("subject", ""),
            }

        if should_stop:
            break

        # Update running state for ALL commits before the target
        for fp in files_in_this:
            if fp not in file_first_seen:
                file_first_seen[fp] = v_date
            file_last_touch_hash[fp] = h
            file_change_count[fp] += 1
            if is_risky:
                file_risky_count[fp] += 1
            if is_revert:
                file_revert_count[fp] += 1
            file_authors[fp].add(author)

        af = author_state[author]
        af["total"] = af.get("total", 0) + 1
        for fp in files_in_this:
            af["files"][fp] += 1
            d = str(Path(fp).parent)
            if d and d != ".":
                af["dirs"][d] += 1
        af["last_date"] = v_date

        fl = sorted(files_in_this)
        if 2 <= len(fl) <= 30:
            for i in range(len(fl)):
                for j in range(i + 1, len(fl)):
                    co_change[(fl[i], fl[j])] += 1

    state = {
        "file_change_count": file_change_count,
        "file_risky_count": file_risky_count,
        "file_revert_count": file_revert_count,
        "file_first_seen": file_first_seen,
        "file_last_touch_hash": file_last_touch_hash,
        "file_authors": file_authors,
        "author_state": author_state,
        "co_change": co_change,
    }
    return state, target_info


def compute_m1_features(
    state: dict,
    graph: dict,
    target_hash: str,
    target_date: datetime,
    author: str,
    files_touched: set,
    is_merge: int,
    risky_hashes: set[str],
) -> dict:
    """Compute ALL M.1 features for one commit from running state.

    THIS IS THE SINGLE SOURCE OF TRUTH. Both bulk and SC call this function.
    The state must represent all commits BEFORE the target commit.
    """
    import numpy as np

    # Normalize target_date
    cd = target_date
    if cd.tzinfo is not None:
        cd = cd.replace(tzinfo=None)

    fcc = state["file_change_count"]
    frc = state["file_risky_count"]
    frv = state["file_revert_count"]
    ffs = state["file_first_seen"]
    flth = state["file_last_touch_hash"]
    fa = state["file_authors"]
    ast = state["author_state"]
    co = state["co_change"]

    # ── M.1a: File-level history ──
    ch_vals = [fcc[f] for f in files_touched] if files_touched else [0]
    rk_vals = [frc[f] for f in files_touched] if files_touched else [0]
    rv_vals = [frv[f] for f in files_touched] if files_touched else [0]
    ag_vals = [(cd - ffs[f]).days for f in files_touched if f in ffs]
    au_vals = [len(fa[f]) for f in files_touched] if files_touched else [0]

    def mm(vals):
        return (max(vals), float(np.mean(vals))) if vals else (0, 0.0)

    fpc = mm(ch_vals)
    fpr = mm(rk_vals)
    frc_v = mm(rv_vals)
    fad = mm(ag_vals)
    fac = mm(au_vals)

    # days_since_last_change
    fld = []
    for fp in files_touched:
        lth = flth.get(fp)
        if lth and lth != target_hash and lth in graph:
            fld.append((cd - graph[lth]["date"]).days)
        else:
            fld.append(9999)

    # ── M.1b: Author-file familiarity ──
    af = ast[author]
    afc = []
    adc = []
    fft_f = 1
    fft_d = 1
    for fp in files_touched:
        c = af["files"][fp]
        afc.append(c)
        if c > 0:
            fft_f = 0
        d = str(Path(fp).parent)
        if d and d != ".":
            dc = af["dirs"][d]
            adc.append(dc)
            if dc > 0:
                fft_d = 0
    adl = max(0, (cd - af["last_date"]).days) if af["last_date"] else 9999

    # ── M.1d: Co-change ──
    fl = sorted(files_touched)
    co_vals = []
    if 2 <= len(fl) <= 30:
        for i in range(len(fl)):
            for j in range(i + 1, len(fl)):
                co_vals.append(co.get((fl[i], fl[j]), 0))

    n_files = len(files_touched) if files_touched else 0
    dirs = set()
    for fp in (files_touched or set()):
        d = str(Path(fp).parent)
        if d and d != ".":
            dirs.add(d)
    n_dirs = max(len(dirs), 1)

    return {
        "file_prior_changes_max": fpc[0],
        "file_prior_changes_mean": fpc[1],
        "file_prior_risky_max": fpr[0],
        "file_prior_risky_mean": fpr[1],
        "file_revert_count_max": frc_v[0],
        "file_revert_count_mean": frc_v[1],
        "file_age_days_max": fad[0],
        "file_age_days_mean": fad[1],
        "file_authors_count_max": fac[0],
        "file_authors_count_mean": fac[1],
        "days_since_last_change_max": max(fld) if fld else 0,
        "days_since_last_change_mean": float(np.mean(fld)) if fld else 0.0,
        "author_file_prior_commits_max": max(afc) if afc else 0,
        "author_file_prior_commits_mean": float(np.mean(afc)) if afc else 0.0,
        "author_dir_prior_commits_max": max(adc) if adc else 0,
        "author_dir_prior_commits_mean": float(np.mean(adc)) if adc else 0.0,
        "is_author_first_touch_dir": fft_d,
        "author_days_since_last_commit": adl,
        "churn_ratio": 0,  # placeholder — computed from CSV columns, not graph
        "change_entropy": 0,
        "max_file_churn": 0,
        "is_test_only": 0,
        "test_to_code_ratio": 0,
        "config_touch": 0,
        "is_merge": is_merge,
        "files_per_dir_ratio": n_files / n_dirs if n_dirs > 0 else 0,
        "co_change_strength_max": max(co_vals) if co_vals else 0,
        "co_change_strength_mean": float(np.mean(co_vals)) if co_vals else 0.0,
    }


def compute_change_shape(lines_added: int, lines_deleted: int,
                         files_touched_count: int, touched_files: set) -> dict:
    """Compute change-shape features — matches history_features.py EXACTLY.

    Called by both paths with the same inputs.
    """
    import numpy as np

    churn_ratio = lines_deleted / (lines_added + 1)

    if files_touched_count > 1:
        p = 1.0 / files_touched_count
        change_entropy = -files_touched_count * p * np.log2(p)
    else:
        change_entropy = 0.0

    max_file_churn = (lines_added + lines_deleted) / files_touched_count if files_touched_count > 0 else 0

    test_patterns = ("test", "spec", "_test.", "_spec.", "tests/", "test_", "__tests__")
    config_patterns = (".yaml", ".yml", ".toml", ".lock", "Dockerfile", ".github/",
                       "docker-compose", "Makefile", ".env", ".ini", ".cfg", "setup.py",
                       "setup.cfg", "pyproject.toml", "package.json", "Cargo.toml")

    test_count = 0
    config_count = 0
    for fp in touched_files:
        fp_lower = fp.lower()
        if any(pat in fp_lower for pat in test_patterns):
            test_count += 1
        if any(pat in fp_lower for pat in config_patterns):
            config_count += 1

    is_test_only = 1 if files_touched_count > 0 and test_count == files_touched_count else 0
    test_to_code_ratio = test_count / files_touched_count if files_touched_count > 0 else 0
    config_touch = 1 if config_count > 0 else 0

    return {
        "churn_ratio": churn_ratio,
        "change_entropy": change_entropy,
        "max_file_churn": max_file_churn,
        "is_test_only": is_test_only,
        "test_to_code_ratio": test_to_code_ratio,
        "config_touch": config_touch,
    }
