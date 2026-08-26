# ruff: noqa: E402,E701,E702,I001
#!/usr/bin/env python3
"""
M.1: Compute missing features INCREMENTALLY (O(n) per repo) and evaluate.

Walks commits chronologically, maintains running state. No O(n^2) scans.
"""

import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

try:
    import lightgbm as lgb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])
    import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOS_DIR = PROJECT_ROOT / "repos"
DATA_DIR = PROJECT_ROOT / "data"
REPO_NAMES = ["django", "react", "rust", "kubernetes", "kafka"]


def run_git(repo_path: str, args: list[str], timeout: int = 600) -> str:
    """Run a git command with UTF-8 encoding."""
    r = subprocess.run(
        ["git"] + args,
        cwd=repo_path, capture_output=True, timeout=timeout, check=False,
        encoding="utf-8", errors="replace",
    )
    return r.stdout


def build_graph(repo_path: str, since: str, until: str) -> dict:
    """Build commit graph with file paths, author (normalized email), dates, merge info."""
    # Use %aE (email) instead of %aN (name) — emails are stable,
    # display names vary. Normalized via normalize_author_id.
    from ml.m1_shared import normalize_author_id
    fmt = "%H|%ct|%aE|%s"
    stdout = run_git(repo_path, ["log", f"--since={since}", f"--until={until}",
                                  f"--pretty=format:{fmt}", "--name-only", "--no-merges", "HEAD"])
    graph = {}
    ch = None
    cf = []
    ct = 0
    ca = ""
    cs = ""
    for line in stdout.split("\n"):
        line = line.rstrip()
        if not line:
            continue
        parts = line.split("|", 3)
        if len(parts) == 4 and len(parts[0]) == 40:
            if ch is not None:
                graph[ch] = {"date": datetime.fromtimestamp(ct, tz=timezone.utc).replace(tzinfo=None),
                             "files": cf, "subject": cs, "author": ca, "is_merge": False}
            ch, ct, ca, cs = parts[0], int(parts[1]), normalize_author_id(parts[2]), parts[3]
            cf = []
        else:
            cf.append(line)
    if ch is not None:
        graph[ch] = {"date": datetime.fromtimestamp(ct, tz=timezone.utc).replace(tzinfo=None),
                      "files": cf, "subject": cs, "author": ca, "is_merge": False}

    # Merge commits — batch in a single git log call (avoids 24K individual calls for Rust)
    merge_fmt = "%H|%ct|%aE|%s"
    merge_stdout = run_git(repo_path, ["log", f"--since={since}", f"--until={until}",
                                        f"--pretty=format:{merge_fmt}", "--merges", "HEAD"])
    for line in merge_stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|", 3)
        if len(parts) == 4 and len(parts[0]) == 40:
            mh, mt, ma, ms = parts[0], int(parts[1]), normalize_author_id(parts[2]), parts[3]
            if mh in graph:
                graph[mh]["is_merge"] = True
            else:
                graph[mh] = {"date": datetime.fromtimestamp(mt, tz=timezone.utc).replace(tzinfo=None),
                             "files": [], "subject": ms, "author": ma, "is_merge": True}
    return graph


def compute_features_incremental(repo_name: str, df: pd.DataFrame, graph: dict) -> pd.DataFrame:
    """Compute all M.1 features INCREMENTALLY in one chronological pass."""
    repo_df = df[df["source_repo"] == repo_name].copy()
    # R.1 FIX: compute risky per-repo from graph, NOT pooled from CSV.
    # The CSV's risky column is correct per-repo, but pooling all repos
    # gives a different set than the per-repo graph computation.
    from ml.m1_shared import compute_risky_hashes
    risky_hashes = compute_risky_hashes(graph)

    # Sort graph chronologically
    sorted_graph = sorted(graph.items(), key=lambda x: x[1]["date"])

    # ── Running state ──────────────────────────────────────────────
    file_change_count = defaultdict(int)
    file_risky_count = defaultdict(int)
    file_revert_count = defaultdict(int)
    file_first_seen = {}
    file_last_touch_hash = {}
    file_authors = defaultdict(set)
    author_state = defaultdict(lambda: {"files": defaultdict(int), "dirs": defaultdict(int), "last_date": None})
    co_change = defaultdict(int)

    hash_to_row = {}
    for _, row in repo_df.iterrows():
        hash_to_row[row["hash"]] = row

    # Process ALL graph entries chronologically (to maintain file state),
    # but only compute features for CSV entries
    results = []
    for h, v in sorted_graph:
        # Always update state for file tracking
        files_in_this = set(v.get("files", []))
        author = v.get("author", "")
        is_risky = h in risky_hashes
        subj = v.get("subject", "")
        is_revert = "revert" in subj.lower()

        # Only compute features if this commit is in our CSV
        if h in hash_to_row:
            row = hash_to_row[h]
            cd = pd.to_datetime(row["committer_date"])
            if cd.tzinfo:
                cd = cd.astimezone(timezone.utc).tz_localize(None)
            cd = cd.replace(tzinfo=None)
            author = row.get("author", "")
            files_touched = files_in_this
            is_merge = 1 if v.get("is_merge", False) else 0

            # Build state dict from running variables
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

            # Call shared function — SAME code path as single-commit extraction
            from ml.m1_shared import compute_m1_features
            features = compute_m1_features(
                state=state, graph=graph, target_hash=h,
                target_date=cd, author=author,
                files_touched=files_touched, is_merge=is_merge,
                risky_hashes=risky_hashes,
            )
            # Remove co-change (not in current config)
            features.pop("co_change_strength_max", None)
            features.pop("co_change_strength_mean", None)
            features["hash"] = h
            results.append(features)

        # ── Update running state for ALL commits (needed for file tracking) ──
        for fp in files_in_this:
            if fp not in file_first_seen:
                file_first_seen[fp] = v["date"]
            file_last_touch_hash[fp] = h
            file_change_count[fp] += 1
            if is_risky:
                file_risky_count[fp] += 1
            if is_revert:
                file_revert_count[fp] += 1
            file_authors[fp].add(author)

        af2 = author_state[author]
        for fp in files_in_this:
            af2["files"][fp] += 1
            d = str(Path(fp).parent)
            if d and d != ".":
                af2["dirs"][d] += 1
        af2["last_date"] = v["date"]

        # Update co-change
        fl2 = sorted(files_in_this)
        if 2 <= len(fl2) <= 30:
            for i in range(len(fl2)):
                for j in range(i + 1, len(fl2)):
                    co_change[(fl2[i], fl2[j])] += 1

    # Align results with repo_df index — map by hash
    result_df = pd.DataFrame(results)
    if "hash" in result_df.columns:
        result_df = result_df.set_index("hash")
        # Reindex to match repo_df's hash order
        result_df = result_df.reindex(repo_df["hash"].values, fill_value=0)
        result_df.index = repo_df.index
    return result_df


def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)
    samples = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(set(y_true[idx])) < 2:
            continue
        samples.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return float(np.mean(samples)), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def train_evaluate(X_train, y_train, X_test, y_test):
    model = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100,
                                random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "y_prob": y_prob,
    }


def loro_eval(X, y, repos, feat_cols, gname):
    all_repos = sorted(set(repos))
    repo_m = {}
    for held in all_repos:
        m_tr = repos != held
        m_te = repos == held
        Xtr, ytr = X.loc[m_tr, feat_cols].values, y[m_tr].values
        Xte, yte = X.loc[m_te, feat_cols].values, y[m_te].values
        if len(yte) == 0 or len(ytr) == 0:
            continue
        repo_m[held] = train_evaluate(Xtr, ytr, Xte, yte)

    all_yt, all_yp = [], []
    for held in all_repos:
        if held not in repo_m:
            continue
        m_te = repos == held
        all_yt.extend(y[m_te].values.tolist())
        all_yp.extend(repo_m[held]["y_prob"].tolist())
    all_yt, all_yp = np.array(all_yt), np.array(all_yp)
    roc_m, roc_lo, roc_hi = bootstrap_auc_ci(all_yt, all_yp)
    base_rate = float(np.mean(all_yt))
    mm = {k: float(np.mean([repo_m[r][k] for r in repo_m])) for k in
           ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"]}
    return {"n_features": len(feat_cols), "per_repo": repo_m, "mean": mm,
            "roc_auc_ci": (roc_m, roc_lo, roc_hi),
            "pr_auc_lift": mm["pr_auc"] - base_rate, "base_rate": base_rate}


def main():
    print("=" * 80)
    print("M.1: FEATURE EXPANSION + CUMULATIVE LORO EVALUATION (incremental)")
    print("=" * 80)

    df = pd.read_csv(DATA_DIR / "commit_features.csv")
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    since, until = "2024-07-01", "2026-07-07"
    all_new = []
    for rn in REPO_NAMES:
        t0 = time.time()
        rp = str(REPOS_DIR / rn)
        graph = build_graph(rp, since, until)
        t_graph = time.time() - t0
        t1 = time.time()
        feats = compute_features_incremental(rn, df, graph)
        t_feat = time.time() - t1
        print(f"  {rn}: graph={len(graph)} ({t_graph:.1f}s), features={feats.shape} ({t_feat:.1f}s)")
        all_new.append(feats)

    new_features = pd.concat(all_new, ignore_index=True)
    enhanced = pd.concat([df.reset_index(drop=True), new_features.reset_index(drop=True)], axis=1)
    enhanced.to_csv(DATA_DIR / "commit_features_m1.csv", index=False)
    print(f"\nSaved {enhanced.shape}")

    # Feature groups
    base = ["lines_added", "lines_deleted", "files_touched", "dirs_touched",
            "author_prior_commits", "hour_of_day", "day_of_week", "commit_msg_length", "is_fix_bug_revert"]
    file_hist = [c for c in new_features.columns if c.startswith(("file_prior_", "file_revert_", "file_age_", "file_authors_", "days_since_"))]
    author_file = [c for c in new_features.columns if c.startswith(("author_file_", "author_dir_", "is_author_first_", "author_days_"))]
    change_shape = ["churn_ratio", "change_entropy", "max_file_churn", "is_test_only", "test_to_code_ratio", "config_touch", "is_merge", "files_per_dir_ratio"]
    coupling = [c for c in new_features.columns if c.startswith("co_change_")]

    print(f"\nFeature groups: base={len(base)} file_hist={len(file_hist)} author_file={len(author_file)} change_shape={len(change_shape)} coupling={len(coupling)}")
    print(f"  file_hist: {file_hist}")
    print(f"  author_file: {author_file}")
    print(f"  coupling: {coupling}")

    # Cumulative LORO
    print("\n" + "─" * 60)
    print("CUMULATIVE LORO EVALUATION")
    print("─" * 60)

    groups = [
        (base, "baseline_9"),
        (base + file_hist, "+file_history"),
        (base + file_hist + author_file, "+author_file"),
        (base + file_hist + author_file + change_shape, "+change_shape"),
        (base + file_hist + author_file + change_shape + coupling, "+coupling"),
    ]

    X = enhanced[[c for g, _ in groups for c in g]].fillna(0)
    y = enhanced["risky"]
    repos = enhanced["source_repo"].values

    results = {}
    prev_roc = None
    for cols, gname in groups:
        r = loro_eval(X, y, repos, cols, gname)
        results[gname] = r
        rm, rl, rh = r["roc_auc_ci"]
        delta = f" ({rm - prev_roc:+.4f})" if prev_roc is not None else ""
        print(f"  {gname:<25} {r['n_features']:>3} feats  ROC-AUC: {rm:.4f} [{rl:.4f},{rh:.4f}]{delta}  PR-lift: {r['pr_auc_lift']:.4f}  MCC: {r['mean']['mcc']:.4f}")
        prev_roc = rm

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    print(f"{'Group':<25} {'#F':>3} {'ROC-AUC':>30} {'PR-lift':>10} {'MCC':>8} {'F1':>8}")
    print("─" * 90)
    for gname, r in results.items():
        rm, rl, rh = r["roc_auc_ci"]
        print(f"{gname:<25} {r['n_features']:>3} {rm:.4f} [{rl:.4f}, {rh:.4f}]{'':<5} {r['pr_auc_lift']:>10.4f} {r['mean']['mcc']:>8.4f} {r['mean']['f1']:>8.4f}")

    # Leakage control
    full_cols = base + file_hist + author_file + change_shape + coupling
    full_roc = results["+coupling"]["roc_auc_ci"][0]
    print(f"\n{'─'*60}")
    print("LEAKAGE CONTROL")
    print(f"{'─'*60}")
    if full_roc > 0.85:
        print(f"  ROC-AUC {full_roc:.4f} > 0.85 — running ablation...")
        for gn, gc in [("file_history", file_hist), ("author_file", author_file),
                        ("change_shape", change_shape), ("coupling", coupling)]:
            reduced = [c for c in full_cols if c not in gc]
            r = loro_eval(X, y, repos, reduced, f"no_{gn}")
            d = full_roc - r["roc_auc_ci"][0]
            tag = "LEAK?" if d >= 0.05 else "ok"
            print(f"    no_{gn}: ROC-AUC {r['roc_auc_ci'][0]:.4f} Δ={d:+.4f} [{tag}]")
    else:
        print(f"  ROC-AUC {full_roc:.4f} <= 0.85 — no alarm, all groups appear clean")

    # Per-repo
    print(f"\n{'─'*60}")
    print("PER-REPO DETAIL (full set)")
    print(f"{'─'*60}")
    fr = results["+coupling"]
    print(f"{'Repo':<15} {'ROC-AUC':>10} {'PR-AUC':>10} {'MCC':>8} {'F1':>8}")
    print("─" * 55)
    for repo in sorted(REPO_NAMES):
        m = fr["per_repo"].get(repo, {})
        if m:
            print(f"{repo:<15} {m['roc_auc']:>10.4f} {m['pr_auc']:>10.4f} {m['mcc']:>8.4f} {m['f1']:>8.4f}")
    print(f"{'MEAN':<15} {fr['mean']['roc_auc']:>10.4f} {fr['mean']['pr_auc']:>10.4f} {fr['mean']['mcc']:>8.4f} {fr['mean']['f1']:>8.4f}")

    print("\n" + "=" * 80)
    print("M.1 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
