#!/usr/bin/env python3
"""W7.2: Production precision by band on frozen OOW eval set.
Uses batch_score_commits for features (single graph walk).
"""
import json, sys, os, time
import numpy as np
from pathlib import Path
from datetime import datetime

GK = Path(__file__).resolve().parent.parent
os.chdir(GK)
sys.path.insert(0, ".")

import ml.single_commit_features as scf
scf.WINDOW_START = "2025-01-01"
scf.WINDOW_END = "2026-09-01"
scf.FORWARD_LOOK_END = "2026-09-01"
scf._graph_cache = {}

from ml.scoring import _load_model, _determine_band, evaluate_commit_full, _load_config
from ml.single_commit_features import _get_full_graph
from sklearn.metrics import roc_auc_score

REPO_ROOT = GK
REPOS = {
    "django": str(REPO_ROOT / "repos" / "django"),
    "react": str(REPO_ROOT / "repos" / "react"),
    "kafka": str(REPO_ROOT / "repos" / "kafka"),
    "kubernetes": str(REPO_ROOT / "repos" / "kubernetes"),
    "rust": str(REPO_ROOT / "repos" / "rust"),
}


def wilson_ci(successes, total, z=1.96):
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def compute_outcomes_fast(repo_path, oow_shas, forward_days=7):
    """Compute realized outcomes with file-indexed forward look.
    
    Graph format: sorted_graph = [(sha, {"date": datetime, "files": [str], ...})]
    """
    print(f"  Building graph...", file=sys.stderr, flush=True)
    t0 = time.time()
    graph_dict, risky_hashes, sorted_graph = _get_full_graph(repo_path)
    print(f"  Graph: {len(graph_dict)} commits in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    sha_set = set(graph_dict.keys())
    sorted_idx = {sha: i for i, (sha, _) in enumerate(sorted_graph)}

    # Pre-build: for each file, list of (commit_idx, date)
    file_commit_map = {}
    for i, (sha, node) in enumerate(sorted_graph):
        files = node.get("files", [])
        if isinstance(files, str):
            files = files.split(",")
        cd = node.get("date")
        for fn in files:
            if fn not in file_commit_map:
                file_commit_map[fn] = []
            file_commit_map[fn].append((i, cd))

    print(f"  File index: {len(file_commit_map)} files in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    outcomes = {}
    oow_set = [s for s in oow_shas if s in sha_set]
    print(f"  {len(oow_set)} / {len(oow_shas)} OOW commits in graph", file=sys.stderr, flush=True)

    for k, sha in enumerate(oow_set):
        idx = sorted_idx[sha]
        _, commit_node = sorted_graph[idx]
        cd = commit_node.get("date")

        # Get files changed
        files = commit_node.get("files", [])
        if isinstance(files, str):
            files = files.split(",")
        if not files:
            outcomes[sha] = 0
            continue

        # Forward look: only check future commits that touch the SAME files
        found = False
        for fn in files:
            if fn in file_commit_map:
                for fidx, fdate in file_commit_map[fn]:
                    if fidx <= idx:
                        continue
                    days = (fdate - cd).total_seconds() / 86400
                    if days > forward_days:
                        break
                    if days >= 0:
                        found = True
                        break
            if found:
                break

        if found:
            outcomes[sha] = 1
        else:
            msg = (commit_node.get("subject", "") or "").lower()
            outcomes[sha] = 1 if "revert" in msg[:80] else 0

        if (k + 1) % 200 == 0:
            print(f"  Outcomes: {k+1}/{len(oow_set)} in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    print(f"  Outcomes done: {sum(outcomes.values())}/{len(outcomes)} positive in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    return outcomes


def score_repo(repo_name, shas):
    """Score a list of OOW commits for one repo."""
    repo_path = REPOS[repo_name]
    t0 = time.time()

    # 1. Compute outcomes
    outcomes = compute_outcomes_fast(repo_path, shas)
    t1 = time.time()
    print(f"  Outcomes computed in {t1-t0:.1f}s", file=sys.stderr, flush=True)

    # 2. Batch extract features (single graph walk)
    oow_shas_list = list(outcomes.keys())
    print(f"  Batch extracting features for {len(oow_shas_list)} commits...", file=sys.stderr, flush=True)
    features_list = scf.batch_score_commits(repo_path, oow_shas_list)
    t2 = time.time()
    print(f"  Features extracted in {t2-t1:.1f}s", file=sys.stderr, flush=True)

    # 3. Score each commit
    scored = []
    _, feature_columns = _load_config()

    for sha, feat_dict in zip(oow_shas_list, features_list):
        if not feat_dict:
            continue
        try:
            feat_keys = [k for k in feature_columns if k in feat_dict]
            feature_values = [float(feat_dict.get(k, 0)) for k in feat_keys]
            while len(feature_values) < len(feature_columns):
                feature_values.append(0.0)

            # Get files
            files_val = feat_dict.get("touched_files", [])
            if isinstance(files_val, str):
                files_val = files_val.split(",") if files_val else []

            result = evaluate_commit_full(
                feature_values=feature_values[:len(feature_columns)],
                repo_name=repo_name,
                commit_hash=sha,
                author=feat_dict.get("_graph_author", ""),
                message=feat_dict.get("commit_message", feat_dict.get("subject", "")),
                files=files_val,
                lines_added=int(feat_dict.get("lines_added", 0)),
                lines_deleted=int(feat_dict.get("lines_deleted", 0)),
                files_touched=int(feat_dict.get("touched_files_count", feat_dict.get("files_touched_count", 0))),
                dirs_touched=int(feat_dict.get("dirs_touched", 0)),
                is_merge=bool(feat_dict.get("is_merge", 0)),
                author_prior_commits=int(feat_dict.get("_author_prior_commits", feat_dict.get("author_prior_commits", 0))),
            )
            scored.append({
                "sha": sha,
                "score": result.risk_score,
                "band": result.band,
                "label": outcomes[sha],
            })
        except Exception as e:
            print(f"  ERROR scoring {sha}: {e}", file=sys.stderr, flush=True)

    print(f"  Scored {len(scored)} commits in {time.time()-t2:.1f}s", file=sys.stderr, flush=True)
    return scored


def compute_metrics(scored):
    """Compute band-level metrics and ROC-AUC from scored list."""
    if not scored:
        return None
    scores_arr = np.array([s["score"] for s in scored])
    labels_arr = np.array([s["label"] for s in scored])
    n = len(scored)
    base_rate = float(labels_arr.mean())

    band_metrics = {}
    for band in ["high", "medium", "low"]:
        band_rows = [s for s in scored if s["band"] == band]
        nb = len(band_rows)
        if nb == 0:
            band_metrics[band] = {"n": 0, "risky_rate": 0, "precision": 0, "ci": (0, 0), "lift": 0}
            continue
        n_risky = sum(s["label"] for s in band_rows)
        risky_rate = n_risky / nb
        ci = wilson_ci(n_risky, nb)
        lift = risky_rate / base_rate if base_rate > 0 else 0
        band_metrics[band] = {
            "n": nb, "risky_rate": round(risky_rate, 4),
            "precision": round(risky_rate, 4),
            "ci": (round(ci[0], 4), round(ci[1], 4)),
            "lift": round(lift, 2)
        }

    # Bootstrap ROC-AUC
    auc, auc_ci = None, None
    if labels_arr.sum() > 0 and labels_arr.sum() < n:
        boot_aucs = []
        rng = np.random.RandomState(42)
        for _ in range(1000):
            idx = rng.choice(n, n, replace=True)
            bs, bl = scores_arr[idx], labels_arr[idx]
            if bl.sum() == 0 or bl.sum() == len(bl):
                continue
            boot_aucs.append(roc_auc_score(bl, bs))
        if boot_aucs:
            auc = float(np.mean(boot_aucs))
            auc_ci = (float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5)))

    return {
        "n": n, "base_rate": round(base_rate, 4),
        "bands": band_metrics,
        "roc_auc": round(auc, 4) if auc else None,
        "roc_auc_ci": (round(auc_ci[0], 4), round(auc_ci[1], 4)) if auc_ci else None,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", help="Run only this repo")
    args = parser.parse_args()

    with open(GK / "data" / "oow_eval_set.json") as f:
        eval_set = json.load(f)
    eval_repos = eval_set.get("repos", eval_set)

    results = {}
    all_scored = []

    for repo_name, shas_raw in eval_repos.items():
        shas = [s["hash"] if isinstance(s, dict) else s for s in shas_raw]
        if args.repo and repo_name != args.repo:
            continue
        if len(shas) == 0:
            continue

        print(f"\n{'='*50}", file=sys.stderr, flush=True)
        print(f"  REPO: {repo_name} ({len(shas)} OOW commits)", file=sys.stderr, flush=True)
        print(f"{'='*50}", file=sys.stderr, flush=True)

        scored = score_repo(repo_name, shas)
        if scored:
            metrics = compute_metrics(scored)
            results[repo_name] = metrics
            all_scored.extend(scored)

            with open(GK / "data" / f"w72_{repo_name}.json", "w") as f:
                json.dump({"metrics": metrics, "scored_count": len(scored)}, f, indent=2)

    # Print summary
    print("\n\nW7.2 PRODUCTION PRECISION BY BAND (Frozen OOW Eval Set, Correct Thresholds)")
    print("=" * 95)
    print(f"{'Repo':<14} {'N':>5} {'Base':>6} {'High n':>7} {'High prec':>10} {'95% CI':>14} {'Lift':>6}  {'AUC':>6} {'95% CI':>16}")
    print("-" * 95)
    for repo, r in results.items():
        if not r:
            continue
        h = r["bands"]["high"]
        auc_str = f"{r['roc_auc']:.4f}" if r['roc_auc'] else "N/A"
        ci_str = f"[{r['roc_auc_ci'][0]:.4f}, {r['roc_auc_ci'][1]:.4f}]" if r['roc_auc_ci'] else ""
        print(f"{repo:<14} {r['n']:>5} {r['base_rate']:>5.1%} {h['n']:>7} "
              f"{h['precision']:>9.1%} [{h['ci'][0]:.1%}, {h['ci'][1]:.1%}] {h['lift']:>5.1f}x  {auc_str:>6} {ci_str:>16}")
    print("-" * 95)

    print(f"\n{'Repo':<14} {'Med n':>7} {'Med prec':>10} {'95% CI':>14} {'Lift':>6}  {'Low n':>7} {'Low prec':>10} {'95% CI':>14} {'Lift':>6}")
    print("-" * 95)
    for repo, r in results.items():
        if not r:
            continue
        m = r["bands"]["medium"]
        l = r["bands"]["low"]
        print(f"{repo:<14} {m['n']:>7} {m['precision']:>9.1%} [{m['ci'][0]:.1%}, {m['ci'][1]:.1%}] {m['lift']:>5.1f}x  "
              f"{l['n']:>7} {l['precision']:>9.1%} [{l['ci'][0]:.1%}, {l['ci'][1]:.1%}] {l['lift']:>5.1f}x")
    print()

    # Pooled
    if all_scored:
        pooled_labels = np.array([s["label"] for s in all_scored])
        pooled_base = pooled_labels.mean()
        print(f"POOLED: n={len(all_scored)}, base_rate={pooled_base:.1%}")
        for band in ["high", "medium", "low"]:
            br = [s for s in all_scored if s["band"] == band]
            nb = len(br)
            if nb > 0:
                nr = sum(s["label"] for s in br)
                prec = nr / nb
                ci = wilson_ci(nr, nb)
                lift = prec / pooled_base if pooled_base > 0 else 0
                print(f"  {band:>8}: n={nb:>5}, prec={prec:.1%} [{ci[0]:.1%}, {ci[1]:.1%}], lift={lift:.1f}x")
        print()

    with open(GK / "data" / "w72_production_precision.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved to data/w72_production_precision.json")


if __name__ == "__main__":
    main()
