#!/usr/bin/env python3
"""W1.3: Score real multi-commit PRs from git merge history."""
import json, subprocess, sys, os
from pathlib import Path
from datetime import datetime, timezone

os.chdir(Path(__file__).parent.parent)
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.scoring import evaluate_commit_full

REPO_MAP = {
    "django": Path("repos/django"),
    "react": Path("repos/react"),
    "kubernetes": Path("repos/kubernetes"),
}


def find_merge_prs(repo_path, n=10):
    """Find merge commits that are PR merges (have >1 parent) with 3+ commits."""
    try:
        # Get merge commits from last 6 months
        out = subprocess.check_output(
            ["git", "log", "--merges", "--since=2026-01-01", f"-{n*5}",
             "--format=%H|%P|%ct|%an|%s"],
            cwd=repo_path, text=True, timeout=30
        )
    except Exception as e:
        print(f"  git log failed: {e}")
        return []

    prs = []
    for line in out.strip().split("\n"):
        if not line or "|" not in line:
            continue
        parts = line.split("|", 4)
        if len(parts) < 5:
            continue
        merge_hash = parts[0]
        parents = parts[1].split()
        ts = int(parts[2])
        author = parts[3]
        message = parts[4]

        if len(parents) < 2:
            continue

        # Get commits in this merge (second parent is the branch)
        feature_parent = parents[1]
        try:
            commits_out = subprocess.check_output(
                ["git", "log", "--oneline", f"{feature_parent}..{parents[0]}"],
                cwd=repo_path, text=True, timeout=10
            )
            commit_lines = [l.strip() for l in commits_out.strip().split("\n") if l.strip()]
        except Exception:
            commit_lines = []

        if len(commit_lines) >= 3:
            prs.append({
                "merge_hash": merge_hash,
                "title": message[:120],
                "author": author,
                "timestamp": ts,
                "commit_count": len(commit_lines),
                "commit_hashes": [c.split()[0] for c in commit_lines[:20]],
            })

    return prs[:n]


def score_commit(repo_name, repo_path, commit_hash):
    """Score a single commit using compute_single_commit_m1_features + shared scoring."""
    try:
        from ml.single_commit_features import compute_single_commit_m1_features

        # Get commit metadata
        out = subprocess.check_output(
            ["git", "show", "--format=%H|%ct|%an|%aE|%s", "--name-only", "--no-patch", commit_hash],
            cwd=repo_path, text=True, timeout=10
        )
        parts = out.strip().split("|", 4)
        if len(parts) < 4:
            return None

        ts = int(parts[1])
        author = parts[2]
        message = parts[4] if len(parts) > 4 else ""
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)

        # Get files
        files_out = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash],
            cwd=repo_path, text=True, timeout=10
        )
        files = set(f.strip() for f in files_out.strip().split("\n") if f.strip())

        # Get numstat
        numstat = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--numstat", "-r", commit_hash],
            cwd=repo_path, text=True, timeout=10
        )
        lines_added = lines_deleted = 0
        for nl in numstat.strip().split("\n"):
            ns = nl.split("\t")
            if len(ns) >= 2:
                try:
                    lines_added += int(ns[0])
                    lines_deleted += int(ns[1])
                except ValueError:
                    pass

        # Get diff for content rules
        try:
            diff_text = subprocess.check_output(
                ["git", "show", "--format=", "--no-patch", "-p", commit_hash],
                cwd=repo_path, text=True, timeout=10
            )
        except Exception:
            diff_text = ""

        # Compute actual features using the shared extraction path
        import yaml
        with open("ml/config.yaml") as f:
            config = yaml.safe_load(f)
        feature_cols = config.get("feature_columns", [])

        m1_result = compute_single_commit_m1_features(
            repo_path=str(repo_path),
            commit_hash=commit_hash,
            commit_date=dt,
            author_name=author,
            touched_files=files,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
        )

        feature_values = [float(m1_result.get(fc, 0.0) or 0.0) for fc in feature_cols]

        result = evaluate_commit_full(
            feature_values=feature_values,
            repo_name=repo_name,
            commit_hash=commit_hash,
            author=author,
            message=message,
            files=list(files),
            lines_added=lines_added,
            lines_deleted=lines_deleted,
            diff_text=diff_text,
            repo_path=str(repo_path),
        )

        return {
            "hash": commit_hash[:12],
            "author": author,
            "message": message[:80],
            "score": result.risk_score,
            "band": result.band,
            "files_count": len(files),
            "lines_added": lines_added,
            "lines_deleted": lines_deleted,
            "rule_results": [
                {"rule": r.rule_name, "severity": r.severity.value, "passed": r.passed, "message": r.message}
                for r in result.rule_results if not r.passed
            ],
            "blocked": result.blocked,
            "shap_top3": result.shap_top3,
        }
    except Exception as e:
        print(f"    Error scoring {commit_hash[:8]}: {e}")
        return None


def main():
    all_prs = []

    for repo_name, repo_path in REPO_MAP.items():
        if not repo_path.exists():
            print(f"Skipping {repo_name} — repo not found")
            continue

        print(f"\n=== {repo_name} ===")
        prs = find_merge_prs(repo_path, n=5)
        print(f"  Found {len(prs)} merge PRs with 3+ commits")

        for pr in prs:
            print(f"\n  PR: {pr['title'][:60]}... ({pr['commit_count']} commits)")
            scored_commits = []
            for ch in pr["commit_hashes"][:10]:  # limit to 10 commits per PR
                sc = score_commit(repo_name, repo_path, ch)
                if sc:
                    scored_commits.append(sc)

            if not scored_commits:
                continue

            # Aggregate to PR level
            scores = [c["score"] for c in scored_commits]
            bands = [c["band"] for c in scored_commits]
            band_counts = {"low": bands.count("low"), "medium": bands.count("medium"), "high": bands.count("high")}
            max_band = "high" if band_counts["high"] > 0 else "medium" if band_counts["medium"] > 0 else "low"
            riskiest = max(scored_commits, key=lambda c: c["score"])
            any_blocked = any(c["blocked"] for c in scored_commits)

            # Detect PR-level patterns
            all_files = []
            for c in scored_commits:
                all_files.extend(c.get("files", []))
            file_freq = {}
            for f in all_files:
                file_freq[f] = file_freq.get(f, 0) + 1
            churn_files = [f for f, cnt in file_freq.items() if cnt >= 3]

            pr_result = {
                "repo": repo_name,
                "title": pr["title"][:80],
                "merge_hash": pr["merge_hash"][:12],
                "commit_count": len(scored_commits),
                "verdict": max_band,
                "mean_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "min_score": min(scores),
                "band_counts": band_counts,
                "riskiest_commit": riskiest["hash"],
                "blocked": any_blocked,
                "churn_files": churn_files,
                "commits": scored_commits,
            }
            all_prs.append(pr_result)

            # Print summary
            print(f"    Verdict: {max_band} | Mean: {sum(scores)/len(scores):.3f} | "
                  f"Bands: {band_counts} | Riskiest: {riskiest['hash']} ({riskiest['score']:.3f})")
            if churn_files:
                print(f"    Churn files (3+ commits): {churn_files}")
            fired_rules = []
            for c in scored_commits:
                for r in c["rule_results"]:
                    fired_rules.append(r["rule"])
            if fired_rules:
                print(f"    Rules fired: {dict((r, fired_rules.count(r)) for r in set(fired_rules))}")

    # Write results
    out = Path("data/w13_real_prs.json")
    with open(out, "w") as f:
        json.dump(all_prs, f, indent=2)
    print(f"\nWrote {len(all_prs)} PRs to {out}")

    # Summary
    print("\n--- Summary ---")
    for pr in all_prs:
        print(f"  [{pr['repo']}] {pr['title'][:50]}... → {pr['verdict']} "
              f"({pr['commit_count']} commits, mean={pr['mean_score']:.3f})")


if __name__ == "__main__":
    main()
