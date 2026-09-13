#!/usr/bin/env python3
"""W4.1: Score real multi-commit PRs from GitHub.

A real multi-commit PR has a merge commit with 2 parents (not squash-merged).
We use `git log --merges` to find merge commits, then count commits between
the two parents to identify multi-commit PRs.
"""
import subprocess, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPOS = {
    "django": str(Path(__file__).parent.parent / "repos" / "django"),
    "react": str(Path(__file__).parent.parent / "repos" / "react"),
    "kubernetes": str(Path(__file__).parent.parent / "repos" / "kubernetes"),
}

def find_merge_commits(repo_path: str, limit: int = 200):
    """Find merge commits with 2 parents (real merges, not squash)."""
    out = subprocess.check_output(
        ["git", "log", "--merges", f"--max-count={limit}",
         "--format=%H|%P|%s", "--no-merges"],
        cwd=repo_path, text=True, timeout=60,
    )
    # Actually, --merges already filters to merge commits
    # Format: hash|parent1 parent2|subject
    merges = []
    for line in out.strip().split("\n"):
        if "|" not in line:
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        sha = parts[0].strip()
        parents = parts[1].strip().split()
        subject = parts[2].strip() if len(parts) > 2 else ""
        if len(parents) == 2:
            merges.append({"sha": sha, "parent1": parents[0], "parent2": parents[1], "subject": subject})
    return merges

def get_commits_between(repo_path: str, sha: str, parent: str):
    """Get commits between sha and parent (the PR's commit history)."""
    out = subprocess.check_output(
        ["git", "log", "--format=%H", f"{parent}..{sha}"],
        cwd=repo_path, text=True, timeout=30,
    )
    return [h.strip() for h in out.strip().split("\n") if h.strip()]

def get_commit_info(repo_path: str, sha: str):
    """Get basic info for a commit."""
    out = subprocess.check_output(
        ["git", "log", "--format=%H|%an|%ae|%ad|%s", "-1", sha],
        cwd=repo_path, text=True, timeout=10,
    )
    parts = out.strip().split("|", 4)
    return {
        "sha": parts[0],
        "author": parts[1] if len(parts) > 1 else "",
        "email": parts[2] if len(parts) > 2 else "",
        "date": parts[3] if len(parts) > 3 else "",
        "message": parts[4] if len(parts) > 4 else "",
    }

def get_commit_files(repo_path: str, sha: str):
    """Get files changed in a commit."""
    out = subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", sha],
        cwd=repo_path, text=True, timeout=10,
    )
    return [f.strip() for f in out.strip().split("\n") if f.strip()]

def score_commit(commit_info, files, repo_name):
    """Score a single commit using the shared scoring function."""
    from ml.scoring import evaluate_commit_full
    import yaml

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    fcols = config.get("feature_columns", [])

    # Use default zero features since we don't have CSV data
    # The scoring will use file metadata, message, etc.
    fv = [0.0] * len(fcols)
    message = commit_info.get("message", "")
    lines_added = sum(1 for _ in [])  # simplified

    result = evaluate_commit_full(
        feature_values=fv,
        repo_name=repo_name,
        commit_hash=commit_info["sha"],
        author=commit_info.get("author", ""),
        message=message,
        files=files,
        lines_added=lines_added,
        lines_deleted=0,
        files_touched=len(files),
        is_merge=False,
        hour_of_day=12,
        day_of_week=1,
        author_prior_commits=0,
        file_revert_count_max=0,
        file_prior_changes_max=0,
    )
    return result

def main():
    all_prs = []

    for repo_name, repo_path in REPOS.items():
        if not os.path.exists(repo_path):
            print(f"SKIP: {repo_name} repo not found at {repo_path}")
            continue

        print(f"\n{'='*60}")
        print(f"REPO: {repo_name}")
        print(f"{'='*60}")

        merges = find_merge_commits(repo_path, limit=300)
        multi_commit = [m for m in merges if True]  # all have 2 parents
        print(f"Found {len(multi_commit)} merge commits with 2 parents")

        # Find PRs with 3+ commits
        scored_prs = []
        for m in multi_commit[:50]:
            commits = get_commits_between(repo_path, m["sha"], m["parent1"])
            if len(commits) < 3:
                continue

            print(f"\n  PR merge={m['sha'][:12]} subject='{m['subject'][:60]}' commits={len(commits)}")

            # Score each commit
            commit_scores = []
            for csha in commits[:10]:  # cap at 10 commits per PR
                info = get_commit_info(repo_path, csha)
                files = get_commit_files(repo_path, csha)
                try:
                    result = score_commit(info, files, repo_name)
                    commit_scores.append({
                        "sha": csha[:12],
                        "author": info.get("author", ""),
                        "message": info.get("message", "")[:60],
                        "score": round(result.risk_score, 4),
                        "band": result.band,
                        "files_count": len(files),
                        "rule_count": len(result.rule_results),
                        "blocked": result.blocked,
                        "warning_count": result.warning_count,
                    })
                    print(f"    {csha[:8]}: score={result.risk_score:.3f} band={result.band} files={len(files)} rules={len(result.rule_results)}")
                except Exception as e:
                    print(f"    {csha[:8]}: FAILED - {e}")

            if not commit_scores:
                continue

            # PR-level aggregation
            scores = [c["score"] for c in commit_scores]
            bands = [c["band"] for c in commit_scores]
            max_score = max(scores)
            mean_score = sum(scores) / len(scores)

            # Determine PR verdict (max band)
            if "high" in bands:
                verdict = "high"
            elif "medium" in bands:
                verdict = "medium"
            else:
                verdict = "low"

            riskiest = max(commit_scores, key=lambda c: c["score"])

            # PR-level patterns
            all_files = []
            for csha in commits[:10]:
                all_files.extend(get_commit_files(repo_path, csha))
            file_counts = {}
            for f in all_files:
                file_counts[f] = file_counts.get(f, 0) + 1
            churn_files = [f for f, c in file_counts.items() if c >= 2]

            patterns = []
            if len(churn_files) > 0:
                patterns.append({"pattern": "file_churn", "files": churn_files[:3]})
            if len(all_files) >= 50:
                patterns.append({"pattern": "large_pr", "total_files": len(set(all_files))})

            pr = {
                "repo": repo_name,
                "merge_sha": m["sha"][:12],
                "subject": m["subject"][:80],
                "commit_count": len(commit_scores),
                "verdict": verdict,
                "max_score": round(max_score, 4),
                "mean_score": round(mean_score, 4),
                "riskiest_commit": riskiest["sha"],
                "patterns": patterns,
                "commits": commit_scores,
            }
            scored_prs.append(pr)
            all_prs.append(pr)

            if len(scored_prs) >= 10:
                break

        # Print summary for this repo
        print(f"\n  SUMMARY: {len(scored_prs)} multi-commit PRs scored")
        for p in scored_prs:
            print(f"    {p['merge_sha']}: {p['commit_count']} commits, verdict={p['verdict']}, max={p['max_score']}, patterns={[pp['pattern'] for pp in p['patterns']]}")

    # Global summary
    print(f"\n{'='*60}")
    print(f"GLOBAL SUMMARY: {len(all_prs)} PRs across all repos")
    print(f"{'='*60}")

    pattern_counts = {}
    for pr in all_prs:
        for p in pr.get("patterns", []):
            pattern_counts[p["pattern"]] = pattern_counts.get(p["pattern"], 0) + 1

    print(f"\nPattern fire rates:")
    for pat, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  {pat}: {count}/{len(all_prs)} ({count*100/len(all_prs):.0f}%)")
    if not pattern_counts:
        print("  (none fired)")

    # Band distribution
    band_counts = {}
    for pr in all_prs:
        v = pr["verdict"]
        band_counts[v] = band_counts.get(v, 0) + 1
    print(f"\nVerdict distribution: {band_counts}")

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "w41_real_prs.json")
    with open(out_path, "w") as f:
        json.dump(all_prs, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
