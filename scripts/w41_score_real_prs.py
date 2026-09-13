#!/usr/bin/env python3
"""W4.1: Score real multi-commit PRs from GitHub merge commits.

Exercises U.1's PR-level aggregation: multi-commit scoring, PR-level
verdict, riskiest commit identification, and PR-level pattern detection.
"""
import subprocess, json, os, sys, time
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = Path(__file__).parent.parent
REPOS = {
    "django": str(REPO_ROOT / "repos" / "django"),
    "react": str(REPO_ROOT / "repos" / "react"),
    "kubernetes": str(REPO_ROOT / "repos" / "kubernetes"),
}

# Only use merges with 3-30 commits (skip mega-merges)
MIN_COMMITS = 3
MAX_COMMITS = 30

def find_multi_commit_merges(repo_path, limit=300, min_c=3, max_c=30):
    """Find merge commits with exactly 2 parents and between min/max commits."""
    out = subprocess.check_output(
        ["git", "-C", repo_path, "log", "--merges", f"--max-count={limit}",
         "--format=%H %P"],
        text=True, timeout=30,
    )
    merges = []
    for line in out.strip().split("\n"):
        parts = line.split()
        if len(parts) < 3:
            continue
        sha = parts[0]
        parents = parts[1:]
        if len(parents) != 2:
            continue
        # Count commits in the merge
        n = int(subprocess.check_output(
            ["git", "-C", repo_path, "rev-list", "--count", f"{parents[0]}..{sha}"],
            text=True, timeout=10,
        ).strip())
        if min_c <= n <= max_c:
            subj = subprocess.check_output(
                ["git", "-C", repo_path, "log", "--format=%s", "-1", sha],
                text=True, timeout=5,
            ).strip()
            merges.append({"sha": sha, "parent": parents[0], "n": n, "subject": subj})
    return merges


def get_commits_in_merge(repo_path, merge_sha, parent):
    """Get ordered list of commits in a merge."""
    out = subprocess.check_output(
        ["git", "-C", repo_path, "log", "--format=%H|%an|%ae|%ad|%s",
         f"{parent}..{merge_sha}"],
        text=True, timeout=30,
    )
    commits = []
    for line in out.strip().split("\n"):
        if "|" not in line:
            continue
        parts = line.split("|", 4)
        commits.append({
            "sha": parts[0],
            "author": parts[1] if len(parts) > 1 else "",
            "email": parts[2] if len(parts) > 2 else "",
            "date": parts[3] if len(parts) > 3 else "",
            "message": parts[4] if len(parts) > 4 else "",
        })
    return commits


def get_commit_files(repo_path, sha):
    """Get files changed in a commit."""
    out = subprocess.check_output(
        ["git", "-C", repo_path, "diff-tree", "--no-commit-id", "-r", "--name-only", sha],
        text=True, timeout=10,
    )
    return [f.strip() for f in out.strip().split("\n") if f.strip()]


def compute_basic_features(commit, files, repo_path):
    """Compute basic features for scoring without the full graph walk."""
    import hashlib
    n_files = len(files)
    dirs = set()
    for f in files:
        parts = f.split("/")
        if len(parts) > 1:
            dirs.add(parts[0])
    n_dirs = len(dirs) if dirs else 1

    # Basic features aligned with the model's 35 feature columns
    msg = commit.get("message", "")
    msg_len = len(msg)
    is_fix = 1 if any(kw in msg.lower() for kw in ["fix", "bug", "revert"]) else 0

    # Check test files
    test_patterns = ["test", "spec", "_test.py", "Test.java"]
    is_test_only = 1 if all(any(p in f.lower() for p in test_patterns) for f in files) else 0
    test_count = sum(1 for f in files if any(p in f.lower() for p in test_patterns))
    test_ratio = test_count / max(n_files, 1)

    # Config files
    config_patterns = [".yaml", ".yml", ".toml", ".lock", "Dockerfile", ".github/", "package.json"]
    has_config = 1 if any(any(p in f for p in config_patterns) for f in files) else 0

    # Churn ratio (simplified: estimate from file count)
    churn_ratio = 0.0

    # Build 35-element feature vector
    features = [
        10.0,  # lines_added (estimate)
        5.0,   # lines_deleted (estimate)
        float(n_files),
        float(n_dirs),
        0.0,   # author_prior_commits
        12.0,  # hour_of_day
        1,     # day_of_week
        float(msg_len),
        float(is_fix),
        # File history (unknown without graph)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # Author familiarity
        0.0, 0.0, 0.0, 0.0, 0.0,
        # Change shape
        churn_ratio, 1.5, float(n_files * 10), is_test_only, test_ratio, has_config,
        0.0,  # is_merge
        float(n_files) / max(n_dirs, 1),
    ]
    return features


def score_commit(commit, files, repo_path, repo_name):
    """Score a single commit."""
    from ml.scoring import evaluate_commit_full

    features = compute_basic_features(commit, files, repo_path)
    result = evaluate_commit_full(
        feature_values=features,
        repo_name=repo_name,
        commit_hash=commit["sha"],
        author=commit.get("author", ""),
        message=commit.get("message", ""),
        files=files,
        lines_added=10,
        lines_deleted=5,
        files_touched=len(files),
        is_merge=False,
        hour_of_day=12,
        day_of_week=1,
        author_prior_commits=0,
        file_revert_count_max=0,
        file_prior_changes_max=0,
    )
    return result


def detect_pr_patterns(commits, all_files, all_file_counts):
    """Detect PR-level patterns."""
    patterns = []

    # file_churn: files modified in 3+ commits
    churn_files = [f for f, c in all_file_counts.items() if c >= 3]
    if churn_files:
        patterns.append({
            "pattern": "file_churn",
            "message": f"{len(churn_files)} file(s) modified in 3+ commits",
            "files": churn_files[:5],
        })

    # large_pr: many files changed
    unique_files = len(set(all_files))
    if unique_files >= 20:
        patterns.append({
            "pattern": "large_pr",
            "message": f"{unique_files} unique files changed across PR",
        })

    # revert chain: any commit message contains "revert"
    reverts = [c for c in commits if "revert" in c.get("message", "").lower()]
    if len(reverts) >= 2:
        patterns.append({
            "pattern": "revert_chain",
            "message": f"{len(reverts)} revert commits in PR",
        })

    # test_churn: test files added then removed (or vice versa)
    test_files = set()
    for c in commits:
        msg = c.get("message", "").lower()
        if "test" in msg or "spec" in msg:
            test_files.update(c.get("_files", []))
    # Simplified: just check if test files are in the churn
    test_churn = [f for f in churn_files if "test" in f.lower() or "spec" in f.lower()]
    if test_churn:
        patterns.append({
            "pattern": "test_churn",
            "message": f"Test files modified across multiple commits",
            "files": test_churn[:3],
        })

    return patterns


def main():
    all_results = []
    timing = {"3": [], "10": [], "20": []}
    pattern_counter = Counter()

    for repo_name, repo_path in REPOS.items():
        if not os.path.exists(repo_path):
            print(f"SKIP: {repo_name} (repo not found)")
            continue

        print(f"\n{'='*60}")
        print(f"  REPO: {repo_name}")
        print(f"{'='*60}")

        merges = find_multi_commit_merges(repo_path, limit=500)
        print(f"  Found {len(merges)} multi-commit merges (3-{MAX_COMMITS} commits)")

        scored = 0
        for m in merges:
            if scored >= 10:
                break

            # Get commits in this merge
            commits = get_commits_in_merge(repo_path, m["sha"], m["parent"])
            if len(commits) < MIN_COMMITS:
                continue

            # Score each commit and time it
            start = time.time()
            commit_results = []
            all_files = []
            all_file_counts = Counter()

            for c in commits:
                files = get_commit_files(repo_path, c["sha"])
                c["_files"] = files
                all_files.extend(files)
                for f in files:
                    all_file_counts[f] += 1

                try:
                    result = score_commit(c, files, repo_path, repo_name)
                    commit_results.append({
                        "sha": c["sha"][:12],
                        "author": c.get("author", ""),
                        "message": c.get("message", "")[:80],
                        "score": round(result.risk_score, 4),
                        "band": result.band,
                        "files_count": len(files),
                        "rule_count": len(result.rule_results),
                        "blocked": result.blocked,
                    })
                except Exception as e:
                    print(f"    FAILED {c['sha'][:8]}: {e}")

            elapsed = time.time() - start

            if not commit_results:
                continue

            # PR aggregation
            scores = [c["score"] for c in commit_results]
            bands = [c["band"] for c in commit_results]
            max_score = max(scores)
            mean_score = sum(scores) / len(scores)

            if "high" in bands:
                verdict = "high"
            elif "medium" in bands:
                verdict = "medium"
            else:
                verdict = "low"

            riskiest = max(commit_results, key=lambda c: c["score"])

            # Detect patterns
            patterns = detect_pr_patterns(commits, all_files, all_file_counts)

            pr = {
                "repo": repo_name,
                "merge_sha": m["sha"][:12],
                "subject": m["subject"][:80],
                "commit_count": len(commit_results),
                "verdict": verdict,
                "max_score": round(max_score, 4),
                "mean_score": round(mean_score, 4),
                "riskiest_commit": riskiest["sha"],
                "total_files": len(set(all_files)),
                "patterns": patterns,
                "commits": commit_results,
                "scoring_time_s": round(elapsed, 2),
            }

            for p in patterns:
                pattern_counter[p["pattern"]] += 1

            # Track timing by commit count
            n = len(commit_results)
            bucket = "3" if n <= 4 else "10" if n <= 12 else "20"
            timing[bucket].append(elapsed)

            all_results.append(pr)
            scored += 1

            print(f"  PR {m['sha'][:12]}: {len(commit_results)} commits, verdict={verdict}, "
                  f"max={max_score:.3f}, files={len(set(all_files))}, "
                  f"patterns={[p['pattern'] for p in patterns]}, time={elapsed:.1f}s")
            for cr in commit_results:
                print(f"    {cr['sha']}: score={cr['score']:.3f} band={cr['band']} "
                      f"files={cr['files_count']} rules={cr['rule_count']}")

    # Global summary
    print(f"\n{'='*60}")
    print(f"  GLOBAL SUMMARY: {len(all_results)} PRs scored")
    print(f"{'='*60}")

    # Pattern fire rates
    print(f"\n  Pattern fire rates:")
    for pat, count in pattern_counter.most_common():
        print(f"    {pat}: {count}/{len(all_results)} ({count*100/len(all_results):.0f}%)")
    if not pattern_counter:
        print("    (none fired across any PR)")

    # Band distribution
    band_dist = Counter(pr["verdict"] for pr in all_results)
    print(f"\n  Verdict distribution: {dict(band_dist)}")

    # Timing
    print(f"\n  Scoring timing:")
    for bucket, times in timing.items():
        if times:
            print(f"    {bucket}-commit PRs: p50={sorted(times)[len(times)//2]:.1f}s, "
                  f"mean={sum(times)/len(times):.1f}s, n={len(times)}")

    # Save results
    out_path = str(REPO_ROOT / "data" / "w41_real_prs.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    main()
