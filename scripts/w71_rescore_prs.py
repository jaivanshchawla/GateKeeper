#!/usr/bin/env python3
"""W7.1c: Re-score 30 real multi-commit PRs with corrected thresholds.

Uses extract_single_commit → compute_single_commit_m1_features → m1_shared
for proper feature extraction. Runs 10 PRs from django, react, kubernetes.
"""
import subprocess, json, os, sys, time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ROOT = Path(__file__).parent.parent
REPOS = {
    "django": str(REPO_ROOT / "repos" / "django"),
    "react": str(REPO_ROOT / "repos" / "react"),
    "kubernetes": str(REPO_ROOT / "repos" / "kubernetes"),
}
COMMIT_WINDOW = ("2024-07-01", "2026-07-07")

def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", repo, *args],
        text=True, timeout=30, encoding="utf-8", errors="replace",
    ).strip()

def find_multi_merges(repo_path, limit=500):
    out = git(repo_path, "log", "--merges", f"--max-count={limit}", "--format=%H %P")
    merges = []
    for line in out.split("\n"):
        parts = line.split()
        if len(parts) < 3:
            continue
        sha = parts[0]
        parents = parts[1:]
        if len(parents) != 2:
            continue
        n = int(git(repo_path, "rev-list", "--count", f"{parents[0]}..{sha}"))
        if 3 <= n <= 30:
            subj = git(repo_path, "log", "--format=%s", "-1", sha)
            merges.append({"sha": sha, "parent": parents[0], "n": n, "subject": subj})
    return merges

def main():
    from ml.extract_features import CommitFeatureExtractor
    from ml.scoring import evaluate_commit_full, reset_cache
    import yaml
    
    reset_cache()
    config = yaml.safe_load(open(REPO_ROOT / "ml" / "config.yaml"))
    feature_columns = config["feature_columns"]
    
    all_results = []
    timing = {"3": [], "10": [], "20": []}
    pattern_counter = Counter()
    
    for repo_name, repo_path in REPOS.items():
        if not os.path.exists(repo_path):
            print(f"SKIP: {repo_name}")
            continue
        
        print(f"\n{'='*60}")
        print(f"  REPO: {repo_name}")
        print(f"{'='*60}")
        
        merges = find_multi_merges(repo_path, limit=500)
        print(f"  Found {len(merges)} multi-commit merges")
        
        extractor = CommitFeatureExtractor(
            repo_path=repo_path,
            since=COMMIT_WINDOW[0],
            label_window_days=7,
        )
        
        scored = 0
        for m in merges:
            if scored >= 10:
                break
            
            commit_shas = git(repo_path, "log", "--format=%H", f"{m['parent']}..{m['sha']}").split("\n")
            commit_shas = [s for s in commit_shas if s]
            if len(commit_shas) < 3:
                continue
            
            start = time.time()
            commit_results = []
            all_files = []
            all_file_counts = Counter()
            
            for sha in commit_shas:
                try:
                    features = extractor.extract_single_commit(repo_path, sha)
                    
                    # Build feature vector
                    feat_values = []
                    for col in feature_columns:
                        val = features.get(col, 0)
                        if val is None:
                            val = 0
                        feat_values.append(float(val))
                    
                    files_str = features.get("touched_files", "")
                    files = [f.strip() for f in files_str.split("|") if f.strip()]
                    all_files.extend(files)
                    for f in files:
                        all_file_counts[f] += 1
                    
                    result = evaluate_commit_full(
                        feature_values=feat_values,
                        repo_name=repo_name,
                        commit_hash=sha,
                        author=features.get("author", ""),
                        message=features.get("commit_message", "")[:80],
                        files=files,
                        lines_added=int(features.get("lines_added", 0)),
                        lines_deleted=int(features.get("lines_deleted", 0)),
                        files_touched=int(features.get("files_touched", len(files))),
                        is_merge=bool(features.get("is_merge", 0)),
                        hour_of_day=int(features.get("hour_of_day", 12)),
                        day_of_week=int(features.get("day_of_week", 0)),
                        author_prior_commits=int(features.get("author_prior_commits", 0)),
                        file_revert_count_max=int(features.get("file_revert_count_max", 0)),
                        file_prior_changes_max=int(features.get("file_prior_changes_max", 0)),
                    )
                    commit_results.append({
                        "sha": sha[:12],
                        "author": features.get("author", ""),
                        "message": features.get("commit_message", "")[:80],
                        "score": round(result.risk_score, 4),
                        "band": result.band,
                        "files_count": len(files),
                        "rule_count": len(result.rule_results),
                        "blocked": result.blocked,
                        "rules": [{"name": r.rule_name, "severity": r.severity.value,
                                   "passed": r.passed, "message": r.message}
                                  for r in result.rule_results],
                    })
                except Exception as e:
                    print(f"    FAILED {sha[:8]}: {e}")
            
            elapsed = time.time() - start
            if not commit_results:
                continue
            
            scores = [c["score"] for c in commit_results]
            bands = [c["band"] for c in commit_results]
            verdict = "high" if "high" in bands else "medium" if "medium" in bands else "low"
            riskiest = max(commit_results, key=lambda c: c["score"])
            
            # Patterns
            patterns = []
            churn_files = [f for f, c in all_file_counts.items() if c >= 3]
            if churn_files:
                patterns.append({"pattern": "file_churn", "files": churn_files[:5]})
            if len(set(all_files)) >= 20:
                patterns.append({"pattern": "large_pr", "count": len(set(all_files))})
            revert_count = sum(1 for c in commit_results if "revert" in c["message"].lower())
            if revert_count >= 2:
                patterns.append({"pattern": "revert_chain", "count": revert_count})
            test_churn = [f for f in churn_files if "test" in f.lower()]
            if test_churn:
                patterns.append({"pattern": "test_churn", "files": test_churn[:3]})
            
            for p in patterns:
                pattern_counter[p["pattern"]] += 1
            
            pr = {
                "repo": repo_name,
                "merge_sha": m["sha"][:12],
                "subject": m["subject"][:80],
                "commit_count": len(commit_results),
                "verdict": verdict,
                "max_score": round(max(scores), 4),
                "mean_score": round(sum(scores)/len(scores), 4),
                "riskiest_commit": riskiest["sha"],
                "total_files": len(set(all_files)),
                "patterns": patterns,
                "commits": commit_results,
                "scoring_time_s": round(elapsed, 2),
            }
            all_results.append(pr)
            scored += 1
            
            n = len(commit_results)
            bucket = "3" if n <= 4 else "10" if n <= 12 else "20"
            timing[bucket].append(elapsed)
            
            pat_names = [p["pattern"] for p in patterns]
            print(f"  PR {m['sha'][:12]}: {len(commit_results)}c, verdict={verdict}, "
                  f"max={max(scores):.3f}, patterns={pat_names}, time={elapsed:.1f}s")
            for cr in commit_results:
                rules_fired = sum(1 for r in cr["rules"] if not r["passed"])
                print(f"    {cr['sha']}: score={cr['score']:.3f} band={cr['band']} "
                      f"rules={cr['rule_count']} ({rules_fired} fired) shap=True")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {len(all_results)} PRs scored")
    print(f"{'='*60}")
    print(f"\n  Pattern fire rates:")
    for pat, count in pattern_counter.most_common():
        print(f"    {pat}: {count}/{len(all_results)} ({count*100/len(all_results):.0f}%)")
    if not pattern_counter:
        print("    (none)")
    
    band_dist = Counter(pr["verdict"] for pr in all_results)
    print(f"\n  Verdict distribution: {dict(band_dist)}")
    
    print(f"\n  Scoring timing (end-to-end including extraction):")
    for bucket, times in sorted(timing.items()):
        if times:
            print(f"    {bucket}-commit PRs: p50={sorted(times)[len(times)//2]:.1f}s, "
                  f"mean={sum(times)/len(times):.1f}s, n={len(times)}")
    
    out_path = str(REPO_ROOT / "data" / "w71_real_prs.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

if __name__ == "__main__":
    main()
