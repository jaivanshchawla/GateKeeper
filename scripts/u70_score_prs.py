#!/usr/bin/env python3
"""U.7.0d: Score recent commits as proxy PRs for the dashboard PR view."""
import json, sys, os, subprocess, time, traceback
from pathlib import Path
from datetime import datetime, timezone

os.chdir(Path(__file__).parent.parent)
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml, numpy as np, skops.io as sio
from ml.single_commit_features import compute_single_commit_m1_features

REPO_MAP = {
    "django": Path("repos/django"),
    "react": Path("repos/react"),
    "kafka": Path("repos/kafka"),
    "kubernetes": Path("repos/kubernetes"),
    "rust": Path("repos/rust"),
}

with open("ml/config.yaml") as f:
    config = yaml.safe_load(f)
feature_cols = config.get("feature_columns", [])

model_path = "models/gatekeeper_risk_model.skops"
trusted = [
    "collections.OrderedDict", "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier", "numpy.dtype", "numpy.ndarray",
    "pandas.core.frame.DataFrame", "pandas.core.series.Series",
]
model = sio.loads(open(model_path, "rb").read(), trusted=trusted)


def get_recent_commits(repo_name, rp, n=3):
    """Get N recent multi-file non-merge commits."""
    try:
        out = subprocess.check_output(
            ["git", "log", "--no-merges", "--since=2026-06-01",
             f"-{n*10}", "--format=%H|%ct|%an|%aE|%s"],
            cwd=rp, text=True, timeout=30
        )
    except Exception as e:
        print(f"  git log failed: {e}")
        return []

    commits = []
    for line in out.strip().split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|", 4)
        if len(parts) < 4:
            continue
        sha, ts, author, author_email = parts[0], int(parts[1]), parts[2], parts[3]
        message = parts[4] if len(parts) > 4 else ""

        try:
            files_out = subprocess.check_output(
                ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", sha],
                cwd=rp, text=True, timeout=10
            )
            files = set(f.strip() for f in files_out.strip().split("\n") if f.strip())
        except Exception:
            files = set()

        try:
            numstat = subprocess.check_output(
                ["git", "diff-tree", "--no-commit-id", "--numstat", "-r", sha],
                cwd=rp, text=True, timeout=10
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
        except Exception:
            lines_added = lines_deleted = 0

        if len(files) > 1:
            commits.append({
                "hash": sha, "timestamp": ts, "author": author,
                "author_email": author_email, "message": message[:120],
                "files": files, "lines_added": lines_added,
                "lines_deleted": lines_deleted, "repo": repo_name,
            })
        if len(commits) >= n:
            break
    return commits


def score_commit(commit, rp):
    sha = commit["hash"]
    dt = datetime.fromtimestamp(commit["timestamp"], tz=timezone.utc)
    files = commit["files"]

    result = compute_single_commit_m1_features(
        repo_path=str(rp), commit_hash=sha, commit_date=dt,
        author_name=commit["author"], touched_files=files,
        lines_added=commit["lines_added"], lines_deleted=commit["lines_deleted"],
    )

    feat_values = []
    for fc in feature_cols:
        val = result.get(fc, 0.0)
        if val is None:
            val = 0.0
        feat_values.append(float(val))

    feat_array = np.array([feat_values])
    prob = float(model.predict_proba(feat_array)[0][1])
    band = "high" if prob >= 0.9 else "medium" if prob >= 0.75 else "low"

    return {
        "hash": sha,
        "author": commit["author"],
        "author_email": commit["author_email"],
        "timestamp": dt.isoformat(),
        "message": commit["message"],
        "score": prob,
        "band": band,
        "lines_added": commit["lines_added"],
        "lines_deleted": commit["lines_deleted"],
        "files_count": len(files),
        "files": sorted(files)[:10],
        "rule_results": [],
    }


def main():
    all_prs = []

    for repo_name, rp in REPO_MAP.items():
        if not rp.exists():
            continue
        print(f"\n=== {repo_name} ===")
        commits = get_recent_commits(repo_name, rp, n=3)
        if not commits:
            print("  No suitable multi-file commits found")
            continue

        for c in commits:
            sha_short = c["hash"][:8]
            try:
                scored = score_commit(c, rp)
                pr = {
                    "id": f"{repo_name}-{sha_short}",
                    "number": sha_short,
                    "repo": repo_name,
                    "verdict": scored["band"],
                    "commit_count": 1,
                    "file_count": scored["files_count"],
                    "total_lines_added": scored["lines_added"],
                    "total_lines_deleted": scored["lines_deleted"],
                    "mean_score": scored["score"],
                    "max_score": scored["score"],
                    "min_score": scored["score"],
                    "riskiest_sha": c["hash"],
                    "created_at": scored["timestamp"],
                    "commits": [scored],
                }
                all_prs.append(pr)
                print(f"  #{sha_short}: {scored['band']} "
                      f"({scored['files_count']} files, score={scored['score']:.3f}) "
                      f"msg={c['message'][:50]}")
            except Exception as e:
                print(f"  #{sha_short}: FAILED - {e}")
                traceback.print_exc()

    result = {"prs": all_prs}
    out = Path("data/scored_prs.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {len(all_prs)} PRs to {out}")


if __name__ == "__main__":
    main()
