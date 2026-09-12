#!/usr/bin/env python3
"""W1.4: Re-score all PRs with shared evaluate_commit_full() to populate rules + SHAP."""
import json, sys, os, subprocess
from pathlib import Path
from datetime import datetime, timezone

os.chdir(Path(__file__).parent.parent)
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.scoring import evaluate_commit_full
import yaml

with open("ml/config.yaml") as f:
    config = yaml.safe_load(f)
feature_cols = config.get("feature_columns", [])

REPO_MAP = {
    "django": Path("repos/django"),
    "react": Path("repos/react"),
    "kafka": Path("repos/kafka"),
    "kubernetes": Path("repos/kubernetes"),
    "rust": Path("repos/rust"),
}


def score_commit(repo_name, rp, sha):
    """Score one commit with full features + rules + SHAP."""
    try:
        from ml.single_commit_features import compute_single_commit_m1_features

        out = subprocess.check_output(
            ["git", "show", "--format=%H|%ct|%an|%aE|%s", "--name-only", "--no-patch", sha],
            cwd=rp, text=True, timeout=10
        )
        parts = out.strip().split("|", 4)
        ts = int(parts[1])
        author = parts[2]
        message = parts[4] if len(parts) > 4 else ""
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)

        files_out = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", sha],
            cwd=rp, text=True, timeout=10
        )
        files = set(f.strip() for f in files_out.strip().split("\n") if f.strip())

        numstat = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--numstat", "-r", sha],
            cwd=rp, text=True, timeout=10
        )
        la = ld = 0
        for nl in numstat.strip().split("\n"):
            ns = nl.split("\t")
            if len(ns) >= 2:
                try: la += int(ns[0]); ld += int(ns[1])
                except ValueError: pass

        try:
            diff_text = subprocess.check_output(
                ["git", "show", "--format=", "--no-patch", "-p", sha],
                cwd=rp, text=True, timeout=10
            )
        except: diff_text = ""

        m1 = compute_single_commit_m1_features(
            repo_path=str(rp), commit_hash=sha, commit_date=dt,
            author_name=author, touched_files=files, lines_added=la, lines_deleted=ld,
        )
        fv = [float(m1.get(fc, 0.0) or 0.0) for fc in feature_cols]

        result = evaluate_commit_full(
            feature_values=fv, repo_name=repo_name, commit_hash=sha,
            author=author, message=message, files=list(files),
            lines_added=la, lines_deleted=ld, diff_text=diff_text, repo_path=str(rp),
        )

        return {
            "hash": sha, "author": author, "author_email": parts[3] if len(parts) > 3 else "",
            "timestamp": dt.isoformat(), "message": message[:120],
            "score": result.risk_score, "band": result.band,
            "lines_added": la, "lines_deleted": ld, "files_count": len(files),
            "files": sorted(files)[:10],
            "rule_results": [r.to_dict() for r in result.rule_results],
            "shap_top3": result.shap_top3,
        }
    except Exception as e:
        print(f"  skip {sha[:8]}: {e}")
        return None


def main():
    all_prs = []
    for repo_name, rp in REPO_MAP.items():
        if not rp.exists(): continue
        print(f"\n=== {repo_name} ===")
        try:
            out = subprocess.check_output(
                ["git", "log", "--no-merges", "--since=2026-06-01", "-5",
                 "--format=%H|%ct|%an|%aE|%s"],
                cwd=rp, text=True, timeout=30
            )
        except: continue

        commits = []
        for line in out.strip().split("\n"):
            if not line or "|" not in line: continue
            p = line.split("|", 4)
            if len(p) < 4: continue
            sha, ts, author, email = p[0], int(p[1]), p[2], p[3]
            msg = p[4] if len(p) > 4 else ""
            try:
                fo = subprocess.check_output(
                    ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", sha],
                    cwd=rp, text=True, timeout=10
                )
                nfiles = len([f for f in fo.strip().split("\n") if f.strip()])
            except: nfiles = 0
            if nfiles > 1:
                commits.append({"sha": sha, "ts": ts, "author": author, "email": email, "message": msg})

        for c in commits[:3]:
            sc = score_commit(repo_name, rp, c["sha"])
            if not sc: continue
            pr = {
                "id": f"{repo_name}-{c['sha'][:8]}",
                "number": c["sha"][:8],
                "repo": repo_name,
                "verdict": sc["band"],
                "commit_count": 1,
                "file_count": sc["files_count"],
                "total_lines_added": sc["lines_added"],
                "total_lines_deleted": sc["lines_deleted"],
                "mean_score": sc["score"],
                "max_score": sc["score"],
                "min_score": sc["score"],
                "riskiest_sha": c["sha"],
                "created_at": sc["timestamp"],
                "commits": [sc],
            }
            all_prs.append(pr)
            rules_fired = [r["rule"] for r in sc["rule_results"] if not r.get("passed", True)]
            print(f"  #{c['sha'][:8]}: {sc['band']} ({sc['files_count']} files, {sc['score']:.3f}) "
                  f"rules={len(rules_fired)} shap={len(sc['shap_top3'])} msg={c['message'][:40]}")

    out = Path("data/scored_prs.json")
    with open(out, "w") as f:
        json.dump({"prs": all_prs}, f, indent=2)
    print(f"\nWrote {len(all_prs)} PRs to {out}")


if __name__ == "__main__":
    main()
