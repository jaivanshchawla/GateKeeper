#!/usr/bin/env python3
"""V5.2: Run simulator on real data — current vs strict config."""
import os, sys, subprocess, time, yaml, json, numpy as np
from datetime import datetime, timedelta, timezone
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPOS = {
    "django": "repos/django",
    "react": "repos/react",
    "kafka": "repos/kafka",
}

STRICT_CONFIG = {
    "rules": {
        "large_change": {"max_lines": 300, "severity": "block"},
        "too_many_files": {"max_files": 10, "severity": "block"},
        "no_tests": {"severity": "block"},
        "config_and_code": {"severity": "block"},
        "revert_hotspot": {"revert_count": 2, "window_days": 30, "severity": "block"},
        "first_touch": {"severity": "warn"},
        "weekend_deploy": {"severity": "warn"},
        "stale_file": {"days": 90, "severity": "warn"},
        "direct_to_main": {"severity": "block"},
    },
    "ml_scoring": {"enabled": True, "band_thresholds": {"high": 0.80, "medium": 0.65}},
    "risk_budget": {"enabled": True, "window_days": 30, "max_high_pct": 0.10},
    "fail_on": ["block"],
}

from rules.engine import RuleEngine, load_config as load_current_config
from rules.base import CommitContext, Severity

current_config = load_current_config()
current_engine = RuleEngine(current_config)
strict_engine = RuleEngine(STRICT_CONFIG)

# Load model
import skops.io as sio
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops")
trusted = ["collections.OrderedDict","lightgbm.basic.Booster","lightgbm.sklearn.LGBMClassifier",
           "numpy.dtype","numpy.ndarray","pandas.core.frame.DataFrame","pandas.core.series.Series"]
model = sio.loads(open(model_path, "rb").read(), trusted=trusted)
config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")))
fcols = config["feature_columns"]
thresholds = config.get("thresholds", {})

def get_band(score, repo=""):
    rt = thresholds.get(repo, thresholds.get("_global", {"high": 0.86, "medium": 0.75}))
    if score >= rt["high"]: return "high"
    elif score >= rt["medium"]: return "medium"
    return "low"

print("=" * 80)
print("V5.2: SIMULATOR ON REAL DATA")
print("=" * 80)

for repo_name, rp_rel in REPOS.items():
    rp = os.path.join(os.path.dirname(__file__), "..", rp_rel)
    if not os.path.exists(rp): continue

    print(f"\n{'─'*60}")
    print(f"  {repo_name}")
    print(f"{'─'*60}")

    since = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
    r = subprocess.run(
        ["git", "log", f"--since={since}", "--no-merges",
         "--format=%H|%ct|%aE|%s", "--max-count=200"],
        cwd=rp, capture_output=True, text=True, timeout=60
    )

    commits = []
    for line in r.stdout.strip().split("\n"):
        if "|" not in line: continue
        parts = line.split("|", 3)
        if len(parts) < 4: continue
        h, ts, author, subject = parts[0], int(parts[1]), parts[2], parts[3]

        fr = subprocess.run(["git", "diff-tree", "--no-commit-id", "-r", "--name-only", h[:8]],
                            cwd=rp, capture_output=True, text=True, timeout=10)
        files = [f.strip() for f in fr.stdout.strip().split("\n") if f.strip()]

        lr = subprocess.run(["git", "diff-tree", "--no-commit-id", "-r", "--numstat", h[:8]],
                            cwd=rp, capture_output=True, text=True, timeout=10)
        la = ld = 0
        for lline in lr.stdout.strip().split("\n"):
            p2 = lline.split("\t")
            if len(p2) >= 2:
                try: la += int(p2[0]) if p2[0] != "-" else 0; ld += int(p2[1]) if p2[1] != "-" else 0
                except: pass

        commits.append({"hash": h, "ts": ts, "author": author, "subject": subject,
                        "files": files, "la": la, "ld": ld,
                        "dirs": len(set(str(os.path.dirname(f)) for f in files if os.path.dirname(f)))})

    # Score and evaluate both configs
    from ml.extract_features import CommitFeatureExtractor
    from ml.single_commit_features import clear_cache
    clear_cache()
    ext = CommitFeatureExtractor(repo_path=rp, since="2024-07-01", label_window_days=7)

    results = []
    t0 = time.time()
    for i, c in enumerate(commits):
        if i > 0 and i % 50 == 0:
            print(f"  ... {i}/{len(commits)} ({time.time()-t0:.0f}s)")

        try:
            feat = ext.extract_single_commit(rp, c["hash"])
            fv = [feat.get(col, 0) for col in fcols]
            score = float(model.predict_proba(np.array([fv]))[0][1])
        except: score = 0.5

        band = get_band(score, repo_name)

        dt = datetime.fromtimestamp(c["ts"], tz=timezone.utc)
        ctx = CommitContext(
            hash=c["hash"], author=c["author"], message=c["subject"],
            files=c["files"], lines_added=c["la"], lines_deleted=c["ld"],
            files_touched=len(c["files"]), dirs_touched=c["dirs"],
            hour_of_day=dt.hour, day_of_week=dt.weekday(),
            risk_score=score, risk_label=band,
        )

        c_results = current_engine.evaluate(ctx)
        s_results = strict_engine.evaluate(ctx)
        c_blocked = current_engine.should_block(c_results)
        s_blocked = strict_engine.should_block(s_results)
        c_hits = [r.rule_name for r in c_results if not r.passed]
        s_hits = [r.rule_name for r in s_results if not r.passed]

        changed = (c_blocked != s_blocked) or (c_hits != s_hits)
        results.append({
            "hash": c["hash"][:8], "date": dt.strftime("%Y-%m-%d"),
            "score": round(score, 3), "band": band,
            "c_blocked": c_blocked, "s_blocked": s_blocked,
            "c_hits": c_hits, "s_hits": s_hits, "changed": changed,
        })

    elapsed = time.time() - t0
    n = len(results)

    # Aggregate
    c_blocked_n = sum(1 for r in results if r["c_blocked"])
    s_blocked_n = sum(1 for r in results if r["s_blocked"])
    c_warned_n = sum(1 for r in results if not r["c_blocked"] and r["c_hits"])
    s_warned_n = sum(1 for r in results if not r["s_blocked"] and r["s_hits"])
    changed_n = sum(1 for r in results if r["changed"])
    newly_blocked = sum(1 for r in results if r["s_blocked"] and not r["c_blocked"])
    newly_freed = sum(1 for r in results if not r["s_blocked"] and r["c_blocked"])

    print(f"\n  Results ({n} commits, {elapsed:.0f}s):")
    print(f"  {'':>20} {'Current':>10} {'Strict':>10}")
    print(f"  {'─'*40}")
    print(f"  {'Blocked':>20} {c_blocked_n:>10} {s_blocked_n:>10}")
    print(f"  {'Warned':>20} {c_warned_n:>10} {s_warned_n:>10}")
    print(f"  {'Changed':>20} {'':>10} {changed_n:>10}")
    print(f"  {'Newly blocked':>20} {'':>10} {newly_blocked:>10}")
    print(f"  {'Newly freed':>20} {'':>10} {newly_freed:>10}")

    # Show some changed commits
    changed_commits = [r for r in results if r["changed"]][:5]
    if changed_commits:
        print(f"\n  Sample changed commits:")
        for r in changed_commits:
            print(f"    {r['hash']} {r['date']} score={r['score']} band={r['band']}")
            print(f"      Current: blocked={r['c_blocked']} hits={r['c_hits']}")
            print(f"      Strict:  blocked={r['s_blocked']} hits={r['s_hits']}")

    # Sanity check
    if s_blocked_n == 0:
        print(f"\n  ⚠ WARNING: Strict config blocks nothing — check config is being read")
    elif s_blocked_n == n:
        print(f"\n  ⚠ WARNING: Strict config blocks everything — check thresholds")
    else:
        print(f"\n  ✓ Sanity check passed: strict blocks {s_blocked_n}/{n} ({s_blocked_n/n*100:.1f}%)")
