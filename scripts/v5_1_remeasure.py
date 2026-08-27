#!/usr/bin/env python3
"""V5.1: Re-extract OOW features with identity resolution ON."""
import os, sys, subprocess, time, yaml, json, numpy as np, skops.io as sio
from datetime import datetime, timezone, timedelta
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPOS = {
    "django": "repos/django",
    "react": "repos/react",
    "kafka": "repos/kafka",
    "kubernetes": "repos/kubernetes",
    "rust": "repos/rust",
}
TRAINING_END = "2026-06-30"
WINDOW_DAYS = 7
MAX_COMMITS = 200  # per repo

# Load model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops")
trusted = ["collections.OrderedDict","lightgbm.basic.Booster","lightgbm.sklearn.LGBMClassifier",
           "numpy.dtype","numpy.ndarray","pandas.core.frame.DataFrame","pandas.core.series.Series"]
model = sio.loads(open(model_path, "rb").read(), trusted=trusted)
config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")))
fcols = config["feature_columns"]

# Load OLD results for comparison
old_results = {}
for repo in REPOS:
    ckpt = os.path.join(os.path.dirname(__file__), "..", "data", f"z1_{repo}_oow.json")
    if os.path.exists(ckpt):
        old_results[repo] = json.load(open(ckpt))

def bootstrap_auc(scores, actuals, n_resamples=1000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_resamples):
        idx = rng.choice(len(actuals), size=len(actuals), replace=True)
        s, a = scores[idx], actuals[idx]
        if len(np.unique(a)) < 2: continue
        aucs.append(roc_auc_score(a, s))
    if len(aucs) < 10: return 0.0, 0.0, 1.0
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


print("=" * 80)
print("V5.1: RE-MEASURE OOW WITH IDENTITY RESOLUTION ON")
print("=" * 80)

# First: check what identities are resolved per repo
print("\n--- Identity Resolution Summary ---")
for repo_name, rp_rel in REPOS.items():
    rp = os.path.join(os.path.dirname(__file__), "..", rp_rel)
    if not os.path.exists(rp):
        continue
    try:
        from policy.identity import build_identity_map
        imap = build_identity_map(rp)
        if imap:
            print(f"  {repo_name}: {len(imap)} aliases resolved")
            for variant, canonical in list(imap.items())[:5]:
                print(f"    {variant} -> {canonical}")
        else:
            print(f"  {repo_name}: no aliases (clean identity)")
    except Exception as e:
        print(f"  {repo_name}: error loading identity map: {e}")

print("\n--- Re-extracting OOW features ---")

for repo_name, rp_rel in REPOS.items():
    rp = os.path.join(os.path.dirname(__file__), "..", rp_rel)
    if not os.path.exists(rp):
        continue

    print(f"\n{'─'*60}")
    print(f"  {repo_name}")
    print(f"{'─'*60}")

    # Get OOW commits
    r = subprocess.run(
        ["git", "log", f"--since={TRAINING_END}", "--no-merges",
         "--format=%H|%ct", f"--max-count={MAX_COMMITS}"],
        cwd=rp, capture_output=True, text=True, timeout=60
    )
    entries = []
    for line in r.stdout.strip().split("\n"):
        if "|" in line:
            h, ts = line.split("|", 1)
            try: entries.append((h.strip(), int(ts)))
            except: pass

    head_r = subprocess.run(["git", "log", "-1", "--format=%ct", "HEAD"],
                            cwd=rp, capture_output=True, text=True, timeout=10)
    head_ts = int(head_r.stdout.strip())
    cutoff_ts = head_ts - WINDOW_DAYS * 86400

    print(f"  OOW commits: {len(entries)}")

    # Force-clear the identity cache to pick up the new resolution
    import ml.m1_shared as m1s
    m1s._identity_map = {}
    m1s._identity_loaded_for = ""

    from ml.extract_features import CommitFeatureExtractor
    from ml.single_commit_features import clear_cache
    clear_cache()
    ext = CommitFeatureExtractor(repo_path=rp, since="2024-07-01", label_window_days=7)

    # Build file index for outcomes
    buffer_start = datetime(2026, 6, 23, tzinfo=timezone.utc)
    all_r = subprocess.run(
        ["git", "log", f"--since={buffer_start.isoformat()}", "--no-merges",
         "--format=COMMIT|%H|%ct", "--name-only"],
        cwd=rp, capture_output=True, text=True, timeout=120
    )
    commits_data = {}
    cur_h, cur_ts, cur_files = None, 0, []
    for line in all_r.stdout.split("\n"):
        if line.startswith("COMMIT|"):
            if cur_h: commits_data[cur_h] = (cur_ts, cur_files)
            parts = line.split("|", 2)
            cur_h = parts[1]; cur_ts = int(parts[2]); cur_files = []
        elif cur_h and line.strip():
            cur_files.append(line.strip())
    if cur_h: commits_data[cur_h] = (cur_ts, cur_files)

    file_commits = defaultdict(list)
    for h, (ts, files) in commits_data.items():
        for f in files:
            file_commits[f].append((ts, h))
    for f in file_commits:
        file_commits[f].sort()

    # Extract features with identity resolution ON
    new_results = {}
    t0 = time.time()
    author_prior_changes = {}  # track changes for V5.1c

    for i, (h, ts) in enumerate(entries):
        if i > 0 and i % 20 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(entries) - i) / rate
            print(f"  ... {i}/{len(entries)} ({elapsed:.0f}s, ETA {eta:.0f}s)")

        try:
            feat = ext.extract_single_commit(rp, h)
            fv = [feat.get(c, 0) for c in fcols]
            score = float(model.predict_proba(np.array([fv]))[0][1])
            apc = feat.get("author_prior_commits", 0)
        except Exception as e:
            score = 0.5
            apc = 0

        # Outcome
        commit_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        window_end_ts = int((commit_dt + timedelta(days=WINDOW_DAYS)).timestamp())
        files_r = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", h[:8]],
            cwd=rp, capture_output=True, text=True, timeout=10
        )
        cfiles = [f.strip() for f in files_r.stdout.strip().split("\n") if f.strip()][:10]
        is_retouched = False
        for fp in cfiles:
            for cts, ch in file_commits.get(fp, []):
                if cts > ts and cts <= window_end_ts and ch[:8] != h[:8]:
                    is_retouched = True
                    break
            if is_retouched: break

        actual = 1 if is_retouched else 0
        within_7d = ts >= cutoff_ts
        new_results[h] = {"ts": ts, "score": score, "actual": actual, "within_7d": within_7d, "apc": apc}

        # Track author_prior_commits change
        if h in old_results.get(repo_name, {}):
            old_apc = old_results[repo_name][h].get("apc", -1)
            if old_apc >= 0 and old_apc != apc:
                author_prior_changes[h] = {"old": old_apc, "new": apc, "delta": apc - old_apc}

    elapsed = time.time() - t0
    print(f"  Scored {len(new_results)} in {elapsed:.0f}s")

    # Compute metrics
    scores_arr = np.array([v["score"] for v in new_results.values()])
    actuals_arr = np.array([v["actual"] for v in new_results.values()])
    n = len(actuals_arr)
    br = float(actuals_arr.mean())

    mean_auc, lo, hi = bootstrap_auc(scores_arr, actuals_arr)
    pr_auc = float(average_precision_score(actuals_arr, scores_arr))

    # Old metrics
    if repo_name in old_results:
        old_scores = np.array([v["score"] for v in old_results[repo_name].values()])
        old_actuals = np.array([v["actual"] for v in old_results[repo_name].values()])
        if len(set(old_actuals)) >= 2:
            old_mean, old_lo, old_hi = bootstrap_auc(old_scores, old_actuals)
        else:
            old_mean = old_lo = old_hi = 0
    else:
        old_mean = old_lo = old_hi = 0

    delta = mean_auc - old_mean if old_mean > 0 else 0

    print(f"\n  Results:")
    print(f"  N: {n}, Base rate: {br:.1%}")
    print(f"  OLD: ROC-AUC={old_mean:.4f} [{old_lo:.4f},{old_hi:.4f}]")
    print(f"  NEW: ROC-AUC={mean_auc:.4f} [{lo:.4f},{hi:.4f}]")
    print(f"  Delta: {delta:+.4f}")

    if author_prior_changes:
        deltas = [v["delta"] for v in author_prior_changes.values()]
        print(f"  author_prior_commits changed: {len(author_prior_changes)} commits")
        print(f"    mean delta: {np.mean(deltas):+.1f}, max: {max(deltas):+.1f}, min: {min(deltas):+.1f}")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", f"v5_1_{repo_name}_oow.json")
    json.dump(new_results, open(out_path, "w"))


print("\n" + "=" * 80)
print("SUMMARY: Before/After Identity Resolution")
print("=" * 80)
print(f"{'Repo':<12} {'OLD AUC':>10} {'NEW AUC':>10} {'Delta':>8} {'APC changes':>12}")
print("─" * 55)
for repo_name in REPOS:
    old_f = os.path.join(os.path.dirname(__file__), "..", "data", f"z1_{repo_name}_oow.json")
    new_f = os.path.join(os.path.dirname(__file__), "..", "data", f"v5_1_{repo_name}_oow.json")
    if os.path.exists(old_f) and os.path.exists(new_f):
        old_d = json.load(open(old_f))
        new_d = json.load(open(new_f))
        old_scores = np.array([v["score"] for v in old_d.values()])
        old_actuals = np.array([v["actual"] for v in old_d.values()])
        new_scores = np.array([v["score"] for v in new_d.values()])
        new_actuals = np.array([v["actual"] for v in new_d.values()])
        if len(set(old_actuals)) >= 2 and len(set(new_actuals)) >= 2:
            old_auc = bootstrap_auc(old_scores, old_actuals)
            new_auc = bootstrap_auc(new_scores, new_actuals)
            delta = new_auc[0] - old_auc[0]
            apc_changes = sum(1 for h in new_d if h in old_d and new_d[h].get("apc", 0) != old_d[h].get("apc", 0))
            print(f"{repo_name:<12} {old_auc[0]:>8.4f}  {new_auc[0]:>8.4f}  {delta:>+7.4f} {apc_changes:>10}")
