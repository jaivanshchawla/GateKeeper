#!/usr/bin/env python3
"""Z.1 final: censoring analysis on all scored repos."""
import os, sys, json, numpy as np
from sklearn.metrics import roc_auc_score
import subprocess
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPOS = {
    "django": "repos/django",
    "react": "repos/react",
    "kafka": "repos/kafka",
    "rust": "repos/rust",
}
LORO = {"django": 0.7607, "react": 0.7579, "kafka": 0.8247, "kubernetes": 0.7952, "rust": 0.8038}


def bootstrap_auc(scores, actuals, n_resamples=500, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_resamples):
        idx = rng.choice(len(actuals), size=len(actuals), replace=True)
        s, a = scores[idx], actuals[idx]
        if len(np.unique(a)) < 2:
            continue
        aucs.append(roc_auc_score(a, s))
    if len(aucs) < 10:
        return 0.0, 0.0, 1.0
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


print("=" * 80)
print("Z.1: RIGHT-CENSORING AT HEAD (DECISIVE TEST)")
print("=" * 80)
print()

gap_all_list = []
gap_vld_list = []
gap_all_excl = []
gap_vld_excl = []

for repo_name, rp_rel in REPOS.items():
    rp = os.path.join(os.path.dirname(__file__), "..", rp_rel)
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "data", f"z1_{repo_name}_oow.json")
    if not os.path.exists(ckpt_path):
        print(f"{repo_name}: SKIPPED (no checkpoint)")
        continue
    d = json.load(open(ckpt_path))

    all_scores = np.array([v["score"] for v in d.values()])
    all_actuals = np.array([v["actual"] for v in d.values()])
    # Use within_7d field if available, else compute from ts
    if "within_7d" in list(d.values())[0]:
        beyond = {h: v for h, v in d.items() if not v.get("within_7d", True)}
    else:
        head_r = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "HEAD"],
            cwd=rp, capture_output=True, text=True, timeout=10,
        )
        head_ts = int(head_r.stdout.strip())
        cutoff_ts = head_ts - 7 * 86400
        beyond = {h: v for h, v in d.items() if v["ts"] < cutoff_ts}
    within_ct = len(d) - len(beyond)
    cens_pct = within_ct / len(d) * 100

    mean_all = lo_all = hi_all = 0.0
    if len(set(all_actuals)) >= 2:
        mean_all, lo_all, hi_all = bootstrap_auc(all_scores, all_actuals)

    mean_v = lo_v = hi_v = 0.0
    if len(beyond) > 10:
        bvals = list(beyond.values())
        v_actuals = np.array([v["actual"] for v in bvals])
        if len(set(v_actuals)) >= 2:
            v_scores = np.array([v["score"] for v in bvals])
            mean_v, lo_v, hi_v = bootstrap_auc(v_scores, v_actuals)

    theirro = LORO.get(repo_name, 0)
    gap_a = mean_all - theirro if mean_all > 0 else 0
    gap_v = mean_v - theirro if mean_v > 0 else 0

    print(f"{repo_name}:")
    print(f"  N={len(d)} | Within 7d of HEAD: {within_ct} ({cens_pct:.1f}%) | Beyond 7d: {len(beyond)}")
    if mean_all > 0:
        print(f"  ALL OOW AUC:    {mean_all:.4f} [{lo_all:.4f},{hi_all:.4f}]  Gap vs LORO: {gap_a:+.4f}")
    if mean_v > 0:
        print(f"  VALID (>7d) AUC: {mean_v:.4f} [{lo_v:.4f},{hi_v:.4f}]  Gap vs LORO: {gap_v:+.4f}")
    else:
        print(f"  VALID (>7d): N={len(beyond)} insufficient")
    print()

    if gap_a != 0:
        gap_all_list.append(gap_a)
    if mean_v > 0:
        gap_vld_list.append(gap_v)
    if repo_name != "react":
        if gap_a != 0:
            gap_all_excl.append(gap_a)
        if mean_v > 0:
            gap_vld_excl.append(gap_v)

print("=" * 80)
print("SUMMARY")
print("=" * 80)
if gap_all_list:
    print(f"All repos mean gap (ALL OOW):  {np.mean(gap_all_list):+.4f}")
if gap_vld_list:
    print(f"All repos mean gap (VALID):    {np.mean(gap_vld_list):+.4f}")
if gap_all_excl:
    print(f"Excl React mean gap (ALL):     {np.mean(gap_all_excl):+.4f}")
if gap_vld_excl:
    print(f"Excl React mean gap (VALID):   {np.mean(gap_vld_excl):+.4f}")
print()
print("INTERPRETATION:")
print("If VALID gap << ALL gap, censoring explains the drop.")
print("If VALID gap ~= ALL gap, censoring is NOT the cause.")
print("If VALID gap ~ 0 excl React, the gap is React-specific, not temporal.")
