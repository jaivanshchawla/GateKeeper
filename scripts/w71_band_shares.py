#!/usr/bin/env python3
"""W7.1: Re-run all band-dependent numbers with corrected thresholds."""
import json, yaml, pandas as pd, numpy as np
from pathlib import Path
from ml.scoring import _load_config, _determine_band, _load_model, reset_cache

reset_cache()
config = yaml.safe_load(open("ml/config.yaml"))
feature_columns = config["feature_columns"]

model = _load_model()
df = pd.read_csv("data/commit_features.csv")

print("=" * 80)
print("W7.1a: Band shares on TRAINING data (correct thresholds)")
print("=" * 80)
print(f"{'Repo':<12} {'N':>6} {'High':>8} {'Medium':>8} {'Low':>8} {'High%':>8} {'Med%':>8} {'Low%':>8}")
print("-" * 80)
all_results = {}
for repo in ["django", "react", "kafka", "kubernetes", "rust"]:
    rdf = df[df["source_repo"] == repo].copy()
    X = rdf[feature_columns].fillna(0).values
    scores = model.predict_proba(X)[:, 1]
    bands = [_determine_band(float(s), repo) for s in scores]
    n = len(bands)
    high = sum(1 for b in bands if b == "high")
    med = sum(1 for b in bands if b == "medium")
    low = sum(1 for b in bands if b == "low")
    all_results[repo] = {"n": n, "high": high, "medium": med, "low": low,
                          "high_pct": round(high/n*100, 1), "med_pct": round(med/n*100, 1),
                          "low_pct": round(low/n*100, 1)}
    print(f"{repo:<12} {n:>6} {high:>8} {med:>8} {low:>8} {high/n*100:>7.1f}% {med/n*100:>7.1f}% {low/n*100:>7.1f}%")

# Pooled
all_bands = []
for repo in ["django", "react", "kafka", "kubernetes", "rust"]:
    rdf = df[df["source_repo"] == repo]
    X = rdf[feature_columns].fillna(0).values
    scores = model.predict_proba(X)[:, 1]
    all_bands.extend([_determine_band(float(s), repo) for s in scores])
n = len(all_bands)
ph = sum(1 for b in all_bands if b == "high")
pm = sum(1 for b in all_bands if b == "medium")
pl = sum(1 for b in all_bands if b == "low")
print("-" * 80)
print(f"{'POOLED':<12} {n:>6} {ph:>8} {pm:>8} {pl:>8} {ph/n*100:>7.1f}% {pm/n*100:>7.1f}% {pl/n*100:>7.1f}%")
print()

print("=" * 80)
print("W7.1b: Budget default recommendation")
print("=" * 80)
# Budget is scoped to default-branch commits reaching main
# For now, compute high-band share per repo (what fraction of commits are high)
print("High-band share per repo (these ARE the riskiest ~10% by construction):")
for repo, r in all_results.items():
    print(f"  {repo}: {r['high_pct']}% high, {r['med_pct']}% medium")
print()
print("Since bands are per-repo percentiles, ~10% high by construction.")
print("A budget cap should be on high-band commits MERGED TO MAIN, not all scored.")
print("Recommended default: 25% of default-branch merges in a 30-day window.")
print("(Allows repos to have occasional high-risk bursts without constant flapping.)")
print()

print("=" * 80)
print("W7.1f: Band-share drift baseline per repo")
print("=" * 80)
drift_baseline = {}
for repo in ["django", "react", "kafka", "kubernetes", "rust"]:
    rdf = df[df["source_repo"] == repo]
    X = rdf[feature_columns].fillna(0).values
    scores = model.predict_proba(X)[:, 1]
    bands = [_determine_band(float(s), repo) for s in scores]
    n = len(bands)
    high_pct = sum(1 for b in bands if b == "high") / n
    drift_baseline[repo] = {"high_share": round(high_pct, 4), "n": n}
    print(f"  {repo}: high_share={high_pct:.4f} (baseline for drift detection)")
    
# Save drift baseline
Path("data").mkdir(exist_ok=True)
with open("data/band_share_baseline.json", "w") as f:
    json.dump(drift_baseline, f, indent=2)
print()
print("Saved to data/band_share_baseline.json")
