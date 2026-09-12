#!/usr/bin/env python3
"""W1.1: Diagnose band boundary bug."""
import yaml, numpy as np, skops.io as sio
import pandas as pd

with open("ml/config.yaml") as f:
    config = yaml.safe_load(f)
thresholds = config.get("thresholds", {})
feature_cols = config.get("feature_columns", [])

model = sio.loads(open("models/gatekeeper_risk_model.skops", "rb").read(), trusted=[
    "collections.OrderedDict", "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier", "numpy.dtype", "numpy.ndarray",
    "pandas.core.frame.DataFrame", "pandas.core.series.Series",
])

df = pd.read_csv("data/commit_features_m1.csv")
print(f"Loaded {len(df)} rows")

for repo in sorted(df["source_repo"].unique()):
    rdf = df[df["source_repo"] == repo].copy()
    X = rdf[feature_cols].fillna(0).values
    scores = model.predict_proba(X)[:, 1]
    total = len(scores)

    repo_thr = thresholds.get(repo, thresholds.get("_global", {}))
    high_cut = repo_thr.get("high", 0.8619)
    medium_cut = repo_thr.get("medium", 0.7536)

    high = int((scores >= high_cut).sum())
    medium = int(((scores >= medium_cut) & (scores < high_cut)).sum())
    low = int((scores < medium_cut).sum())

    correct_high = float(np.percentile(scores, 90))
    correct_medium = float(np.percentile(scores, 75))

    print(f"\n{repo} (n={total}):")
    print(f"  Current thresholds: high={high_cut:.4f}, medium={medium_cut:.4f}")
    print(f"  Current bands: high={high} ({high*100/total:.1f}%), medium={medium} ({medium*100/total:.1f}%), low={low} ({low*100/total:.1f}%)")
    print(f"  Expected (10/15/75): high={int(total*0.10)}, medium={int(total*0.15)}, low={int(total*0.75)}")
    print(f"  Correct cutoffs: high={correct_high:.4f}, medium={correct_medium:.4f}")
    print(f"  HIGH+ MEDIUM+ LOW should equal {total}: {high+medium+low}")

print("\n--- Proposed correct thresholds ---")
for repo in sorted(df["source_repo"].unique()):
    rdf = df[df["source_repo"] == repo].copy()
    X = rdf[feature_cols].fillna(0).values
    scores = model.predict_proba(X)[:, 1]
    h = float(np.percentile(scores, 90))
    m = float(np.percentile(scores, 75))
    print(f"  {repo}:")
    print(f"    high: {h:.4f}")
    print(f"    medium: {m:.4f}")
