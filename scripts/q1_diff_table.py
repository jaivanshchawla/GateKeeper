#!/usr/bin/env python3
"""
Q.1: Print the FULL 50-commit x 35-feature diff table.
10 commits from each of 5 repos, all 35 features compared.
"""
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, ".")

from ml.extract_features import CommitFeatureExtractor
from ml.train import load_config

REPOS = {
    "django": "repos/django",
    "react": "repos/react",
    "rust": "repos/rust",
    "kubernetes": "repos/kubernetes",
    "kafka": "repos/kafka",
}

FEATURE_COLS = load_config("ml/config.yaml")["feature_columns"]
assert len(FEATURE_COLS) == 35, f"Expected 35 features, got {len(FEATURE_COLS)}"

# Load CSV
df = pd.read_csv("data/commit_features.csv")
print(f"CSV has {len(df)} rows, {df['source_repo'].nunique()} repos")
print(f"Features in config: {len(FEATURE_COLS)}")

# Verify all features are in CSV
missing = [f for f in FEATURE_COLS if f not in df.columns]
if missing:
    print(f"MISSING from CSV: {missing}")
    sys.exit(1)

# Sample 10 commits from each repo
samples = []
for repo in ["django", "react", "rust", "kubernetes", "kafka"]:
    repo_df = df[df["source_repo"] == repo].sample(10, random_state=42)
    samples.append(repo_df)
sample_df = pd.concat(samples, ignore_index=True)
print(f"\nSampled {len(sample_df)} commits across 5 repos")

# Compute features via extract_single_commit for each
results = []
for idx, row in sample_df.iterrows():
    repo_name = row["source_repo"]
    commit_hash = row["hash"]
    repo_path = REPOS[repo_name]

    extractor = CommitFeatureExtractor(
        repo_path=repo_path,
        since="2024-07-01",
        label_window_days=7,
    )

    try:
        sc_features = extractor.extract_single_commit(repo_path, commit_hash)
    except Exception as e:
        print(f"  ERROR for {commit_hash[:8]}: {e}")
        continue

    # Compare each feature
    diff_row = {
        "repo": repo_name,
        "hash": commit_hash[:8],
    }
    for feat in FEATURE_COLS:
        bulk_val = row[feat]
        sc_val = sc_features.get(feat, None)
        if sc_val is None:
            diff_row[f"{feat}_bulk"] = bulk_val
            diff_row[f"{feat}_sc"] = "MISSING"
            diff_row[f"{feat}_diff"] = "MISSING"
        else:
            try:
                bv = float(bulk_val)
                sv = float(sc_val)
                diff = abs(bv - sv)
                diff_row[f"{feat}_bulk"] = bv
                diff_row[f"{feat}_sc"] = sv
                diff_row[f"{feat}_diff"] = diff
            except (ValueError, TypeError):
                diff_row[f"{feat}_bulk"] = bulk_val
                diff_row[f"{feat}_sc"] = sc_val
                diff_row[f"{feat}_diff"] = "TYPE_ERR"

    results.append(diff_row)
    sys.stdout.write(f"\r  {idx+1}/{len(sample_df)}")
    sys.stdout.flush()

print(f"\n\n{'='*120}")
print("FULL DIFF TABLE — 50 commits x 35 features")
print(f"{'='*120}")

# Print per-commit summary first
print(f"\n{'Commit':<20} {'Repo':<12} {'Mismatches':<12}")
print("-" * 44)
for r in results:
    mismatches = sum(1 for feat in FEATURE_COLS if r[f"{feat}_diff"] != 0 and r[f"{feat}_diff"] != "MISSING" and r[f"{feat}_diff"] != "TYPE_ERR")
    print(f"{r['hash']:<20} {r['repo']:<12} {mismatches:<12}")

# Print per-feature summary
print(f"\n{'='*120}")
print("PER-FEATURE SUMMARY")
print(f"{'='*120}")
print(f"{'Feature':<45} {'Mismatches':<12} {'Mean |Δ|':<15} {'Max |Δ|':<15} {'Mean Bulk':<15}")
print("-" * 102)

total_mismatches = 0
for feat in FEATURE_COLS:
    mismatches = 0
    abs_diffs = []
    bulk_vals = []
    for r in results:
        d = r[f"{feat}_diff"]
        if d == "MISSING" or d == "TYPE_ERR":
            mismatches += 1
        elif float(d) > 0.001:  # tolerance for floating point
            mismatches += 1
            abs_diffs.append(float(d))
        else:
            abs_diffs.append(float(d))
        try:
            bulk_vals.append(float(r[f"{feat}_bulk"]))
        except:
            pass
    total_mismatches += mismatches
    mean_d = np.mean(abs_diffs) if abs_diffs else 0
    max_d = np.max(abs_diffs) if abs_diffs else 0
    mean_b = np.mean(bulk_vals) if bulk_vals else 0
    print(f"{feat:<45} {mismatches:>4}/50      {mean_d:>10.4f}      {max_d:>10.4f}      {mean_b:>10.4f}")

total_cells = len(results) * len(FEATURE_COLS)
print(f"\nTotal mismatches: {total_mismatches} / {total_cells} ({100*total_mismatches/total_cells:.1f}%)")

# Now print the FULL raw diff table (all 50 commits x 35 features)
print(f"\n{'='*120}")
print("RAW DIFF TABLE — every cell")
print(f"{'='*120}")

# Print header
header = f"{'Repo':<8} {'Hash':<9}"
for feat in FEATURE_COLS:
    header += f" {feat[:12]:>13}"
print(header)
print("-" * len(header))

for r in results:
    line = f"{r['repo']:<8} {r['hash']:<9}"
    for feat in FEATURE_COLS:
        d = r[f"{feat}_diff"]
        if d == "MISSING":
            line += f" {'MISS':>13}"
        elif d == "TYPE_ERR":
            line += f" {'ERR':>13}"
        elif float(d) > 0.001:
            line += f" {float(d):>13.4f}"
        else:
            line += f" {'0':>13}"
    print(line)
