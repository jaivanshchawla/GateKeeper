#!/usr/bin/env python3
"""
Parts C2 + D2: Leakage analysis, protocols, bootstrap CIs, feature ablation.
Run from gatekeeper/ directory.
"""

import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ── Config ─────────────────────────────────────────────────────────────
sys.path.insert(0, ".")
import yaml

with open("ml/config.yaml") as _f:
    FEATURE_COLS = yaml.safe_load(_f)["feature_columns"]

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
LABEL_WINDOW_DAYS = 7
UNTIL = "2026-06-30"


def load_data():
    df = pd.read_csv("data/commit_features.csv")
    df["committer_date"] = pd.to_datetime(df["committer_date"], utc=True)
    return df


def train_evaluate(X_train, y_train, X_test, y_test):
    """Train LightGBM with fixed params, return metrics dict."""
    model = LGBMClassifier(
        num_leaves=31, learning_rate=0.05, n_estimators=100,
        random_state=RANDOM_SEED, verbose=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "model": model,
        "y_pred": y_pred,
        "y_proba": model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred,
    }


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, ci=0.95):
    """Compute bootstrap confidence interval for mean."""
    rng = np.random.RandomState(RANDOM_SEED)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(np.mean(values)), float(lo), float(hi)


# ═══════════════════════════════════════════════════════════════════════
#  C2.1 — Same-file leakage
# ═══════════════════════════════════════════════════════════════════════

def measure_same_file_leakage(df, train_mask, test_mask):
    """Count test commits sharing >=1 touched file with a train commit
    within LABEL_WINDOW_DAYS."""
    # Build file-touch index from training set
    file_to_train_hashes = defaultdict(set)
    for _, row in df[train_mask].iterrows():
        files = str(row.get("touched_files", "")).split("|")
        for f in files:
            if f:
                file_to_train_hashes[f].add(row["hash"])

    # For each test commit, check if any of its files were touched by a
    # training commit within label_window_days
    overlapping = 0
    for _, row in df[test_mask].iterrows():
        test_files = str(row.get("touched_files", "")).split("|")
        test_date = row["committer_date"]
        for f in test_files:
            if f in file_to_train_hashes:
                for th in file_to_train_hashes[f]:
                    train_date = df[df["hash"] == th]["committer_date"].iloc[0]
                    if abs((test_date - train_date).days) <= LABEL_WINDOW_DAYS:
                        overlapping += 1
                        break
                else:
                    continue
                break

    total_test = test_mask.sum()
    pct = overlapping / max(total_test, 1) * 100
    return overlapping, total_test, pct


# ═══════════════════════════════════════════════════════════════════════
#  C2.3 — Four protocols
# ═══════════════════════════════════════════════════════════════════════

def protocol_a_pooled_random(df):
    """(a) Pooled random 80/20 split."""
    from sklearn.model_selection import train_test_split
    X = df[FEATURE_COLS]
    y = df["risky"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    return train_evaluate(X_train, y_train, X_test, y_test)


def protocol_b_purged_time(df):
    """(b) Purged time-ordered split with 7-day embargo."""
    df_sorted = df.sort_values("committer_date").reset_index(drop=True)
    n = len(df_sorted)
    split_idx = int(n * 0.8)
    boundary_date = df_sorted.iloc[split_idx]["committer_date"]
    embargo_end = boundary_date + pd.Timedelta(days=LABEL_WINDOW_DAYS)

    train_mask = df_sorted["committer_date"] < boundary_date
    purge_mask = (df_sorted["committer_date"] >= boundary_date) & \
                 (df_sorted["committer_date"] <= embargo_end)
    test_mask = df_sorted["committer_date"] > embargo_end

    X_train = df_sorted.loc[train_mask, FEATURE_COLS]
    y_train = df_sorted.loc[train_mask, "risky"]
    X_test = df_sorted.loc[test_mask, FEATURE_COLS]
    y_test = df_sorted.loc[test_mask, "risky"]

    print(f"    Purged: train={train_mask.sum()}, purged={purge_mask.sum()}, "
          f"test={test_mask.sum()}")
    return train_evaluate(X_train, y_train, X_test, y_test)


def protocol_c_leave_one_repo_out(df):
    """(c) Leave-one-repo-out: train on 4, test on 1."""
    results = {}
    for repo in df["source_repo"].unique():
        train_mask = df["source_repo"] != repo
        test_mask = df["source_repo"] == repo
        X_train = df.loc[train_mask, FEATURE_COLS]
        y_train = df.loc[train_mask, "risky"]
        X_test = df.loc[test_mask, FEATURE_COLS]
        y_test = df.loc[test_mask, "risky"]
        res = train_evaluate(X_train, y_train, X_test, y_test)
        results[repo] = res
    return results


# ═══════════════════════════════════════════════════════════════════════
#  D2 — Feature ablation with bootstrap CIs
# ═══════════════════════════════════════════════════════════════════════

def feature_ablation(df):
    """Leave-one-feature-out ablation with bootstrap CIs on F1."""
    from sklearn.model_selection import train_test_split

    X = df[FEATURE_COLS].copy()
    y = df["risky"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    results = {}

    # Baseline: all features
    base = train_evaluate(X_train, y_train, X_test, y_test)
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    mean_f1, lo, hi = bootstrap_ci(
        np.where(base["y_pred"] == y_test_np, 1.0, 0.0)
    )
    results["all"] = {"f1": base["f1"], "ci": (lo, hi)}

    print(f"\n  {'Feature':<25} {'F1':>6} {'95% CI':>14} {'Δ':>6}")
    print(f"  {'-'*55}")
    print(f"  {'(all features)':<25} {base['f1']:>6.4f} "
          f"[{lo:.4f}, {hi:.4f}] {'---':>6}")

    # Leave-one-out
    for feat in FEATURE_COLS:
        feats_minus = [f for f in FEATURE_COLS if f != feat]
        X_tr = X_train[feats_minus]
        X_te = X_test[feats_minus]
        res = train_evaluate(X_tr, y_train, X_te, y_test)
        y_pred_binary = np.where(res["y_pred"] == y_test_np, 1.0, 0.0)
        mean_f1, lo, hi = bootstrap_ci(y_pred_binary)
        delta = res["f1"] - base["f1"]
        results[feat] = {"f1": res["f1"], "ci": (lo, hi), "delta": delta}
        print(f"  -{feat:<24} {res['f1']:>6.4f} "
              f"[{lo:.4f}, {hi:.4f}] {delta:>+6.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  C2.6 — Label-density relationship
# ═══════════════════════════════════════════════════════════════════════

def label_density_analysis(df):
    """Print risky_rate vs commits/week per repo and correlation."""
    print("\n  Label-Density Relationship:")
    print(f"  {'Repo':<15} {'Commits/wk':>11} {'Risky%':>8} {'Unique Files':>13}")
    print(f"  {'-'*50}")

    weeks = []
    rates = []
    for repo in df["source_repo"].unique():
        sub = df[df["source_repo"] == repo]
        dates = sub["committer_date"]
        span_weeks = max((dates.max() - dates.min()).days / 7, 1)
        cpw = len(sub) / span_weeks
        risky = sub["risky"].mean() * 100
        n_files = sub["touched_files"].str.split("|").explode().nunique()
        weeks.append(cpw)
        rates.append(risky)
        print(f"  {repo:<15} {cpw:>11.1f} {risky:>7.1f}% {n_files:>13}")

    corr = np.corrcoef(weeks, rates)[0, 1]
    print(f"\n  Pearson correlation (commits/wk vs risky%): {corr:.3f}")
    return corr


# ═══════════════════════════════════════════════════════════════════════
#  C2.5 — Per-repo F1 with bootstrap CIs
# ═══════════════════════════════════════════════════════════════════════

def per_repo_f1_with_ci(df):
    """Leave-one-repo-out F1 with bootstrap CIs."""
    results = protocol_c_leave_one_repo_out(df)
    print("\n  Per-Repo F1 with Bootstrap CIs (Leave-One-Repo-Out):")
    print(f"  {'Repo':<15} {'F1':>6} {'95% CI':>14}")
    print(f"  {'-'*38}")
    f1_vals = []
    ci_lo_vals = []
    ci_hi_vals = []
    for repo, res in results.items():
        f1 = res["f1"]
        # Bootstrap F1 by resampling test-set predictions
        y_pred = res["y_pred"]
        test_mask = df["source_repo"] == repo
        y_test = df.loc[test_mask, "risky"].values
        rng = np.random.RandomState(RANDOM_SEED)
        boot_f1s = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(len(y_test), size=len(y_test), replace=True)
            boot_f1s.append(f1_score(y_test[idx], y_pred[idx], zero_division=0))
        lo = float(np.percentile(boot_f1s, 2.5))
        hi = float(np.percentile(boot_f1s, 97.5))
        f1_vals.append(f1)
        ci_lo_vals.append(lo)
        ci_hi_vals.append(hi)
        print(f"  {repo:<15} {f1:>6.4f} [{lo:.4f}, {hi:.4f}]")

    mean_f1 = np.mean(f1_vals)
    print(f"  {'MEAN':<15} {mean_f1:>6.4f} [{np.mean(ci_lo_vals):.4f}, {np.mean(ci_hi_vals):.4f}]")
    return results


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 60)
    print("PARTS C2 + D2: ANALYSIS ON REBUILT DATASET")
    print("=" * 60)

    df = load_data()
    print(f"\nDataset: {len(df)} rows, {df['source_repo'].nunique()} repos")
    print(f"Columns: {list(df.columns)}")

    # ── C2.1: Same-file leakage ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("C2.1: SAME-FILE LEAKAGE")
    print(f"{'='*60}")
    from sklearn.model_selection import train_test_split
    y = df["risky"]
    idx_train, idx_test = train_test_split(
        range(len(df)), test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    train_mask = df.index.isin(idx_train)
    test_mask = df.index.isin(idx_test)
    overlapping, total_test, pct = measure_same_file_leakage(df, train_mask, test_mask)
    print(f"  Test commits sharing >=1 file with train commit within "
          f"{LABEL_WINDOW_DAYS}d: {overlapping}/{total_test} ({pct:.1f}%)")

    # ── C2.3: Four protocols ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("C2.3: FOUR PROTOCOLS")
    print(f"{'='*60}")

    print("\n  (a) Pooled random 80/20:")
    res_a = protocol_a_pooled_random(df)
    print(f"  Acc={res_a['accuracy']:.4f}  Prec={res_a['precision']:.4f}  "
          f"Rec={res_a['recall']:.4f}  F1={res_a['f1']:.4f}")

    print("\n  (b) Purged time-ordered (7-day embargo):")
    res_b = protocol_b_purged_time(df)
    print(f"  Acc={res_b['accuracy']:.4f}  Prec={res_b['precision']:.4f}  "
          f"Rec={res_b['recall']:.4f}  F1={res_b['f1']:.4f}")

    print("\n  (c) Leave-one-repo-out:")
    res_c = protocol_c_leave_one_repo_out(df)
    print(f"  {'Repo':<15} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'-'*42}")
    for repo, r in res_c.items():
        print(f"  {repo:<15} {r['accuracy']:>6.4f} {r['precision']:>6.4f} "
              f"{r['recall']:>6.4f} {r['f1']:>6.4f}")
    mean_f1_c = np.mean([r["f1"] for r in res_c.values()])
    print(f"  {'MEAN':<15} {'':>6} {'':>6} {'':>6} {mean_f1_c:>6.4f}")

    # ── C2.4: Bootstrap CIs per repo ─────────────────────────────────
    print(f"\n{'='*60}")
    print("C2.4: BOOTSTRAP CIs (LEAVE-ONE-REPO-OUT)")
    print(f"{'='*60}")
    per_repo_f1_with_ci(df)

    # ── C2.6: Label-density ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("C2.6: LABEL-DENSITY RELATIONSHIP")
    print(f"{'='*60}")
    label_density_analysis(df)

    # ── D2: Feature ablation ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("D2: FEATURE ABLATION (leave-one-out with bootstrap CIs)")
    print(f"{'='*60}")
    feature_ablation(df)

    # ── D2.3: Extract single commit timing ────────────────────────────
    print(f"\n{'='*60}")
    print("D2.3: extract_single_commit TIMING")
    print(f"{'='*60}")
    from ml.extract_features import CommitFeatureExtractor

    # Pick one commit from each repo
    test_hashes = []
    for repo in df["source_repo"].unique():
        row = df[df["source_repo"] == repo].iloc[len(df[df["source_repo"] == repo]) // 2]
        test_hashes.append((repo, row["hash"]))

    times = []
    for repo, h in test_hashes:
        rp = f"repos/{repo}"
        ext = CommitFeatureExtractor(repo_path=rp, since="2024-07-01")
        t0_single = time.time()
        ext.extract_single_commit(rp, h)
        elapsed = time.time() - t0_single
        times.append(elapsed)
        print(f"  {repo}: {elapsed*1000:.0f}ms")

    print(f"  Mean: {np.mean(times)*1000:.0f}ms  "
          f"Max: {np.max(times)*1000:.0f}ms  "
          f"Min: {np.min(times)*1000:.0f}ms")

    total_time = time.time() - t0
    print(f"\nTotal analysis time: {total_time:.1f}s")
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
