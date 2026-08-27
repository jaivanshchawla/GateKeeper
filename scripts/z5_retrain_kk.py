#!/usr/bin/env python3
"""Z.5: Temporal train/test split on kafka+kubernetes."""
import os, sys, yaml, numpy as np, skops.io as sio
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from datetime import datetime, timezone
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "commit_features.csv"))
config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")))
fcols = config["feature_columns"]

SPLIT_DATE = "2025-12-01"
SPLIT_TS = int(datetime(2025, 12, 1, tzinfo=timezone.utc).timestamp())

print("=" * 80)
print("Z.5: TEMPORAL TRAIN/TEST ON KAFKA + KUBERNETES")
print(f"Split: train < {SPLIT_DATE}, test >= {SPLIT_DATE}")
print("=" * 80)

for repo in ["kafka", "kubernetes"]:
    rdf = df[df["source_repo"] == repo].copy()
    rdf["cd_ts"] = pd.to_datetime(rdf["committer_date"]).dt.tz_localize(None).astype(np.int64) // 10**9

    train = rdf[rdf["cd_ts"] < SPLIT_TS]
    test = rdf[rdf["cd_ts"] >= SPLIT_TS]

    print(f"\n{'─'*60}")
    print(f"  {repo}: train={len(train)}, test={len(test)}")
    print(f"  Train base: {train['risky'].mean():.1%}, Test base: {test['risky'].mean():.1%}")

    if len(test) < 20 or len(set(test["risky"])) < 2:
        print(f"  Insufficient test data, skipping")
        continue

    X_train = train[fcols].fillna(0)
    y_train = train["risky"]
    X_test = test[fcols].fillna(0)
    y_test = test["risky"]

    # New model (trained on earlier data only)
    m_new = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42, verbose=-1)
    m_new.fit(X_train, y_train)
    scores_new = m_new.predict_proba(X_test)[:, 1]

    # Original model (trained on all data)
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops")
    trusted = ["collections.OrderedDict","lightgbm.basic.Booster","lightgbm.sklearn.LGBMClassifier",
               "numpy.dtype","numpy.ndarray","pandas.core.frame.DataFrame","pandas.core.series.Series"]
    m_orig = sio.loads(open(model_path, "rb").read(), trusted=trusted)
    scores_orig = m_orig.predict_proba(X_test[fcols].fillna(0))[:, 1]

    # Bootstrap CIs
    rng = np.random.RandomState(42)
    def bs_auc(s, a, n=500):
        aucs = []
        for _ in range(n):
            idx = rng.choice(len(a), size=len(a), replace=True)
            if len(np.unique(a[idx])) < 2: continue
            aucs.append(roc_auc_score(a[idx], s[idx]))
        return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

    auc_o, lo_o, hi_o = bs_auc(scores_orig, y_test.values)
    auc_n, lo_n, hi_n = bs_auc(scores_new, y_test.values)
    pr_o = average_precision_score(y_test, scores_orig)
    pr_n = average_precision_score(y_test, scores_new)

    print(f"  Original (trained on all data):  AUC={auc_o:.4f} [{lo_o:.4f},{hi_o:.4f}] PR-AUC={pr_o:.4f}")
    print(f"  Retrained (train < {SPLIT_DATE}):  AUC={auc_n:.4f} [{lo_n:.4f},{hi_n:.4f}] PR-AUC={pr_n:.4f}")
    delta = auc_n - auc_o
    print(f"  Delta: {delta:+.4f} {'(retrained worse — more training data helps)' if delta < -0.01 else '(similar — retraining not needed)' if abs(delta) < 0.01 else '(retrained better — unexpected)'}")
