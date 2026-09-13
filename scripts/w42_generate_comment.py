#!/usr/bin/env python3
"""W4.2: Generate a Gate 2 PR comment body for a real commit."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.score_pr import format_markdown
from ml.scoring import evaluate_commit_full

import pandas as pd, yaml

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)
fcols = config.get("feature_columns", [])

df = pd.read_csv(os.path.join(DATA, "commit_features.csv"))
rdf = df[df["source_repo"] == "django"].copy()
row = rdf.iloc[500]

fv = [float(row.get(c, 0)) for c in fcols]
result = evaluate_commit_full(
    feature_values=fv,
    repo_name="django",
    commit_hash=row["hash"],
    author=row.get("author", ""),
    message=str(row.get("commit_msg", "")),
    files=[],
    lines_added=int(row.get("lines_added", 0)),
    lines_deleted=int(row.get("lines_deleted", 0)),
    files_touched=int(row.get("files_touched", 0)),
    is_merge=bool(row.get("is_merge", 0)),
    hour_of_day=int(row.get("hour_of_day", 12)),
    day_of_week=int(row.get("day_of_week", 0)),
    author_prior_commits=int(row.get("author_prior_commits", 0)),
    file_revert_count_max=int(row.get("file_revert_count_max", 0)),
    file_prior_changes_max=int(row.get("file_prior_changes_max", 0)),
)

markdown = format_markdown(
    commit_hash=row["hash"],
    risk_score=result.risk_score,
    risk_label=result.band,
    factors=[],
    author=row.get("author", "unknown"),
    explanations=result.shap_top3,
    touched_files_info=[],
    rule_results=result.rule_results,
)

print(markdown)
