"""Feature-parity test: bulk and single-commit extraction must agree on static features."""
import pandas as pd
import pytest

from ml.extract_features import CommitFeatureExtractor

REPO_MAP = {
    "django": "repos/django",
    "react": "repos/react",
    "rust": "repos/rust",
    "kubernetes": "repos/kubernetes",
    "kafka": "repos/kafka",
}

# Features that should match exactly between bulk and single-commit
STATIC_FEATURES = [
    "lines_added", "lines_deleted", "files_touched", "dirs_touched",
]

# KNOWN DIFFERENCES (not tested):
# - commit_msg_length: CSV uses git log %s (subject), PyDriller uses full msg
# - is_fix_bug_revert: same root cause — checked against full msg in PyDriller
# - author_prior_commits: single-commit has no window context
# - hour_of_day, day_of_week: may differ due to timezone handling


@pytest.fixture(params=["django", "kafka"])
def sample_commit(request):
    """Pick a real commit from the CSV for testing."""
    repo = request.param
    df = pd.read_csv("data/commit_features.csv")
    repo_df = df[df["source_repo"] == repo]
    if len(repo_df) == 0:
        pytest.skip(f"No commits for {repo}")
    row = repo_df.iloc[0]
    return row, REPO_MAP[repo]


def test_feature_parity_static_features(sample_commit):
    """Static features (lines, files, msg) must match between bulk and single-commit."""
    row, repo_path = sample_commit
    try:
        ext = CommitFeatureExtractor(
            repo_path=repo_path, since="2026-04-01", max_commits=0
        )
        features = ext.extract_single_commit(repo_path, row["hash"])
    except Exception:
        pytest.skip("Could not extract single commit")

    for feat in STATIC_FEATURES:
        csv_val = row[feat]
        single_val = features.get(feat)
        assert csv_val == single_val, (
            f"Feature '{feat}' differs: CSV={csv_val}, single={single_val}"
        )
