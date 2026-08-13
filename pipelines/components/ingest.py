"""
Ingest component for the Gatekeeper retraining pipeline.
Clones/pulls django/django and mines commits via CommitFeatureExtractor.
"""

import os
import subprocess
from collections import defaultdict
from datetime import datetime

from kfp import dsl


# Inline the core extraction logic to avoid import issues in kfp.local subprocesses.
# This reuses the same algorithm as ml/extract_features.py without duplicating
# the design — the class is just embedded here so the component is self-contained
# when kfp.local runs it in an isolated subprocess.
class _IngestCommitFeatureExtractor:
    """Inline copy of CommitFeatureExtractor for self-contained KFP component."""

    def __init__(self, repo_path, since, label_window_days=7):
        self.repo_path = repo_path
        self.since = since
        self.label_window_days = label_window_days
        self.author_prior_counts = defaultdict(int)
        self.file_touches = defaultdict(list)
        self.commit_info = {}

    def _extract_features_from_commit(self, commit):
        lines_added = commit.insertions
        lines_deleted = commit.deletions
        files_touched = commit.files

        touched_files = set()
        directories = set()

        for modified_file in commit.modified_files:
            file_path = modified_file.new_path or modified_file.old_path
            if file_path:
                touched_files.add(file_path)
                dir_path = os.path.dirname(file_path)
                if dir_path:
                    directories.add(dir_path)
                self.file_touches[file_path].append((commit.hash, commit.author_date))

        num_directories = len(directories)
        self.commit_info[commit.hash] = {
            "date": commit.author_date,
            "files": touched_files,
            "msg": commit.msg or "",
        }

        author_name = commit.author.name
        author_prior_commits = self.author_prior_counts[author_name]

        commit_date = commit.author_date
        hour_of_day = commit_date.hour
        day_of_week = commit_date.weekday()

        commit_msg = commit.msg or ""
        commit_msg_length = len(commit_msg)
        is_fix_bug_revert = any(
            kw in commit_msg.lower() for kw in ["fix", "bug", "revert"]
        )

        self.author_prior_counts[author_name] += 1

        return {
            "hash": commit.hash,
            "author": author_name,
            "date": commit_date,
            "lines_added": lines_added,
            "lines_deleted": lines_deleted,
            "files_touched": files_touched,
            "dirs_touched": num_directories,
            "author_prior_commits": author_prior_commits,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "commit_msg_length": commit_msg_length,
            "is_fix_bug_revert": int(is_fix_bug_revert),
            "commit_msg": commit_msg,
        }

    def extract_and_save(self, output_path):
        import pandas as pd

        print(f"Mining commits from {self.repo_path} since {self.since}...")
        since_date = datetime.strptime(self.since, "%Y-%m-%d")
        from pydriller import Repository

        repository = Repository(self.repo_path, since=since_date)

        features = []
        for commit in repository.traverse_commits():
            feature_dict = self._extract_features_from_commit(commit)
            features.append(feature_dict)

        print(f"Collected {len(features)} commits.")
        df = pd.DataFrame(features)

        # Label commits
        df["risky"] = 0
        for idx, row in df.iterrows():
            if "revert" in row["commit_msg"].lower():
                df.at[idx, "risky"] = 1

        for file_path, touches in self.file_touches.items():
            touches.sort(key=lambda x: x[1])
            for i, (hash_i, date_i) in enumerate(touches):
                for j in range(i + 1, len(touches)):
                    hash_j, date_j = touches[j]
                    if (date_j - date_i).days <= self.label_window_days:
                        idx = df[df["hash"] == hash_i].index
                        if len(idx) > 0:
                            df.at[idx[0], "risky"] = 1
                    else:
                        break

        if "commit_msg" in df.columns:
            df = df.drop(columns=["commit_msg"])

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)

        total = len(df)
        positive = int(df["risky"].sum())
        print(f"\nClass Balance: {positive} risky ({positive/total:.2%}), "
              f"{total - positive} safe ({(total-positive)/total:.2%})")

        return df


@dsl.component(
    packages_to_install=["pydriller", "pandas", "pyyaml"],
)
def ingest(
    repo_url: str,
    since_date: str,
    label_window_days: int = 7,
) -> str:
    """
    Clone/pull the target repo and extract commit features.

    Args:
        repo_url: Git repository URL to clone
        since_date: Date string (YYYY-MM-DD) to start mining from
        label_window_days: Days after commit to check for re-touches

    Returns:
        Path to the output features CSV
    """
    import os

    # Clone or pull the repo
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    local_path = f"/tmp/{repo_name}"

    if os.path.exists(local_path):
        print(f"Pulling existing repo at {local_path}...")
        subprocess.run(["git", "-C", local_path, "pull"], check=True)
    else:
        print(f"Cloning {repo_url} to {local_path}...")
        subprocess.run(["git", "clone", repo_url, local_path], check=True)

    # Extract features using the inline extractor
    extractor = _IngestCommitFeatureExtractor(
        repo_path=local_path,
        since=since_date,
        label_window_days=label_window_days,
    )

    output_path = os.path.join(local_path, "..", "gatekeeper_features.csv")
    extractor.extract_and_save(output_path)

    print(f"Ingest complete. Features saved to {output_path}")
    return output_path
