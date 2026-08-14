"""
Ingest component for the Gatekeeper retraining pipeline.
Clones or opens a repository and mines commits via the Phase 1 feature logic.
"""

from kfp import dsl


@dsl.component(
    base_image="gatekeeper-kfp-base",
    packages_to_install=["pydriller", "pandas", "pyyaml"],
)
def ingest(
    repo_url: str,
    since_date: str,
    features_path: dsl.OutputPath("Dataset"),
    label_window_days: int = 7,
    cached_csv_path: str = "",
) -> None:
    """
    Clone/pull the target repo and extract commit features.

    Args:
        repo_url: Git repository URL or local repository path.
        since_date: Date string (YYYY-MM-DD) to start mining from.
        features_path: KFP output path for the feature CSV.
        label_window_days: Days after commit to check for re-touches.
        cached_csv_path: If non-empty and file exists, copy this CSV
            to features_path instead of re-mining (saves ~9 min on
            large repos).
    """
    import os
    import subprocess
    import tempfile
    from collections import defaultdict
    from datetime import datetime, timezone
    from pathlib import Path

    import pandas as pd

    # --- Cache check: skip expensive PyDriller mining if CSV exists ---
    if cached_csv_path:
        cached = Path(cached_csv_path)
        if cached.exists() and cached.stat().st_size > 0:
            df = pd.read_csv(cached)
            print(f"Using cached features from {cached} ({len(df)} rows)")
            if "risky" in df.columns:
                total = len(df)
                positive = int(df["risky"].sum())
                print(
                    f"Class Balance: {positive} risky ({positive / total:.2%}), "
                    f"{total - positive} safe ({(total - positive) / total:.2%})"
                )
            output_path = Path(features_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Cached features copied to {output_path}")
            return

    # --- Full mining path (no cache available) ---
    from pydriller import Repository

    class CommitFeatureExtractor:
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
                    self.file_touches[file_path].append(
                        (commit.hash, commit.author_date)
                    )

            author_name = commit.author.name
            commit_date = commit.author_date
            commit_msg = commit.msg or ""
            is_fix_bug_revert = any(
                kw in commit_msg.lower() for kw in ["fix", "bug", "revert"]
            )

            self.commit_info[commit.hash] = {
                "date": commit_date,
                "files": touched_files,
                "msg": commit_msg,
            }

            features = {
                "hash": commit.hash,
                "author": author_name,
                "date": commit_date,
                "lines_added": lines_added,
                "lines_deleted": lines_deleted,
                "files_touched": files_touched,
                "dirs_touched": len(directories),
                "author_prior_commits": self.author_prior_counts[author_name],
                "hour_of_day": commit_date.hour,
                "day_of_week": commit_date.weekday(),
                "commit_msg_length": len(commit_msg),
                "is_fix_bug_revert": int(is_fix_bug_revert),
                "commit_msg": commit_msg,
            }
            self.author_prior_counts[author_name] += 1
            return features

        def extract(self):
            print(f"Mining commits from {self.repo_path} since {self.since}...")
            since_date_obj = datetime.strptime(self.since, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            repository = Repository(self.repo_path, since=since_date_obj)

            rows = [
                self._extract_features_from_commit(commit)
                for commit in repository.traverse_commits()
            ]
            print(f"Collected {len(rows)} commits.")

            df = pd.DataFrame(rows)
            if df.empty:
                return df

            df["risky"] = 0
            for idx, row in df.iterrows():
                if "revert" in row["commit_msg"].lower():
                    df.at[idx, "risky"] = 1

            for touches in self.file_touches.values():
                touches.sort(key=lambda touch: touch[1])
                for i, (hash_i, date_i) in enumerate(touches):
                    for _, date_j in touches[i + 1:]:
                        if (date_j - date_i).days <= self.label_window_days:
                            idx = df[df["hash"] == hash_i].index
                            if len(idx) > 0:
                                df.at[idx[0], "risky"] = 1
                        else:
                            break

            return df.drop(columns=["commit_msg"], errors="ignore")

    repo_path_candidate = Path(repo_url).expanduser()
    if repo_path_candidate.exists():
        local_path = repo_path_candidate.resolve()
        print(f"Using local repository at {local_path}")
    else:
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        local_path = Path(tempfile.gettempdir()) / repo_name

        if local_path.exists():
            print(f"Pulling existing repo at {local_path}...")
            subprocess.run(["git", "-C", str(local_path), "pull"], check=True)
        else:
            print(f"Cloning {repo_url} to {local_path}...")
            subprocess.run(["git", "clone", repo_url, str(local_path)], check=True)

    extractor = CommitFeatureExtractor(
        repo_path=str(local_path),
        since=since_date,
        label_window_days=label_window_days,
    )
    df = extractor.extract()

    output_path = Path(features_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if df.empty:
        print(f"Ingest complete. No commits found. Features saved to {output_path}")
        return

    total = len(df)
    positive = int(df["risky"].sum())
    print(
        f"\nClass Balance: {positive} risky ({positive / total:.2%}), "
        f"{total - positive} safe ({(total - positive) / total:.2%})"
    )
    print(f"Ingest complete. Features saved to {output_path}")
