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
    repo_urls: str = "",
) -> None:
    """
    Clone/pull repos and extract commit features.

    Supports both single-repo and multi-repo mining. When repo_urls is
    provided (comma-separated), it overrides repo_url and mines each repo
    individually, combining all results with a source_repo column.

    Args:
        repo_url: Single git repository URL or local path (fallback).
        since_date: Date string (YYYY-MM-DD) to start mining from.
        features_path: KFP output path for the feature CSV.
        label_window_days: Days after commit to check for re-touches.
        cached_csv_path: If non-empty and file exists, copy this CSV
            to features_path instead of re-mining.
        repo_urls: Comma-separated list of repo URLs/paths for multi-repo.
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

    def _clone_or_open(url_or_path):
        """Clone a repo or open an existing local path. Returns local_path."""
        candidate = Path(url_or_path).expanduser()
        if candidate.exists():
            local = candidate.resolve()
            print(f"Using local repository at {local}")
            return local
        repo_name = url_or_path.rstrip("/").split("/")[-1].replace(".git", "")
        local = Path(tempfile.gettempdir()) / repo_name
        if local.exists():
            print(f"Pulling existing repo at {local}...")
            subprocess.run(["git", "-C", str(local), "pull"], check=True)
        else:
            print(f"Cloning {url_or_path} to {local}...")
            subprocess.run(["git", "clone", url_or_path, str(local)], check=True)
        return local

    def _mine_single_repo(local_path, since, label_window_days, repo_name=""):
        """Mine features from a single local repository."""

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
                    kw in commit_msg.lower()
                    for kw in ["fix", "bug", "revert"]
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
                # Optimized labeling using file-touch index lookup
                hash_index = {h: i for i, h in enumerate(df["hash"])}
                for touches in self.file_touches.values():
                    touches.sort(key=lambda t: t[1])
                    for i, (hash_i, date_i) in enumerate(touches):
                        for _, date_j in touches[i + 1 :]:
                            if (date_j - date_i).days <= self.label_window_days:
                                if hash_i in hash_index:
                                    df.iat[hash_index[hash_i],
                                        df.columns.get_loc("risky")] = 1
                            else:
                                break
                return df.drop(columns=["commit_msg"], errors="ignore")

        extractor = CommitFeatureExtractor(
            repo_path=str(local_path),
            since=since,
            label_window_days=label_window_days,
        )
        df = extractor.extract()
        if not df.empty and repo_name:
            df["source_repo"] = repo_name
        return df

    # Determine which repos to mine
    repo_list = []
    if repo_urls:
        repo_list = [r.strip() for r in repo_urls.split(",") if r.strip()]
    else:
        repo_list = [repo_url]

    all_dfs = []
    for url in repo_list:
        repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
        local_path = _clone_or_open(url)
        df = _mine_single_repo(local_path, since_date, label_window_days, repo_name)
        if not df.empty:
            all_dfs.append(df)
            pos = int(df["risky"].sum())
            print(
                f"  {repo_name}: {len(df)} rows, "
                f"{pos} risky ({pos / len(df):.1%})"
            )

    if not all_dfs:
        print("No commits found in any repo.")
        output_path = Path(features_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(output_path, index=False)
        return

    df = pd.concat(all_dfs, ignore_index=True)

    output_path = Path(features_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    total = len(df)
    positive = int(df["risky"].sum())
    print(
        f"\nClass Balance: {positive} risky ({positive / total:.2%}), "
        f"{total - positive} safe ({(total - positive) / total:.2%})"
    )
    print(f"Ingest complete. {len(repo_list)} repo(s), {total} total commits.")
    print(f"Features saved to {output_path}")
