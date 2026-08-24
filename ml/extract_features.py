#!/usr/bin/env python3
"""
Commit feature extractor for Gatekeeper MLOps quality gate.
Extracts per-commit metrics using PyDriller and labels risky commits.
"""

import argparse
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from pydriller import Repository


class CommitFeatureExtractor:
    """Extracts features from git commit history and labels risky commits."""

    def __init__(self, repo_path: str, since: str, label_window_days: int = 7, max_commits: int = 0, label_buffer: int = 500):
        """
        Initialize the feature extractor.

        Args:
            repo_path: Path to the cloned git repository
            since: Date string (YYYY-MM-DD) to start mining commits from
            label_window_days: Number of days after commit to check for re-risks
            max_commits: Stop after this many commits (0 = unlimited)
            label_buffer: Extra commits to mine beyond max_commits for 7-day
                forward-look labeling. Only the first max_commits rows become
                training data; the buffer provides label context. Ignored when
                max_commits=0.
        """
        self.repo_path = repo_path
        self.since = since
        self.label_window_days = label_window_days
        self.max_commits = max_commits
        self.label_buffer = label_buffer
        self.author_prior_counts = defaultdict(int)
        
        # Store file touches for labeling (file_path -> [(commit_hash, commit_date)])
        self.file_touches = defaultdict(list)
        # Store commit info for labeling (hash -> {date, files})
        self.commit_info = {}

    def _extract_features_from_commit(self, commit) -> dict:
        """Extract features from a single commit object."""
        # Basic commit info
        lines_added = commit.insertions
        lines_deleted = commit.deletions
        files_touched = commit.files
        
        # Collect file paths for this commit
        touched_files = set()
        directories = set()
        
        for modified_file in commit.modified_files:
            file_path = modified_file.new_path or modified_file.old_path
            if file_path:
                touched_files.add(file_path)
                dir_path = os.path.dirname(file_path)
                if dir_path:
                    directories.add(dir_path)
                
                # Store file touch for labeling
                self.file_touches[file_path].append((commit.hash, commit.author_date))
        
        num_directories = len(directories)
        
        # Store commit info for labeling
        self.commit_info[commit.hash] = {
            "date": commit.author_date,
            "files": touched_files,
            "msg": commit.msg or ""
        }

        # Author's total prior commit count
        author_name = commit.author.name
        author_prior_commits = self.author_prior_counts[author_name]

        # Temporal features
        commit_date = commit.author_date
        hour_of_day = commit_date.hour
        day_of_week = commit_date.weekday()  # 0=Monday, 6=Sunday

        # Commit message features
        commit_msg = commit.msg or ""
        commit_msg_length = len(commit_msg)
        is_fix_bug_revert = any(
            keyword in commit_msg.lower()
            for keyword in ["fix", "bug", "revert"]
        )

        # After extracting features, increment author's prior commit count
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

    def _collect_all_commits(self) -> tuple[list[dict], int]:
        """Collect all commits with features.

        Stops when EITHER the since-date boundary OR the effective cap
        (max_commits + label_buffer) is hit, whichever comes first.

        Returns:
            Tuple of (all_features_list, training_count) where training_count
            is the number of rows that should become training data (the rest
            are buffer commits used only for label-checking).
        """
        effective_cap = 0
        if self.max_commits:
            effective_cap = self.max_commits + self.label_buffer
        cap_str = f", effective cap {effective_cap}" if effective_cap else ""
        print(f"Mining commits from {self.repo_path} since {self.since}{cap_str}...")
        
        features = []
        # Convert string date to datetime object for PyDriller
        since_date = datetime.strptime(self.since, "%Y-%m-%d")  # noqa: DTZ007 — PyDriller requires naive datetime
        repository = Repository(self.repo_path, since=since_date)
        
        for commit in repository.traverse_commits():
            feature_dict = self._extract_features_from_commit(commit)
            features.append(feature_dict)
            
            if effective_cap and len(features) >= effective_cap:
                print(f"Reached effective cap ({effective_cap} = {self.max_commits} + {self.label_buffer} buffer), stopping.")
                break
        
        training_count = min(len(features), self.max_commits) if self.max_commits else len(features)
        print(f"Collected {len(features)} commits ({training_count} training + {len(features) - training_count} buffer for labeling).")
        return features, training_count

    def _label_commits(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Label commits as risky (1) or safe (0)."""
        print("Labeling commits...")
        
        # Build hash→row-index lookup once (avoids O(n) DataFrame scan per commit)
        _hash_to_idx = {h: i for i, h in enumerate(features_df["hash"])}
        risky_hashes: set[str] = set()
        
        # Criterion 1: commit message contains "revert"
        for idx, row in features_df.iterrows():
            if "revert" in row["commit_msg"].lower():
                risky_hashes.add(row["hash"])
        
        # Criterion 2: any file touched again within label_window_days
        for touches in self.file_touches.values():
            if not touches:
                continue
            touches.sort(key=lambda x: x[1])
            
            for i, (hash_i, date_i) in enumerate(touches):
                if hash_i in risky_hashes:
                    continue  # already labeled
                for j in range(i + 1, len(touches)):
                    _hash_j, date_j = touches[j]
                    time_diff = (date_j - date_i).days
                    if time_diff <= self.label_window_days:
                        risky_hashes.add(hash_i)
                        break
                    else:
                        break
        
        # Apply labels in one vectorized pass
        features_df["risky"] = features_df["hash"].apply(
            lambda h: 1 if h in risky_hashes else 0
        )
        
        return features_df

    def extract_single_commit(self, repo_path: str, commit_hash: str) -> dict:
        """
        Extract features for a single specified commit.

        Args:
            repo_path: Path to the cloned git repository
            commit_hash: The hash of the commit to extract features for

        Returns:
            Dictionary with feature values for the commit
        """
        # Use PyDriller to get the specific commit
        repository = Repository(repo_path, single=commit_hash)

        for commit in repository.traverse_commits():
            # Extract features using the existing method
            features = self._extract_features_from_commit(commit)
            # Remove commit_msg as it's not used in training
            features.pop('commit_msg', None)
            return features

        raise ValueError(f"Commit {commit_hash} not found in repository")

    def extract_and_save(self, output_path: str) -> pd.DataFrame:
        """
        Extract features, label commits, and save to CSV.

        When a label_buffer is used, only the first max_commits rows
        become training data — the buffer commits provide label context
        but are discarded before saving.

        Args:
            output_path: Path to save the features CSV

        Returns:
            DataFrame with features and labels
        """
        # Collect all commits (including buffer for labeling)
        features, training_count = self._collect_all_commits()
        
        # Create DataFrame from ALL commits (needed for correct labeling)
        df = pd.DataFrame(features)
        
        # Label commits using the full dataset (including buffer)
        df = self._label_commits(df)
        
        # Trim to training rows only (discard buffer commits)
        if self.max_commits and len(df) > training_count:
            buffer_size = len(df) - training_count
            print(f"Discarding {buffer_size} buffer commits (used only for labeling).")
            df = df.iloc[:training_count].copy()
        
        # Drop the commit_msg column (we don't need it for training)
        if "commit_msg" in df.columns:
            df = df.drop(columns=["commit_msg"])
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")
        
        # Print class balance summary
        total = len(df)
        positive = df["risky"].sum()
        negative = total - positive
        pos_pct = (positive / total) * 100
        neg_pct = (negative / total) * 100
        
        print("\nClass Balance Summary:")
        print(f"Total commits: {total}")
        print(f"Risky (1): {positive} ({pos_pct:.2f}%)")
        print(f"Safe (0): {negative} ({neg_pct:.2f}%)")
        
        return df


def main():
    parser = argparse.ArgumentParser(description="Extract commit features for Gatekeeper")
    parser.add_argument(
        "--repo-path",
        required=True,
        help="Path to the cloned git repository"
    )
    parser.add_argument(
        "--since",
        required=True,
        help="Start date for commit mining (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output",
        default="data/commit_features.csv",
        help="Output CSV path (default: data/commit_features.csv)"
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=0,
        help="Stop after this many commits (0 = unlimited)"
    )
    parser.add_argument(
        "--label-buffer",
        type=int,
        default=500,
        help="Extra commits to mine beyond max-commits for 7-day forward-look labeling (default: 500)"
    )
    parser.add_argument(
        "--config",
        default="ml/config.yaml",
        help="Path to config file (default: ml/config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Load config to get label window
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        label_window_days = config.get("label_window_days", 7)
    else:
        label_window_days = 7
        print(f"Config file not found at {config_path}, using default label_window_days=7")
    
    # Create extractor and run
    extractor = CommitFeatureExtractor(
        repo_path=args.repo_path,
        since=args.since,
        label_window_days=label_window_days,
        max_commits=args.max_commits,
        label_buffer=args.label_buffer,
    )
    
    extractor.extract_and_save(args.output)


if __name__ == "__main__":
    main()