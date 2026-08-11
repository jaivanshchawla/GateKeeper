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
from typing import List, Dict

import pandas as pd
import yaml
from pydriller import Repository


class CommitFeatureExtractor:
    """Extracts features from git commit history and labels risky commits."""

    def __init__(self, repo_path: str, since: str, label_window_days: int = 7):
        """
        Initialize the feature extractor.

        Args:
            repo_path: Path to the cloned git repository
            since: Date string (YYYY-MM-DD) to start mining commits from
            label_window_days: Number of days after commit to check for re-risks
        """
        self.repo_path = repo_path
        self.since = since
        self.label_window_days = label_window_days
        self.author_prior_counts = defaultdict(int)
        
        # Store file touches for labeling (file_path -> [(commit_hash, commit_date)])
        self.file_touches = defaultdict(list)
        # Store commit info for labeling (hash -> {date, files})
        self.commit_info = {}

    def _extract_features_from_commit(self, commit) -> Dict:
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

    def _collect_all_commits(self) -> List[Dict]:
        """Collect all commits with features."""
        print(f"Mining commits from {self.repo_path} since {self.since}...")
        
        features = []
        # Convert string date to datetime object for PyDriller
        since_date = datetime.strptime(self.since, "%Y-%m-%d")
        repository = Repository(self.repo_path, since=since_date)
        
        for commit in repository.traverse_commits():
            feature_dict = self._extract_features_from_commit(commit)
            features.append(feature_dict)
        
        print(f"Collected {len(features)} commits.")
        return features

    def _label_commits(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Label commits as risky (1) or safe (0)."""
        print("Labeling commits...")
        
        # Initialize label column
        features_df["risky"] = 0
        
        # Label commits based on two criteria:
        # 1. If the commit message contains "revert" referencing this commit
        # 2. If any of its files were touched again within the label window
        
        # First, check for revert references
        revert_pattern = "revert"
        for idx, row in features_df.iterrows():
            commit_hash = row["hash"]
            commit_msg = row["commit_msg"].lower()
            
            # Check if this commit is a revert
            if revert_pattern in commit_msg:
                # Mark as risky if it's a revert commit
                features_df.at[idx, "risky"] = 1
        
        # Second, check if any files were touched again within the window
        for file_path, touches in self.file_touches.items():
            # Sort touches by date
            touches.sort(key=lambda x: x[1])
            
            # For each touch, check if any subsequent touch is within the window
            for i, (hash_i, date_i) in enumerate(touches):
                for j in range(i + 1, len(touches)):
                    hash_j, date_j = touches[j]
                    time_diff = (date_j - date_i).days
                    
                    if time_diff <= self.label_window_days:
                        # Mark commit i as risky
                        idx = features_df[features_df["hash"] == hash_i].index
                        if len(idx) > 0:
                            features_df.at[idx[0], "risky"] = 1
                    else:
                        # No need to check further for this commit
                        break
        
        return features_df

    def extract_and_save(self, output_path: str) -> pd.DataFrame:
        """
        Extract features, label commits, and save to CSV.
        
        Args:
            output_path: Path to save the features CSV
            
        Returns:
            DataFrame with features and labels
        """
        # Collect all commits
        features = self._collect_all_commits()
        
        # Create DataFrame
        df = pd.DataFrame(features)
        
        # Label commits (needs commit_msg for revert detection)
        df = self._label_commits(df)
        
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
        label_window_days=label_window_days
    )
    
    extractor.extract_and_save(args.output)


if __name__ == "__main__":
    main()