#!/usr/bin/env python3
"""
T.1e: Update the CSV's author column from PyDriller display names to
normalized git emails, and recompute author_prior_commits using
count_authors_before (which now uses %aE).
"""
import subprocess
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from ml.m1_shared import count_authors_before, normalize_author_id

REPOS_DIR = Path("repos")
REPO_NAMES = ["django", "react", "rust", "kubernetes", "kafka"]

# 1. Build a hash→email mapping for all 5 repos
print("Building hash→email mapping from git log...")
hash_to_email: dict[str, str] = {}
for repo in REPO_NAMES:
    rp = str(REPOS_DIR / repo)
    r = subprocess.run(
        ["git", "log", "--since=2024-07-01", "--until=2026-07-07",
         "--pretty=format:%H|%aE", "--no-merges", "HEAD"],
        cwd=rp, capture_output=True, text=True, timeout=600,
    )
    for line in r.stdout.strip().split("\n"):
        if "|" in line:
            h, email = line.split("|", 1)
            hash_to_email[h.strip()] = email.strip()
    # Also add merge commits
    r2 = subprocess.run(
        ["git", "log", "--since=2024-07-01", "--until=2026-07-07",
         "--pretty=format:%H|%aE", "--merges", "HEAD"],
        cwd=rp, capture_output=True, text=True, timeout=600,
    )
    for line in r2.stdout.strip().split("\n"):
        if "|" in line:
            h, email = line.split("|", 1)
            hash_to_email[h.strip()] = email.strip()
    print(f"  {repo}: {len([v for v in hash_to_email.values()])} total hashes mapped")

print(f"Total hash→email mappings: {len(hash_to_email)}")

# 2. Load CSV
df = pd.read_csv("data/commit_features.csv")
print(f"\nCSV: {len(df)} rows, author column sample: {df['author'].head(3).tolist()}")

# 3. Map hash→normalized email
csv_hashes = set(df["hash"].values)
mapped = 0
unmapped = 0
new_emails = []
for _, row in df.iterrows():
    h = row["hash"]
    raw_email = hash_to_email.get(h, "")
    if raw_email:
        new_emails.append(normalize_author_id(raw_email))
        mapped += 1
    else:
        # Fallback: normalize the existing author name
        new_emails.append(normalize_author_id(str(row["author"])))
        unmapped += 1

print(f"Mapped: {mapped}, Unmapped: {unmapped}")
df["author"] = new_emails
print(f"New author column sample: {df['author'].head(3).tolist()}")
print(f"Unique normalized emails: {df['author'].nunique()}")

# 4. Recompute author_prior_commits using count_authors_before
print("\nRecomputing author_prior_commits from git log...")
for repo in REPO_NAMES:
    rp = str(REPOS_DIR / repo)
    repo_mask = df["source_repo"] == repo
    repo_df = df[repo_mask]
    print(f"  {repo}: {len(repo_df)} rows")

    # Pre-seed: count all commits by normalized email before WINDOW_START
    base_counts = count_authors_before(rp, "2024-07-01T00:00:00")
    print(f"    Base author count: {len(base_counts)} authors")

    # Sort by committer_date to process chronologically
    repo_df_sorted = repo_df.sort_values("committer_date")
    running_counts = dict(base_counts)

    for idx, row in repo_df_sorted.iterrows():
        email = row["author"]
        # Count commits by this email BEFORE this commit's date
        df.at[idx, "author_prior_commits"] = running_counts.get(email, 0)
        # Increment for the next commit
        running_counts[email] = running_counts.get(email, 0) + 1

print(f"\nAuthor prior commits sample:\n{df[['hash', 'source_repo', 'author', 'author_prior_commits']].head(10)}")

# 5. Save
df.to_csv("data/commit_features.csv", index=False)
print(f"\nSaved {df.shape} to data/commit_features.csv")

# 6. Verify no zeros (except genuine first-timers)
zero_count = (df["author_prior_commits"] == 0).sum()
print(f"Rows with author_prior_commits=0: {zero_count} ({100*zero_count/len(df):.1f}%)")
