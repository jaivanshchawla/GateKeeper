#!/usr/bin/env python3
"""
T.1: Recompute author_prior_commits using full repo history.
The SC path uses count_authors_before(repo, commit_date) which counts
ALL commits from the beginning. The CSV must match.
"""
import subprocess
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, ".")
from ml.m1_shared import normalize_author_id

REPOS = {
    "django": "repos/django",
    "react": "repos/react",
    "rust": "repos/rust",
    "kubernetes": "repos/kubernetes",
    "kafka": "repos/kafka",
}

df = pd.read_csv("data/commit_features.csv")
print(f"CSV: {len(df)} rows")

# For each repo, build a full history index of (timestamp, normalized_email)
# using a single git log call, then compute author_prior_commits efficiently
fixed = 0
for repo, rp in REPOS.items():
    print(f"\n  Building full history index for {repo}...")

    # Get ALL commits (no --since/--until) with timestamps and emails
    r = subprocess.run(
        ["git", "log", "--pretty=format:%ct|%aE", "--no-merges", "HEAD"],
        cwd=rp, capture_output=True, text=True, timeout=600,
    )

    # Parse into (timestamp, normalized_email) and sort
    entries = []
    for line in r.stdout.strip().split("\n"):
        if "|" in line:
            parts = line.split("|", 1)
            try:
                ts = int(parts[0])
                email = normalize_author_id(parts[1])
                entries.append((ts, email))
            except ValueError:
                pass

    entries.sort(key=lambda x: x[0])
    print(f"    {repo}: {len(entries)} total commits in history")

    # Build prefix sum: for each email, cumulative count at each timestamp
    # More efficient: build a dict of email -> sorted list of timestamps
    email_timestamps: dict[str, list[int]] = defaultdict(list)
    for ts, email in entries:
        email_timestamps[email].append(ts)

    # For each CSV row, binary-search the email's timestamp list
    import bisect

    repo_mask = df["source_repo"] == repo
    repo_indices = df[repo_mask].index

    repo_fixed = 0
    for idx in repo_indices:
        row = df.loc[idx]
        h = row["hash"]
        email = row["author"]

        # Get timestamp from git
        r2 = subprocess.run(
            ["git", "log", "-1", "--format=%ct", h],
            cwd=rp, capture_output=True, text=True, timeout=30,
        )
        if not r2.stdout.strip():
            continue
        ts = int(r2.stdout.strip())

        # Count entries by this email STRICTLY before this timestamp
        if email in email_timestamps:
            count = bisect.bisect_left(email_timestamps[email], ts)
        else:
            count = 0

        old = row["author_prior_commits"]
        if old != count:
            df.at[idx, "author_prior_commits"] = count
            repo_fixed += 1

    fixed += repo_fixed
    print(f"    {repo}: fixed {repo_fixed} rows")

print(f"\nTotal fixed: {fixed} rows")
df.to_csv("data/commit_features.csv", index=False)
print("Saved")
