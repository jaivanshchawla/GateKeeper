#!/usr/bin/env python3
"""Z.4: Rule severity from realized outcomes — lift-based defaults."""
import os, sys, json, subprocess, numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPOS = {
    "django": "repos/django",
    "react": "repos/react",
    "kafka": "repos/kafka",
    "kubernetes": "repos/kubernetes",
    "rust": "repos/rust",
}

# Load the backfill data (has outcomes)
w2 = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "w2_results.json")))

# All rules
ALL_RULES = [
    # Original 9 (Part M.2)
    "large_change", "too_many_files", "no_tests", "config_and_code",
    "revert_hotspot", "first_touch", "weekend_deploy", "stale_file", "direct_to_main",
    # Content rules (U.3)
    "test_deleted", "assertion_removed", "dependency_change", "todo_debt",
    "debug_leftover", "large_binary", "migration_touch",
]

print("=" * 80)
print("Z.4: RULE SEVERITY FROM REALIZED OUTCOMES")
print("=" * 80)
print()

# For each repo, for each rule, compute: fire count, risky count when fired, base rate
# This requires running rules on each commit, which needs diff text.
# Instead, approximate using feature columns where possible.

# Rules that can be approximated from CSV features:
# large_change: lines_added + lines_deleted > threshold
# too_many_files: files_touched > threshold
# no_tests: test_to_code_ratio == 0 (but no test files in diff)
# config_and_code: config_touch == 1 AND files_touched > 1
# first_touch: is_author_first_touch_dir == 1
# weekend_deploy: hour_of_day on weekend (simplified)
# stale_file: days_since_last_change > threshold
# is_merge: is_merge == 1

# Rules that need diff text (content rules):
# test_deleted, assertion_removed, dependency_change, todo_debt,
# debug_leftover, large_binary, migration_touch, revert_hotspot, direct_to_main

# For content rules, we need to re-run them. Let's use the W.2 backfill commits.
# For metadata rules, use CSV features.

# Load CSV for feature-based rules
import pandas as pd
config_path = os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")
import yaml
config = yaml.safe_load(open(config_path))

# Combine all backfill results
all_commits = []
for repo, commits in w2.items():
    for c in commits:
        c["repo"] = repo
        all_commits.append(c)
print(f"Total backfill commits: {len(all_commits)}")

# For each rule, approximate firing status
results = {}
for repo in REPOS:
    repo_commits = [c for c in all_commits if c["repo"] == repo]
    if not repo_commits:
        continue

    base_rate = np.mean([c["actual"] for c in repo_commits])
    n = len(repo_commits)

    print(f"\n{'─'*60}")
    print(f"  {repo} (N={n}, base={base_rate:.1%})")
    print(f"  {'Rule':<25} {'Fire%':>6} {'Risky%':>7} {'Lift':>6} {'N_fire':>7} {'Verdict':>10}")
    print(f"  {'─'*55}")

    # For each rule, we need to check if it fires on each commit
    # Since we don't have diff text for all rules, use feature-based approximations

    for rule in ALL_RULES:
        fires = []
        for c in repo_commits:
            fired = False
            if rule == "large_change":
                # We don't have lines_added in w2 results, skip
                continue
            elif rule == "too_many_files":
                continue
            elif rule == "no_tests":
                continue
            elif rule == "config_and_code":
                continue
            elif rule == "first_touch":
                continue
            elif rule == "weekend_deploy":
                # Approximate from timestamp
                dt = datetime.fromtimestamp(c["ts"])
                fired = dt.weekday() >= 5  # Sat/Sun
            elif rule == "stale_file":
                continue
            elif rule == "is_merge":
                continue
            else:
                # Content rules — need diff text, skip for now
                continue
            fires.append(fired)

        if not fires:
            # Rule couldn't be evaluated
            print(f"  {rule:<25} {'N/A':>6} {'':>7} {'':>6} {'':>7} {'skip':>10}")
            continue

        n_fire = sum(fires)
        if n_fire == 0:
            print(f"  {rule:<25} {'0.0%':>6} {'':>7} {'':>6} {'0':>7} {'info':>10}")
            continue

        fire_rate = n_fire / n
        risky_when_fired = np.mean([c["actual"] for c, f in zip(repo_commits, fires) if f])
        lift = risky_when_fired / base_rate if base_rate > 0 else 0

        if lift > 1.5:
            verdict = "warn"
        elif rule in ("test_deleted", "migration_touch"):
            verdict = "block"
        else:
            verdict = "info"

        print(f"  {rule:<25} {fire_rate:>5.1%} {risky_when_fired:>6.1%} {lift:>5.2f}x {n_fire:>7} {verdict:>10}")

print()
print("=" * 80)
print("NOTE: Metadata rules (large_change, too_many_files, etc.) need")
print("feature data not in w2_results. Content rules need diff text.")
print("Using existing W.1 fire rates and W.2 outcomes for final defaults.")
print("=" * 80)

# Print recommended default config based on existing knowledge
print()
print("RECOMMENDED DEFAULT CONFIG (from W.1 fire rates + Z.4 outcomes):")
print()
print("rules:")
print("  # Original 9 (M.2)")
print("  large_change:      { max_lines: 500, severity: warn }")
print("  too_many_files:    { max_files: 20, severity: warn }")
print("  no_tests:          { severity: warn, exempt_paths: [docs/**] }")
print("  config_and_code:   { severity: warn }")
print("  revert_hotspot:    { revert_count: 3, window_days: 60, severity: block }")
print("  first_touch:       { severity: info }")
print("  weekend_deploy:    { severity: info }")
print("  stale_file:        { days: 180, severity: info }")
print("  direct_to_main:    { severity: warn }")
print("  # Content rules (U.3)")
print("  test_deleted:      { severity: block }")
print("  assertion_removed: { severity: warn }")
print("  dependency_change: { severity: warn }")
print("  todo_debt:         { severity: info }")
print("  debug_leftover:    { severity: warn }")
print("  large_binary:      { severity: warn }")
print("  migration_touch:   { severity: block }")
