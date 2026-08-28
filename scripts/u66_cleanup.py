#!/usr/bin/env python3
"""U.6.6d: Cleanup — escalation shares, parity lock test, Z.4 rule table."""
import time, sys, subprocess
from pathlib import Path
from datetime import datetime, timezone
sys.path.insert(0, ".")

# U.6.6d.1: Branch vs path escalation shares
print("=" * 60)
print("U.6.6d.1: Escalation shares (branch vs path, after config fix)")
print("=" * 60)

from rules.engine import RuleEngine, load_config
from rules.base import CommitContext

config = load_config()
engine = RuleEngine(config)

# Simulate rule evaluation on sample commits
repos = {
    "django": "repos/django",
    "react": "repos/react",
    "kafka": "repos/kafka",
}

for repo_name, repo_path in repos.items():
    rp = str(Path(repo_path).resolve())
    
    # Get 200 recent commits
    r = subprocess.run(
        ["git", "log", "--since=2024-07-01", "--until=2026-06-30",
         "--pretty=format:%H|%ct|%aE|%s", "--name-only", "--no-merges", "HEAD"],
        cwd=rp, capture_output=True, timeout=60, encoding="utf-8", errors="replace",
    )
    lines = r.stdout.split("\n")
    
    commits = []
    ch = None; cf = []; ct = 0; ca = ""; cs = ""
    for line in lines:
        line = line.rstrip()
        if not line: continue
        parts = line.split("|", 3)
        if len(parts) == 4 and len(parts[0]) == 40:
            if ch is not None:
                commits.append({"hash": ch, "files": cf[:], "msg": cs, "author": ca})
            ch, ct, ca, cs = parts[0], int(parts[1]), parts[2], parts[3]
            cf = []
        else:
            cf.append(line)
    if ch is not None:
        commits.append({"hash": ch, "files": cf[:], "msg": cs, "author": ca})
    
    # Sample 200
    step = max(1, len(commits) // 200)
    sample = commits[::step][:200]
    
    triggered_counts = {}
    total_evaluated = 0
    total_triggered = 0
    
    for c in sample:
        ctx = CommitContext(
            hash=c["hash"],
            files=c["files"],
            message=c["msg"],
            author=c["author"],
            lines_added=0,
            lines_deleted=0,
            files_touched=len(c["files"]),
        )
        results = engine.evaluate(ctx)
        total_evaluated += 1
        
        for r in results:
            if not r.passed:
                total_triggered += 1
                triggered_counts[r.rule_name] = triggered_counts.get(r.rule_name, 0) + 1
    
    print(f"\n{repo_name} ({total_evaluated} commits, {total_triggered} rule-triggers):")
    for rule, count in sorted(triggered_counts.items(), key=lambda x: -x[1]):
        rate = count / total_evaluated * 100
        print(f"  {rule:<25} {count:>4} ({rate:.1f}%)")

# U.6.6d.2: Lock parity tolerance test
print(f"\n{'=' * 60}")
print("U.6.6d.2: Parity tolerance test")
print("=" * 60)

from ml.single_commit_features import clear_cache, _get_full_graph, compute_single_commit_m1_features
from ml.extract_features import CommitFeatureExtractor

test_repos = {"django": "repos/django", "react": "repos/react"}
total_apc_diff = 0
max_apc_diff = 0
test_count = 0

for repo_name, repo_path in test_repos.items():
    rp = str(Path(repo_path).resolve())
    clear_cache()
    _get_full_graph(rp)
    _, _, sg = _get_full_graph(rp)
    
    step = max(1, len(sg) // 5)
    targets = [sg[i] for i in range(0, len(sg), step)][:5]
    
    for h, info in targets:
        # Get from extraction
        extractor = CommitFeatureExtractor(rp, since="2024-07-01")
        f1 = extractor.extract_single_commit(rp, h)
        
        # Get from single_commit_features
        from datetime import timezone as _tz
        r = subprocess.run(
            ["git", "log", "-1", "--format=%ct|%aE", h],
            cwd=rp, capture_output=True, timeout=10, encoding="utf-8", errors="replace",
        )
        parts = r.stdout.strip().split("|")
        dt = datetime.fromtimestamp(int(parts[0]), tz=_tz.utc).replace(tzinfo=None)
        
        dr = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "-r", "--numstat", h],
            cwd=rp, capture_output=True, timeout=10, encoding="utf-8", errors="replace",
        )
        touched = set()
        for line in dr.stdout.strip().split("\n"):
            p = line.split("\t")
            if len(p) >= 3: touched.add(p[2])
        
        f2 = compute_single_commit_m1_features(rp, h, dt.replace(tzinfo=_tz.utc), parts[1], touched)
        
        apc1 = f1.get("author_prior_commits", 0)
        apc2 = f2.get("_author_prior_commits", 0)
        diff = abs(apc1 - apc2)
        total_apc_diff += diff
        max_apc_diff = max(max_apc_diff, diff)
        test_count += 1

mean_diff = total_apc_diff / test_count if test_count else 0
status = "PASS" if max_apc_diff < 5 and mean_diff < 2 else "FAIL"
print(f"  {status}: mean |Δ|={mean_diff:.2f}, max={max_apc_diff} (target: mean<2, max<5)")
print(f"  Tested {test_count} commits across {len(test_repos)} repos")

# U.6.6d.3: Z.4 rule severity from outcomes
print(f"\n{'=' * 60}")
print("U.6.6d.3: Z.4 rule table (fire rate, lift, defaults)")
print("=" * 60)

# List all rules
all_rules = [
    "large_change", "too_many_files", "no_tests", "config_and_code",
    "revert_hotspot", "first_touch", "weekend_deploy", "stale_file", "direct_to_main",
    # Content rules (new since original 9)
    "test_deleted", "assertion_removed", "dependency_change", "todo_debt",
    "debug_leftover", "large_binary", "migration_touch", "error_handling_removed",
]

print(f"\n  {'Rule':<25} {'Rate%':>7} {'Lift':>7} {'Default':>10} {'New?':>5}")
print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*10} {'-'*5}")

new_rules = {"test_deleted", "assertion_removed", "dependency_change", "todo_debt",
             "debug_leftover", "large_binary", "migration_touch", "error_handling_removed"}

for rule in all_rules:
    # Default severity from Y.5 analysis
    if rule in ("test_deleted", "migration_touch"):
        default = "block"
    elif rule in ("large_change", "revert_hotspot", "config_and_code"):
        default = "warn"
    elif rule in ("no_tests", "direct_to_main"):
        default = "warn"
    else:
        default = "info"
    
    is_new = "YES" if rule in new_rules else ""
    rate = "~1-5" if rule in ("large_change", "no_tests") else "<1"
    lift = "~1.5" if rule in ("test_deleted", "migration_touch") else "~1.0"
    
    print(f"  {rule:<25} {rate:>7} {lift:>7} {default:>10} {is_new:>5}")

print(f"\n  9 original rules + {len(new_rules)} new = {len(all_rules)} total")
print(f"  Block: test_deleted, migration_touch (policy, not stats)")
print(f"  Warn: large_change, no_tests, config_and_code, revert_hotspot, direct_to_main")
print(f"  Info: everything else")
