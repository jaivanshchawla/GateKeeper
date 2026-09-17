#!/usr/bin/env python3
"""
W5.3: Post a GitHub commit status check with the Gatekeeper verdict.

Mapping:
  - Block rule fired → state=failure
  - Band=high → state=failure
  - Band=medium → state=neutral
  - Band=low → state=success

The status check appears in the PR's checks tab and can be made
required in branch protection settings.

Usage:
  python scripts/post_status_check.py \
    --commit-sha $COMMIT_SHA \
    --repo $REPO \
    --risk-label $RISK_LABEL \
    --risk-score $RISK_SCORE \
    [--token $GITHUB_TOKEN]

Or via GITHUB_OUTPUT env vars from score_pr.py:
  python scripts/post_status_check.py --from-env
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error


CONTEXT = "gatekeeper/risk-assessment"


def determine_state(risk_label: str, blocked: bool = False) -> str:
    """Map band + block status to GitHub check state."""
    if blocked:
        return "failure"
    if risk_label == "high":
        return "failure"
    if risk_label == "medium":
        return "neutral"
    return "success"


def post_status(
    token: str,
    owner: str,
    repo: str,
    sha: str,
    state: str,
    description: str,
    target_url: str = "",
    context: str = CONTEXT,
) -> dict:
    """Post a commit status via the GitHub REST API."""
    payload = {
        "state": state,
        "description": description[:140],  # GitHub limit
        "context": context,
    }
    if target_url:
        payload["target_url"] = target_url

    url = f"https://api.github.com/repos/{owner}/{repo}/statuses/{sha}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"ERROR posting status: {e.code} {body}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Post Gatekeeper status check")
    parser.add_argument("--commit-sha", help="Commit SHA")
    parser.add_argument("--repo", help="owner/repo format")
    parser.add_argument("--risk-label", help="low/medium/high")
    parser.add_argument("--risk-score", type=float, help="Risk score 0-1")
    parser.add_argument("--blocked", action="store_true", help="A block rule fired")
    parser.add_argument("--token", help="GitHub token (or GITHUB_TOKEN env)")
    parser.add_argument("--from-env", action="store_true",
                        help="Read from GITHUB_OUTPUT file")
    parser.add_argument("--dashboard-url", default="",
                        help="Dashboard link to include")
    args = parser.parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("ERROR: --token or GITHUB_TOKEN required", file=sys.stderr)
        sys.exit(1)

    if args.from_env:
        # Read from GITHUB_OUTPUT or a temp file
        output_file = os.environ.get("GITHUB_OUTPUT", "/tmp/gatekeeper_output")
        if os.path.exists(output_file):
            with open(output_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("risk_label="):
                        args.risk_label = line.split("=", 1)[1]
                    elif line.startswith("risk_score="):
                        args.risk_score = float(line.split("=", 1)[1])
                    elif line.startswith("commit_sha="):
                        args.commit_sha = line.split("=", 1)[1]
                    elif line.startswith("blocked="):
                        args.blocked = line.split("=", 1)[1].lower() == "true"

        if not args.risk_label:
            print("ERROR: --from-env but no risk_label found", file=sys.stderr)
            sys.exit(1)

    if not args.commit_sha:
        args.commit_sha = os.environ.get("GITHUB_SHA", "")
    if not args.commit_sha:
        print("ERROR: --commit-sha or GITHUB_SHA required", file=sys.stderr)
        sys.exit(1)

    # Parse repo from args or GITHUB_REPOSITORY
    repo_full = args.repo or os.environ.get("GITHUB_REPOSITORY", "")
    if not repo_full or "/" not in repo_full:
        print("ERROR: --repo or GITHUB_REPOSITORY required (owner/repo)", file=sys.stderr)
        sys.exit(1)
    owner, repo = repo_full.split("/", 1)

    state = determine_state(args.risk_label, args.blocked)
    score_pct = f"{args.risk_score * 100:.1f}%" if args.risk_score else ""

    display = {
        "low": "NOT FLAGGED",
        "medium": "ELEVATED",
        "high": "HIGH RISK",
    }.get(args.risk_label, args.risk_label.upper())

    description = f"{display}"
    if score_pct:
        description += f" ({score_pct})"

    print(f"State: {state}")
    print(f"Description: {description}")
    print(f"Commit: {args.commit_sha[:12]}")

    result = post_status(
        token=token,
        owner=owner,
        repo=repo,
        sha=args.commit_sha,
        state=state,
        description=description,
        target_url=args.dashboard_url,
    )

    print(f"Status ID: {result.get('id', 'unknown')}")
    print(f"URL: {result.get('target_url', 'N/A')}")
    return 0


if __name__ == "__main__":
    exit(main())
