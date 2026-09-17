#!/usr/bin/env python3
"""
W5.4: SARIF 2.1.0 exporter for Gatekeeper rule results.

Maps each rule result to a SARIF result with:
- ruleId: the Gatekeeper rule name
- level: error/warning/note based on severity and pass/fail
- message: human-readable description
- location: file path and line number where known

Exports to SARIF format consumable by:
- GitHub Code Scanning (Security tab)
- VS Code SARIF extension
- SonarQube
- Any SARIF-compatible tool
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rules.base import Severity


# Map Gatekeeper severity → SARIF level
_SEVERITY_TO_LEVEL = {
    Severity.BLOCK: "error",
    Severity.WARN: "warning",
    Severity.INFO: "note",
}


def _rule_metadata() -> dict[str, dict[str, str]]:
    """Return SARIF rule definitions for all known Gatekeeper rules."""
    rules = {}
    rule_defs = [
        ("large_change", "Large Change", "Commit modifies too many lines", "warning"),
        ("too_many_files", "Too Many Files", "Commit touches too many files", "warning"),
        ("no_tests", "No Tests", "Code changes without test changes", "warning"),
        ("config_and_code", "Config and Code", "Config and code changed together", "warning"),
        ("revert_hotspot", "Revert Hotspot", "File has been reverted multiple times", "error"),
        ("first_touch", "First Touch", "Author has never touched this directory", "note"),
        ("weekend_deploy", "Weekend Deploy", "Commit made on a weekend", "note"),
        ("stale_file", "Stale File", "File hasn't changed in a long time", "note"),
        ("direct_to_main", "Direct to Main", "Direct push to main branch", "warning"),
        ("test_deleted", "Test Deleted", "Test files or assertions removed", "error"),
        ("assertion_removed", "Assertions Removed", "Net reduction in assert/expect count", "warning"),
        ("dependency_change", "Dependency Change", "Lockfile or manifest modified", "warning"),
        ("todo_debt", "TODO Debt", "Added TODO/FIXME/HACK/XXX comments", "note"),
        ("debug_leftover", "Debug Leftover", "Debug statements added to non-test code", "warning"),
        ("large_binary", "Large Binary", "Binary or generated file exceeds size threshold", "warning"),
        ("migration_touch", "Migration Touch", "Database migration files modified", "error"),
        ("error_handling_removed", "Error Handling Removed", "Net reduction in try/catch/except blocks", "warning"),
        ("complexity_delta", "Complexity Delta", "Cyclomatic complexity increase", "warning"),
    ]
    for rule_id, name, desc, default_level in rule_defs:
        rules[rule_id] = {
            "id": rule_id,
            "shortDescription": {"text": name},
            "fullDescription": {"text": desc},
            "defaultConfiguration": {"level": default_level},
        }
    return rules


def export_sarif(
    results: list[dict[str, Any]],
    repo_path: str = "",
    commit_sha: str = "",
    tool_version: str = "1.0.0",
) -> dict:
    """Export Gatekeeper rule results to SARIF 2.1.0 format.

    Args:
        results: list of dicts with keys: rule, severity, passed, message,
                 evidence (optional), file (optional), line (optional)
        repo_path: path to the repo root (for artifactLocation)
        commit_sha: commit hash (for git revision)
        tool_version: Gatekeeper version string

    Returns:
        SARIF 2.1.0 dict ready for JSON serialization
    """
    tool = {
        "driver": {
            "name": "Gatekeeper",
            "semanticVersion": tool_version,
            "version": tool_version,
            "rules": list(_rule_metadata().values()),
        }
    }

    sarif_results = []
    for r in results:
        rule_name = r.get("rule", "unknown")
        passed = r.get("passed", True)
        severity = r.get("severity", "info")
        message = r.get("message", "")
        evidence = r.get("evidence", "")
        file_path = r.get("file", "")
        line = r.get("line")

        # Skip passed rules — SARIF only reports findings
        if passed:
            continue

        # Determine SARIF level from Gatekeeper severity
        level = _SEVERITY_TO_LEVEL.get(severity, "warning")

        # Build message
        full_message = message
        if evidence:
            full_message = f"{message}\n\nEvidence: {evidence}"

        sarif_result: dict[str, Any] = {
            "ruleId": rule_name,
            "level": level,
            "message": {"text": full_message},
        }

        # Add location if file is known
        if file_path:
            location: dict[str, Any] = {
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": file_path,
                        "uriBaseId": "%SRCROOT%",
                    },
                }
            }
            if line is not None:
                location["physicalLocation"]["region"] = {"startLine": int(line)}
            sarif_result["locations"] = [location]

        sarif_results.append(sarif_result)

    # Build the SARIF document
    run: dict[str, Any] = {
        "tool": tool,
        "results": sarif_results,
    }

    if commit_sha:
        run["versionControlProvenance"] = [
            {
                "repositoryUri": f"https://github.com/{repo_path}" if "/" in repo_path else "",
                "revisionId": commit_sha,
            }
        ]

    sarif_doc = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [run],
    }

    return sarif_doc


def validate_sarif(sarif_doc: dict) -> tuple[bool, list[str]]:
    """Basic structural validation of a SARIF document.

    Returns (is_valid, list of error messages).
    """
    errors = []

    # Check top-level fields
    if sarif_doc.get("version") != "2.1.0":
        errors.append(f"version must be '2.1.0', got '{sarif_doc.get('version')}'")

    runs = sarif_doc.get("runs", [])
    if not runs:
        errors.append("runs array must not be empty")

    for i, run in enumerate(runs):
        tool = run.get("tool", {})
        driver = tool.get("driver", {})
        if not driver.get("name"):
            errors.append(f"run[{i}].tool.driver.name is required")
        if not driver.get("rules"):
            errors.append(f"run[{i}].tool.driver.rules must not be empty")

        for j, result in enumerate(run.get("results", [])):
            if not result.get("ruleId"):
                errors.append(f"run[{i}].results[{j}].ruleId is required")
            if result.get("level") not in ("error", "warning", "note", "none"):
                errors.append(f"run[{i}].results[{j}].level must be error/warning/note/none")
            if not result.get("message", {}).get("text"):
                errors.append(f"run[{i}].results[{j}].message.text is required")

    return (len(errors) == 0, errors)


def export_and_write(
    results: list[dict[str, Any]],
    output_path: str,
    repo_path: str = "",
    commit_sha: str = "",
    tool_version: str = "1.0.0",
) -> dict:
    """Export SARIF and write to file. Returns the SARIF doc."""
    sarif_doc = export_sarif(
        results, repo_path=repo_path, commit_sha=commit_sha,
        tool_version=tool_version,
    )

    is_valid, errors = validate_sarif(sarif_doc)
    if not is_valid:
        raise ValueError(f"SARIF validation failed: {'; '.join(errors)}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sarif_doc, indent=2), encoding="utf-8")

    return sarif_doc
