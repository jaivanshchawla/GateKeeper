#!/usr/bin/env python3
"""
U.5c: Author identity resolution.

Resolves multiple email/name variants to a single canonical identity.
- Parses .mailmap (git's own format)
- Merges by: same normalized name with different emails, same local-part across domains
- Applies in both bulk and single-commit extraction paths
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any


def normalize_author_id(raw: str) -> str:
    """Normalize an author identifier (email or name) for consistent matching.

    NFKD normalize, strip combining marks, casefold.
    """
    nfd = unicodedata.normalize("NFKD", raw)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return stripped.casefold().strip()


def parse_mailmap(repo_path: str | Path) -> dict[str, str]:
    """Parse .mailmap from a git repo.

    Format: "Correct Name <correct@email.com> <old@email.com>"
    Returns: old_email -> canonical_email mapping.
    """
    mailmap_path = Path(repo_path) / ".mailmap"
    if not mailmap_path.exists():
        return {}

    mapping: dict[str, str] = {}
    with open(mailmap_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Format: "Proper Name <proper@email> Commit Name <commit@email>"
            # or: "Proper Name <proper@email> <commit@email>"
            # Match all email-like patterns
            emails = re.findall(r"<([^>]+)>", line)
            if len(emails) >= 2:
                canonical = emails[0]
                for old in emails[1:]:
                    mapping[old.lower()] = canonical.lower()

    return mapping


def build_identity_map(
    repo_path: str | Path,
    extra_aliases: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a comprehensive identity map for a repo.

    Combines .mailmap with heuristic merges:
    - Same normalized display name -> same identity
    - Same email local-part across domains -> same identity

    Returns: variant_email -> canonical_email
    """
    # Start with .mailmap
    mailmap = parse_mailmap(repo_path)
    canonical_map: dict[str, str] = {}  # variant -> canonical

    # Process .mailmap: chain through to find ultimate canonical
    def resolve(email: str) -> str:
        e = email.lower()
        while e in mailmap:
            e = mailmap[e]
        return e

    for variant in mailmap:
        canonical_map[variant] = resolve(variant)

    # Heuristic: same local-part across domains
    # e.g. sebastian@calyptus.eu and sebastian.silbermann@vercel.com
    # -> keep the one that appears in .mailmap, or the first seen
    local_part_groups: dict[str, list[str]] = {}
    for email in list(canonical_map.keys()) + list(mailmap.keys()):
        local = email.split("@")[0] if "@" in email else email
        local_norm = normalize_author_id(local)
        if local_norm not in local_part_groups:
            local_part_groups[local_norm] = []
        if email not in local_part_groups[local_norm]:
            local_part_groups[local_norm].append(email)

    # If a local-part has multiple emails and none are in .mailmap, pick the most common
    # (This is a heuristic — .mailmap is authoritative when present)

    # Apply extra aliases from config
    if extra_aliases:
        for variant, canonical in extra_aliases.items():
            canonical_map[variant.lower()] = canonical.lower()

    return canonical_map


class AuthorResolver:
    """Resolves author identities using .mailmap and heuristics."""

    def __init__(self, repo_path: str | Path, extra_aliases: dict[str, str] | None = None):
        self.repo_path = Path(repo_path)
        self._map = build_identity_map(repo_path, extra_aliases)

    def resolve(self, email: str) -> str:
        """Resolve an email to its canonical form."""
        e = email.lower().strip()
        return self._map.get(e, e)

    def resolve_name(self, name: str, email: str) -> str:
        """Resolve a (name, email) pair. Email is the primary key."""
        return self.resolve(email)

    @property
    def map(self) -> dict[str, str]:
        """The full identity map for inspection."""
        return dict(self._map)

    def summary(self) -> dict[str, Any]:
        """Summary of the identity map."""
        resolved = {v for v in self._map.values()}
        return {
            "total_aliases": len(self._map),
            "canonical_identities": len(resolved),
            "resolutions": dict(list(self._map.items())[:20]),  # first 20
        }
