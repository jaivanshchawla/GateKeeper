#!/usr/bin/env python3
"""
Unit tests for CommitFeatureExtractor feature computation logic.

Uses synthetic mock PyDriller commit objects to test feature extraction
without needing to clone a real repository.
"""

import os

# Add parent directory to path so we can import the module
import sys
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.extract_features import CommitFeatureExtractor


def create_mock_commit(
    hash: str = "abc123",
    author_name: str = "TestAuthor",
    author_email: str = None,
    author_date: datetime = None,
    committer_date: datetime = None,
    insertions: int = 10,
    deletions: int = 5,
    files: int = 2,
    modified_files: list[dict] = None,
    msg: str = "test commit",
) -> SimpleNamespace:
    """Create a mock PyDriller commit object for testing."""
    if author_date is None:
        author_date = datetime(2024, 1, 15, 10, 30, 0)
    if committer_date is None:
        committer_date = author_date
    if author_email is None:
        author_email = f"{author_name.lower()}@example.com"

    if modified_files is None:
        modified_files = [
            {"new_path": "src/module.py", "old_path": None},
        ]

    # Create mock modified file objects
    mock_modified_files = []
    for mf in modified_files:
        mock_modified_files.append(
            SimpleNamespace(
                new_path=mf.get("new_path"),
                old_path=mf.get("old_path"),
            )
        )

    return SimpleNamespace(
        hash=hash,
        author=SimpleNamespace(name=author_name, email=author_email),
        author_date=author_date,
        committer_date=committer_date,
        insertions=insertions,
        deletions=deletions,
        files=files,
        modified_files=mock_modified_files,
        msg=msg,
    )


class TestCommitFeatureExtractor:
    """Test suite for CommitFeatureExtractor feature computation."""

    def test_feature_extraction_basic(self):
        """Test that feature extraction produces expected keys."""
        extractor = CommitFeatureExtractor(
            repo_path="/fake/repo",
            since="2024-01-01",
            label_window_days=7,
        )

        commit = create_mock_commit(
            hash="abc123def456", # pragma: allowlist secret
            author_name="Alice",
            insertions=42,
            deletions=10,
            files=3,
            modified_files=[
                {"new_path": "src/main.py", "old_path": None},
                {"new_path": "src/utils.py", "old_path": None},
                {"new_path": "tests/test_main.py", "old_path": None},
            ],
            msg="Add new feature",
        )

        features = extractor._extract_features_from_commit(commit)

        # Check all expected feature keys are present
        expected_keys = {
            "hash",
            "author",
            "date",
            "lines_added",
            "lines_deleted",
            "files_touched",
            "dirs_touched",
            "author_prior_commits",
            "hour_of_day",
            "day_of_week",
            "commit_msg_length",
            "is_fix_bug_revert",
            "commit_msg",
        }
        assert set(features.keys()) == expected_keys

    def test_feature_extraction_lines(self):
        """Test that lines added/deleted are correctly extracted."""
        extractor = CommitFeatureExtractor(
            repo_path="/fake/repo",
            since="2024-01-01",
        )

        commit = create_mock_commit(insertions=100, deletions=25)
        features = extractor._extract_features_from_commit(commit)

        assert features["lines_added"] == 100
        assert features["lines_deleted"] == 25

    def test_feature_extraction_files_and_dirs(self):
        """Test that files and directories touched are counted correctly."""
        extractor = CommitFeatureExtractor(
            repo_path="/fake/repo",
            since="2024-01-01",
        )

        commit = create_mock_commit(
            files=5,
            modified_files=[
                {"new_path": "src/main.py", "old_path": None},
                {"new_path": "src/utils.py", "old_path": None},
                {"new_path": "src/helpers.py", "old_path": None},
                {"new_path": "tests/test_main.py", "old_path": None},
                {"new_path": "docs/readme.md", "old_path": None},
            ],
        )
        features = extractor._extract_features_from_commit(commit)

        assert features["files_touched"] == 5
        # Directories: src, tests, docs = 3
        assert features["dirs_touched"] == 3

    def test_feature_extraction_temporal(self):
        """Test that temporal features (hour, day of week) are extracted."""
        extractor = CommitFeatureExtractor(
            repo_path="/fake/repo",
            since="2024-01-01",
        )

        # Wednesday at 23:45
        commit_date = datetime(2024, 1, 17, 23, 45, 0)  # Wednesday
        commit = create_mock_commit(author_date=commit_date)

        features = extractor._extract_features_from_commit(commit)

        assert features["hour_of_day"] == 23
        assert features["day_of_week"] == 2  # Wednesday = 2

    def test_feature_extraction_author_prior_commits(self):
        """Test that author prior commit count increments correctly."""
        extractor = CommitFeatureExtractor(
            repo_path="/fake/repo",
            since="2024-01-01",
        )

        # First commit from Alice
        commit1 = create_mock_commit(
            author_name="Alice",
            msg="First commit",
        )
        features1 = extractor._extract_features_from_commit(commit1)
        assert features1["author_prior_commits"] == 0

        # Second commit from Alice
        commit2 = create_mock_commit(
            author_name="Alice",
            msg="Second commit",
        )
        features2 = extractor._extract_features_from_commit(commit2)
        assert features2["author_prior_commits"] == 1

        # First commit from Bob
        commit3 = create_mock_commit(
            author_name="Bob",
            msg="Bob's first commit",
        )
        features3 = extractor._extract_features_from_commit(commit3)
        assert features3["author_prior_commits"] == 0

    def test_feature_extraction_fix_bug_revert_keywords(self):
        """Test that fix/bug/revert keywords are detected."""
        extractor = CommitFeatureExtractor(
            repo_path="/fake/repo",
            since="2024-01-01",
        )

        # Test "fix" keyword
        commit_fix = create_mock_commit(msg="Fix login issue")
        features_fix = extractor._extract_features_from_commit(commit_fix)
        assert features_fix["is_fix_bug_revert"] == 1

        # Test "bug" keyword
        commit_bug = create_mock_commit(msg="bug: fix null pointer")
        features_bug = extractor._extract_features_from_commit(commit_bug)
        assert features_bug["is_fix_bug_revert"] == 1

        # Test "revert" keyword
        commit_revert = create_mock_commit(msg="Revert changes from #123")
        features_revert = extractor._extract_features_from_commit(commit_revert)
        assert features_revert["is_fix_bug_revert"] == 1

        # Test no keyword
        commit_normal = create_mock_commit(msg="Add new feature")
        features_normal = extractor._extract_features_from_commit(commit_normal)
        assert features_normal["is_fix_bug_revert"] == 0

    def test_feature_extraction_commit_msg_length(self):
        """Test that commit message length is correctly computed."""
        extractor = CommitFeatureExtractor(
            repo_path="/fake/repo",
            since="2024-01-01",
        )

        msg = "Fix critical security vulnerability in auth module"
        commit = create_mock_commit(msg=msg)
        features = extractor._extract_features_from_commit(commit)

        assert features["commit_msg_length"] == len(msg)

    def test_feature_extraction_empty_modified_files(self):
        """Test handling of empty modified files list."""
        extractor = CommitFeatureExtractor(
            repo_path="/fake/repo",
            since="2024-01-01",
        )

        commit = create_mock_commit(
            modified_files=[],
            files=0,
        )
        features = extractor._extract_features_from_commit(commit)

        assert features["files_touched"] == 0
        assert features["dirs_touched"] == 0

    def test_label_commits_revert_detection(self):
        """Test that revert commits are labeled as risky."""
        extractor = CommitFeatureExtractor(
            repo_path="/fake/repo",
            since="2024-01-01",
            label_window_days=7,
        )

        # Create commits - one normal, one revert
        features_list = [
            {
                "hash": "abc123",
                "author": "Alice",
                "date": datetime(2024, 1, 15, 10, 0),
                "lines_added": 50,
                "lines_deleted": 0,
                "files_touched": 3,
                "dirs_touched": 2,
                "author_prior_commits": 0,
                "hour_of_day": 10,
                "day_of_week": 0,
                "commit_msg_length": 20,
                "is_fix_bug_revert": 0,
                "commit_msg": "Add new feature",
            },
            {
                "hash": "def456",
                "author": "Bob",
                "date": datetime(2024, 1, 16, 14, 0),
                "lines_added": 0,
                "lines_deleted": 50,
                "files_touched": 3,
                "dirs_touched": 2,
                "author_prior_commits": 0,
                "hour_of_day": 14,
                "day_of_week": 1,
                "commit_msg_length": 30,
                "is_fix_bug_revert": 1,
                "commit_msg": "Revert changes from abc123",
            },
        ]

        # Manually populate file_touches and commit_info for labeling
        extractor.file_touches = {
            "src/main.py": [
                ("abc123", datetime(2024, 1, 15, 10, 0)),
                ("def456", datetime(2024, 1, 16, 14, 0)),
            ],
        }
        extractor.commit_info = {
            "abc123": {"date": datetime(2024, 1, 15), "files": {"src/main.py"}},
            "def456": {"date": datetime(2024, 1, 16), "files": {"src/main.py"}},
        }

        df = pd.DataFrame(features_list)
        labeled_df = extractor._label_commits(df)

        # Both should be risky: abc123 because its file was touched again within 7 days,
        # def456 because it's a revert
        assert labeled_df[labeled_df["hash"] == "abc123"]["risky"].values[0] == 1
        assert labeled_df[labeled_df["hash"] == "def456"]["risky"].values[0] == 1

    def test_feature_types_are_numeric(self):
        """Test that extracted features have correct types for ML."""
        extractor = CommitFeatureExtractor(
            repo_path="/fake/repo",
            since="2024-01-01",
        )

        commit = create_mock_commit(
            insertions=42,
            deletions=10,
            files=3,
        )
        features = extractor._extract_features_from_commit(commit)

        # Numeric features should be int
        assert isinstance(features["lines_added"], int)
        assert isinstance(features["lines_deleted"], int)
        assert isinstance(features["files_touched"], int)
        assert isinstance(features["dirs_touched"], int)
        assert isinstance(features["author_prior_commits"], int)
        assert isinstance(features["hour_of_day"], int)
        assert isinstance(features["day_of_week"], int)
        assert isinstance(features["commit_msg_length"], int)
        assert isinstance(features["is_fix_bug_revert"], int)

    def test_extract_single_commit_returns_dict(self):
        """Test that extract_single_commit returns a dictionary with expected keys."""
        extractor = CommitFeatureExtractor(
            repo_path="/fake/repo",
            since="2024-01-01",
        )

        # Create a mock repository that returns a single commit
        mock_commit = create_mock_commit(
            hash="deadbeef123",
            author_name="Tester",
            insertions=25,
            deletions=5,
            files=2,
            modified_files=[
                {"new_path": "src/file1.py", "old_path": None},
                {"new_path": "src/file2.py", "old_path": None},
            ],
            msg="Update implementation",
        )

        # Mock subprocess.run calls (git log for commit date + author counts)
        from unittest import mock
        import subprocess

        def mock_subprocess_run(cmd, **kwargs):
            cmd_str = " ".join(cmd)
            # git log -1 --format=%ct <hash> -> return commit timestamp
            if "--format=%ct" in cmd_str:
                result = mock.MagicMock()
                result.stdout = "1700000000"
                result.returncode = 0
                return result
            # git log --before=<date> --format=%aE -> return author counts (email)
            if "--format=%aE" in cmd_str and "--before" in cmd_str:
                result = mock.MagicMock()
                result.stdout = "tester@example.com\ntester@example.com\nother@example.com"
                result.returncode = 0
                return result
            # git log --name-only -> return file paths for graph (uses %aE now)
            if "--name-only" in cmd_str:
                result = mock.MagicMock()
                result.stdout = "deadbeef123|1700000000|tester@example.com|Update implementation\nsrc/file1.py\nsrc/file2.py\n"
                result.returncode = 0
                return result
            # git log --merges -> empty
            if "--merges" in cmd_str:
                result = mock.MagicMock()
                result.stdout = ""
                result.returncode = 0
                return result
            # Default
            result = mock.MagicMock()
            result.stdout = ""
            result.returncode = 0
            return result

        with mock.patch("ml.extract_features.Repository") as MockRepo, \
             mock.patch("subprocess.run", side_effect=mock_subprocess_run), \
             mock.patch("ml.m1_shared.subprocess.run", side_effect=mock_subprocess_run):
            mock_instance = MockRepo.return_value
            mock_instance.traverse_commits.return_value = [mock_commit]

            result = extractor.extract_single_commit("/fake/repo", "deadbeef123")

            assert isinstance(result, dict)
            assert result["hash"] == "deadbeef123"
            assert result["lines_added"] == 25
            # Author is now normalized email, not display name
            assert "tester@example.com" in result["author"]
            # commit_msg should be removed
            assert "commit_msg" not in result
