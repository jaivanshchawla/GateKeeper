"""Feast feature views for commit risk features."""

from datetime import timedelta

from data_source import commit_features_source
from entities import commit_entity
from feast import FeatureView, Field
from feast.types import Int32

commit_features_view = FeatureView(
    name="commit_features",
    entities=[commit_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="lines_added", dtype=Int32),
        Field(name="lines_deleted", dtype=Int32),
        Field(name="files_touched", dtype=Int32),
        Field(name="dirs_touched", dtype=Int32),
        Field(name="author_prior_commits", dtype=Int32),
        Field(name="hour_of_day", dtype=Int32),
        Field(name="day_of_week", dtype=Int32),
        Field(name="commit_msg_length", dtype=Int32),
        Field(name="is_fix_bug_revert", dtype=Int32),
        Field(name="risky", dtype=Int32),
    ],
    source=commit_features_source,
    online=True,
)
