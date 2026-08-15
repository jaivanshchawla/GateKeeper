"""Feast data source for commit features."""

from feast import FileSource

commit_features_source = FileSource(
    path="../data/commit_features.parquet",
    timestamp_field="date",
)
