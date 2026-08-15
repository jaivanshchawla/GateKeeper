"""Feast entities for the Gatekeeper feature store."""

from feast import Entity
from feast.value_type import ValueType

commit_entity = Entity(
    name="commit",
    join_keys=["commit_hash"],
    description="A git commit identified by its hash",
    value_type=ValueType.STRING,
)
