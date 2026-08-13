"""
Validation component for the Gatekeeper retraining pipeline.
Checks row count sanity, class balance, and schema validity.
"""

from kfp import dsl


@dsl.component(
    packages_to_install=["pandas", "pyyaml"],
)
def validate(
    features_path: str,
    min_rows: int = 100,
    min_positive_pct: float = 0.05,
) -> str:
    """
    Validate the extracted features before training.

    Checks:
    1. Row count sanity check (at least min_rows)
    2. Class balance check (warn if positive class < 5%)
    3. Schema check (all expected columns exist)

    Args:
        features_path: Path to the features CSV
        min_rows: Minimum number of rows expected
        min_positive_pct: Minimum fraction of positive class (default 5%)

    Returns:
        Path to the validated features CSV
    """
    import sys
    import yaml
    import pandas as pd
    from pathlib import Path

    print(f"Validating features at {features_path}")

    # Load the data
    df = pd.read_csv(features_path)
    total_rows = len(df)
    print(f"Total rows: {total_rows}")

    # 1. Row count sanity check
    if total_rows < min_rows:
        msg = f"FAIL: Only {total_rows} rows, expected at least {min_rows}"
        print(msg, file=sys.stderr)
        raise ValueError(msg)
    print(f"✓ Row count OK: {total_rows} >= {min_rows}")

    # 2. Class balance check
    if "risky" not in df.columns:
        msg = "FAIL: 'risky' column not found in features"
        print(msg, file=sys.stderr)
        raise ValueError(msg)

    positive_count = int(df["risky"].sum())
    positive_pct = positive_count / total_rows
    print(f"Class balance: {positive_count} positive ({positive_pct:.2%}), "
          f"{total_rows - positive_count} negative ({1 - positive_pct:.2%})")

    if positive_pct < min_positive_pct:
        print(
            f"WARNING: Positive class is {positive_pct:.2%}, "
            f"below threshold of {min_positive_pct:.2%}. "
            f"Consider resampling or adjusting labeling criteria.",
            file=sys.stderr,
        )
    else:
        print(f"✓ Class balance OK: {positive_pct:.2%} >= {min_positive_pct:.2%}")

    # 3. Schema check - load expected columns from config
    project_root = str(Path(__file__).parent.parent.parent)
    config_path = Path(project_root) / "ml" / "config.yaml"

    expected_columns = []
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        expected_columns = config.get("feature_columns", [])
        # Also expect 'risky' and 'hash'
        expected_columns.extend(["risky", "hash"])

    if expected_columns:
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            msg = f"FAIL: Missing columns in features: {missing}"
            print(msg, file=sys.stderr)
            raise ValueError(msg)
        print(f"✓ Schema OK: All {len(expected_columns)} expected columns present")
    else:
        print("No config found, skipping schema check")

    print("All validations passed!")
    return features_path
