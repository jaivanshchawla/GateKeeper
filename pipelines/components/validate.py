"""
Validation component for the Gatekeeper retraining pipeline.
Checks row count sanity, class balance, and schema validity.
"""

from kfp import dsl


@dsl.component(
    base_image="gatekeeper-kfp-base",
    packages_to_install=["pandas", "pyyaml"],
)
def validate(
    features_path: dsl.InputPath("Dataset"),
    validated_features_path: dsl.OutputPath("Dataset"),
    min_rows: int = 100,
    min_positive_pct: float = 0.05,
) -> None:
    """
    Validate the extracted features before training.

    Args:
        features_path: Path to the input features CSV.
        validated_features_path: KFP output path for the validated features CSV.
        min_rows: Minimum number of rows expected.
        min_positive_pct: Minimum fraction of positive class before warning.
    """
    import sys
    from pathlib import Path

    import pandas as pd
    import yaml

    print(f"Validating features at {features_path}")
    df = pd.read_csv(features_path)
    total_rows = len(df)
    print(f"Total rows: {total_rows}")

    if total_rows < min_rows:
        msg = f"FAIL: Only {total_rows} rows, expected at least {min_rows}"
        print(msg, file=sys.stderr)
        raise ValueError(msg)
    print(f"Row count OK: {total_rows} >= {min_rows}")

    if "risky" not in df.columns:
        msg = "FAIL: 'risky' column not found in features"
        print(msg, file=sys.stderr)
        raise ValueError(msg)

    positive_count = int(df["risky"].sum())
    positive_pct = positive_count / total_rows
    print(
        f"Class balance: {positive_count} positive ({positive_pct:.2%}), "
        f"{total_rows - positive_count} negative ({1 - positive_pct:.2%})"
    )

    if positive_pct < min_positive_pct:
        print(
            f"WARNING: Positive class is {positive_pct:.2%}, "
            f"below threshold of {min_positive_pct:.2%}.",
            file=sys.stderr,
        )
    else:
        print(f"Class balance OK: {positive_pct:.2%} >= {min_positive_pct:.2%}")

    project_root = Path.cwd()
    config_path = project_root / "ml" / "config.yaml"

    expected_columns = []
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        expected_columns = config.get("feature_columns", [])
        expected_columns.extend(["risky", "hash"])

    if expected_columns:
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            msg = f"FAIL: Missing columns in features: {missing}"
            print(msg, file=sys.stderr)
            raise ValueError(msg)
        print(f"Schema OK: All {len(expected_columns)} expected columns present")
    else:
        print("No config found, skipping schema check")

    output_path = Path(validated_features_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("All validations passed!")
    print(f"Validated features saved to {output_path}")
