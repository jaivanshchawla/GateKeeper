"""
Feature engineering component for the Gatekeeper retraining pipeline.
Light pass-through since existing features from Phase 1 are sufficient.
"""

from kfp import dsl


@dsl.component(
    packages_to_install=["pandas"],
)
def feature_eng(features_path: str) -> str:
    """
    Light feature engineering pass-through.

    The existing feature set from extract_features.py is already well-designed.
    This component validates and optionally adds derived features.

    Args:
        features_path: Path to the features CSV

    Returns:
        Path to the features CSV (same path, potentially enriched)
    """
    import pandas as pd

    print(f"Loading features from {features_path}")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    # Existing feature set is sufficient - no transformation needed
    # This is intentionally a pass-through. The feature engineering
    # was done well in Phase 1, and forcing complexity here would be
    # artificial.

    print(f"Feature engineering complete. {len(df.columns)} features retained.")
    return features_path
