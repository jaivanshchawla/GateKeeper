"""
Feature engineering component for the Gatekeeper retraining pipeline.
Light pass-through since existing features from Phase 1 are sufficient.
"""

from kfp import dsl


@dsl.component(
    packages_to_install=["pandas"],
)
def feature_eng(
    features_path: dsl.InputPath("Dataset"),
    engineered_features_path: dsl.OutputPath("Dataset"),
) -> None:
    """
    Light feature engineering pass-through.

    Args:
        features_path: Path to the input features CSV.
        engineered_features_path: KFP output path for the retained features CSV.
    """
    from pathlib import Path

    import pandas as pd

    print(f"Loading features from {features_path}")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    output_path = Path(engineered_features_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Feature engineering complete. {len(df.columns)} features retained.")
    print(f"Engineered features saved to {output_path}")
