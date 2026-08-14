#!/usr/bin/env python3
"""
Gatekeeper Retraining Pipeline (KFP definition).

Defines the genuine @dsl.pipeline for deployment on a KFP server or
Kubernetes cluster. For local execution, use run_retrain.py, which runs
this same graph through kfp.local.

The @dsl.component-decorated functions in pipelines/components/ are
designed for KFP deployment. kfp.local can run them as subprocesses,
but requires each component to be self-contained (no local imports).

To run locally:
    python pipelines/run_retrain.py

To compile for KFP:
    python pipelines/retrain_pipeline.py  # Compiles to YAML
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path for imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from kfp import dsl  # noqa: E402

from pipelines.components.automl_search import automl_search  # noqa: E402
from pipelines.components.feature_eng import feature_eng  # noqa: E402
from pipelines.components.ingest import ingest  # noqa: E402
from pipelines.components.register_model import register_model  # noqa: E402
from pipelines.components.validate import validate  # noqa: E402


@dsl.pipeline(
    name="gatekeeper-retrain-pipeline",
    description="Retrain the Gatekeeper risk prediction model",
)
def retrain_pipeline(
    repo_url: str = "https://github.com/django/django.git",
    since_date: str = (
        datetime.now(timezone.utc) - timedelta(days=3 * 365)
    ).strftime("%Y-%m-%d"),
    label_window_days: int = 7,
    min_rows: int = 100,
    min_positive_pct: float = 0.05,
    cached_csv_path: str = "",
):
    """
    End-to-end retraining pipeline for KFP deployment.

    Args:
        repo_url: Git repository URL to mine
        since_date: Date to start mining commits from
        label_window_days: Window for risky commit labeling
        min_rows: Minimum row count for validation
        min_positive_pct: Minimum positive class fraction
        cached_csv_path: If set, ingest reuses this CSV instead of mining
    """
    # Step 1: Ingest — clone repo and extract features
    ingest_task = ingest(
        repo_url=repo_url,
        since_date=since_date,
        label_window_days=label_window_days,
        cached_csv_path=cached_csv_path,
    )

    # Step 2: Feature engineering (light pass-through)
    feature_eng_task = feature_eng(features_path=ingest_task.outputs["features_path"])

    # Step 3: Validate features
    validate_task = validate(
        features_path=feature_eng_task.outputs["engineered_features_path"],
        min_rows=min_rows,
        min_positive_pct=min_positive_pct,
    )

    # Step 4: AutoML search
    automl_task = automl_search(
        features_path=validate_task.outputs["validated_features_path"],
    )

    # Step 5: Register best model
    register_model(
        model_path=automl_task.outputs["model_path"],
        automl_results_path=automl_task.outputs["automl_results_path"],
    )

    # Dependencies are automatically inferred by KFP v2 from data flow:
    # ingest -> feature_eng (via features_path)
    # feature_eng -> validate (via features_path)
    # validate -> automl_search (via features_path)
    # automl_search -> register_model (via model_path and automl_results_path)


if __name__ == "__main__":
    # Compile the pipeline to YAML for KFP server deployment
    import kfp.compiler

    compiler = kfp.compiler.Compiler()
    compiler.compile(
        pipeline_func=retrain_pipeline,
        package_path="retrain_pipeline.yaml",
    )
    print("Pipeline compiled to retrain_pipeline.yaml")
    print("For local execution, use: python pipelines/run_retrain.py")
