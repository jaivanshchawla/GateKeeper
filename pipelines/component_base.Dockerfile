FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgomp1 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Pre-install ALL dependencies that pipeline components need.
# PyCaret removed due to sklearn version conflicts — AutoML uses
# manual LightGBM/RandomForest/LogisticRegression comparison instead.
RUN pip install --no-cache-dir \
    kfp \
    pandas \
    scikit-learn \
    lightgbm \
    mlflow \
    pydriller \
    skops \
    pyyaml

# Verify key imports work
ENV GIT_PYTHON_REFRESH=quiet
RUN python -c "import pydriller, pandas, sklearn, lightgbm, mlflow, skops, yaml; print(f'sklearn {sklearn.__version__}, lightgbm {lightgbm.__version__}')"

WORKDIR /workspace
