#!/usr/bin/env python3
"""
Export a stratified sample of commit_features.csv for CI drift testing.

This creates a representative subset (300-500 rows) that preserves the ~45/55
class balance, committed directly to git since DVC isn't available in CI.
"""

import os
import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Load the full dataset
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "commit_features.csv")
df = pd.read_csv(data_path)

print(f"Full dataset: {len(df)} rows")
print(f"Class balance: {df['risky'].value_counts(normalize=True).to_dict()}")

# Target sample size
TARGET_SIZE = 400

# Calculate how many to sample from each class to preserve balance
class_balance = df["risky"].value_counts(normalize=True)
n_risky = int(TARGET_SIZE * class_balance[1])
n_non_risky = TARGET_SIZE - n_risky

print(f"\nTarget sample size: {TARGET_SIZE}")
print(f"  Risky (1): {n_risky}")
print(f"  Non-risky (0): {n_non_risky}")

# Stratified sampling
df_risky = df[df["risky"] == 1].sample(n=min(n_risky, len(df[df["risky"] == 1])), random_state=42)
df_non_risky = df[df["risky"] == 0].sample(n=min(n_non_risky, len(df[df["risky"] == 0])), random_state=42)

# Combine and shuffle
sample = pd.concat([df_risky, df_non_risky]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nSample dataset: {len(sample)} rows")
print(f"Class balance: {sample['risky'].value_counts(normalize=True).to_dict()}")

# Verify the balance is close to original
original_balance = class_balance[1]
sample_balance = sample["risky"].mean()
print(f"\nOriginal positive rate: {original_balance:.2%}")
print(f"Sample positive rate: {sample_balance:.2%}")
print(f"Difference: {abs(original_balance - sample_balance):.2%}")

# Save to smoke_tests directory
output_path = os.path.join(os.path.dirname(__file__), "..", "smoke_tests", "reference_data_sample.csv")
sample.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
print(f"File size: {os.path.getsize(output_path):,} bytes")
