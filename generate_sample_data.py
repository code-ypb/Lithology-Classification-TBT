#!/usr/bin/env python3
"""Sample data generator for TCN-BiLSTM-Transformer lithology identification.

Generates synthetic well-log data with realistic statistical properties for
testing and demonstration purposes. The generated data mimics the format
of real well-log datasets but does not contain any proprietary information.

Usage:
    python generate_sample_data.py --output_dir sample_data/ --num_wells 9
"""

import argparse
import os
import numpy as np
import pandas as pd


# Lithology-specific well-log parameter ranges
# Format: {curve_name: (low, high, std)}
LITHOLOGY_PARAMS = {
    "Medium Sandstone": {
        "CAL": (249.0, 250.5, 0.8), "DEN": (2.55, 2.65, 0.05),
        "DT": (230.0, 280.0, 20.0), "GR": (30.0, 60.0, 12.0),
        "RT": (30.0, 150.0, 40.0), "SP": (-80.0, -50.0, 10.0),
    },
    "Mudstone": {
        "CAL": (249.0, 252.0, 1.2), "DEN": (2.40, 2.55, 0.06),
        "DT": (260.0, 330.0, 25.0), "GR": (80.0, 130.0, 18.0),
        "RT": (3.0, 20.0, 5.0), "SP": (-40.0, 20.0, 15.0),
    },
    "Glutenite": {
        "CAL": (249.0, 253.0, 1.5), "DEN": (2.60, 2.72, 0.05),
        "DT": (180.0, 240.0, 20.0), "GR": (40.0, 75.0, 15.0),
        "RT": (30.0, 180.0, 50.0), "SP": (-40.0, 10.0, 12.0),
    },
    "Siltstone": {
        "CAL": (249.0, 251.0, 0.9), "DEN": (2.50, 2.60, 0.04),
        "DT": (220.0, 280.0, 18.0), "GR": (50.0, 85.0, 12.0),
        "RT": (15.0, 65.0, 20.0), "SP": (-30.0, 15.0, 12.0),
    },
    "Coarse Sandstone": {
        "CAL": (249.0, 251.5, 1.0), "DEN": (2.55, 2.68, 0.05),
        "DT": (190.0, 250.0, 22.0), "GR": (25.0, 55.0, 10.0),
        "RT": (50.0, 300.0, 80.0), "SP": (-85.0, -55.0, 10.0),
    },
}

# Facies label mapping (matching the paper's convention)
# Labels are 0-4 (Fine Sandstone excluded)
LITHOLOGY_TO_FACIES = {
    "Medium Sandstone": 0,
    "Mudstone": 1,
    "Glutenite": 2,
    "Siltstone": 3,
    "Coarse Sandstone": 4,
}

FEATURE_COLS = ["CAL", "DEN", "DT", "GR", "RT", "SP"]


def generate_well(well_idx, depth_start=100.0, depth_end=600.0, depth_step=0.125):
    """Generate synthetic well-log data for a single well.

    Args:
        well_idx: Well index (used for randomization offset).
        depth_start: Starting depth in meters.
        depth_end: Ending depth in meters.
        depth_step: Depth sampling interval in meters.

    Returns:
        pd.DataFrame with columns: Depth, CAL, DEN, DT, GR, RT, SP, LITHOLOGY, Facies
    """
    rng = np.random.RandomState(42 + well_idx)
    depths = np.arange(depth_start, depth_end, depth_step)

    lithology_names = list(LITHOLOGY_PARAMS.keys())
    n_lithologies = len(lithology_names)

    # Create realistic layer sequence with varying thickness
    layer_sequence = []
    points_per_layer = max(10, len(depths) // (n_lithologies * 3))

    current_idx = 0
    while current_idx < len(depths):
        lith_name = lithology_names[rng.randint(0, n_lithologies)]
        # Variable layer thickness (thinner for some, thicker for others)
        thickness = rng.randint(max(5, points_per_layer // 3), points_per_layer * 2)
        layer_sequence.extend([lith_name] * thickness)
        current_idx += thickness

    records = []
    for i, depth in enumerate(depths):
        lith_idx = min(i, len(layer_sequence) - 1)
        lith_name = layer_sequence[lith_idx]
        params = LITHOLOGY_PARAMS[lith_name]

        row = {"Depth": round(depth, 3), "LITHOLOGY": lith_name}

        for curve in FEATURE_COLS:
            low, high, std = params[curve]
            mean = (low + high) / 2
            value = rng.normal(mean, std * 0.3)
            value = np.clip(value, low * 0.8, high * 1.2)
            if curve == "RT":
                value = max(0.5, value)
            row[curve] = round(value, 4)

        row["Facies"] = LITHOLOGY_TO_FACIES[lith_name]
        records.append(row)

    df = pd.DataFrame(records)

    # Add ~2% missing values for realism
    mask = rng.random(len(df)) < 0.02
    for curve in FEATURE_COLS:
        df.loc[mask, curve] = np.nan

    return df


def generate_sample_data(output_dir, num_wells=9):
    """Generate sample well-log dataset.

    Args:
        output_dir: Directory to save CSV files.
        num_wells: Number of wells to generate.
    """
    os.makedirs(output_dir, exist_ok=True)

    well_names = [f"Well_{i+1:02d}" for i in range(num_wells)]

    for i, name in enumerate(well_names):
        depth_start = 100.0 + i * 30
        depth_end = depth_start + 450 + i * 25
        df = generate_well(i, depth_start, depth_end)

        filepath = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"Generated: {filepath} ({len(df)} depth points)")

    print(f"\nGenerated {num_wells} wells in {output_dir}/")
    print(f"Lithology classes: {list(LITHOLOGY_PARAMS.keys())}")
    print(f"Log curves: {FEATURE_COLS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate sample well-log data for lithology identification"
    )
    parser.add_argument(
        "--output_dir", type=str, default="sample_data",
        help="Output directory for generated CSV files"
    )
    parser.add_argument(
        "--num_wells", type=int, default=9,
        help="Number of wells to generate"
    )
    args = parser.parse_args()
    generate_sample_data(args.output_dir, args.num_wells)
