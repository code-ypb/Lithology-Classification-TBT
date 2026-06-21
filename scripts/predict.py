#!/usr/bin/env python3
"""Prediction script for single-well lithology identification.

Loads a trained TCN-BiLSTM-Transformer model from a checkpoint, runs
inference on a single well CSV file, and saves the predictions with
depths and class probabilities to a CSV file.

Usage:
    python scripts/predict.py --input well_data.csv --checkpoint output/best_model.pt --output predictions.csv
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for `src` imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.core.model import Model_TCN_BiLSTM_Transformer
from src.core.feature_engineering import engineer_features
from src.core.postprocess import ensemble_postprocess
from src.utils.data_loader import load_raw_well, create_windows, set_seed


# Default class label names
DEFAULT_LABEL_NAMES = {
    0: "Medium Sandstone",
    1: "Mudstone",
    2: "Glutenite",
    3: "Siltstone",
    4: "Coarse Sandstone",
}


def load_model_from_checkpoint(checkpoint_path, device):
    """Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Torch device.

    Returns:
        Tuple of (model, config_dict, feature_names).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    feature_names = checkpoint.get("feature_names", None)

    model = Model_TCN_BiLSTM_Transformer(
        d_input=config["d_input"],
        num_classes=config["num_classes"],
        window_size=config["window_size"],
        tcn_channels=config.get("tcn_channels", [32, 64]),
        lstm_hidden=config.get("lstm_hidden", 128),
        lstm_layers=config.get("lstm_layers", 1),
        nhead=config.get("nhead", 4),
        trans_fwd=config.get("trans_fwd", 128),
        trans_layers=config.get("trans_layers", 2),
        dropout=config.get("dropout", 0.2),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config, feature_names


def predict_well(model, X, device, batch_size=512):
    """Run inference on windowed well data.

    Args:
        model: Trained model.
        X: Input features of shape (n_windows, window_size, n_features).
        device: Torch device.
        batch_size: Batch size for inference.

    Returns:
        Tuple of (predictions, probabilities).
            - predictions: np.ndarray of shape (n_windows,) with predicted class indices.
            - probabilities: np.ndarray of shape (n_windows, num_classes) with softmax probabilities.
    """
    model.eval()
    all_logits = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(
                X[start:start + batch_size], dtype=torch.float32, device=device
            )
            logits = model(xb)
            all_logits.append(logits)

    logits_all = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits_all, dim=-1).cpu().numpy()
    preds = probs.argmax(axis=1)

    return preds, probs


def main():
    parser = argparse.ArgumentParser(
        description="Predict lithology for a single well using a trained model"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input well-log CSV file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint (.pt file)")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Path to save the prediction CSV file")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for inference")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--postprocess", action="store_true",
                        help="Apply ensemble post-processing to predictions")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of lithology classes")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  TCN-BiLSTM-Transformer — Single-Well Prediction")
    print("=" * 60)
    print(f"  Input:      {args.input}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output:     {args.output}")
    print(f"  Device:     {device}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("\n[1/4] Loading model from checkpoint...")
    model, config, feature_names = load_model_from_checkpoint(args.checkpoint, device)
    print(f"  Model loaded — d_input={config['d_input']}, "
          f"num_classes={config['num_classes']}, "
          f"window_size={config['window_size']}")

    # ------------------------------------------------------------------
    # Load scaler
    # ------------------------------------------------------------------
    print("\n[2/4] Loading scaler...")
    import joblib
    checkpoint_dir = os.path.dirname(args.checkpoint)
    scaler_path = os.path.join(checkpoint_dir, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"  Scaler loaded from {scaler_path}")
    else:
        print(f"  [WARNING] Scaler not found at {scaler_path}. "
              "A new scaler will be fitted on input data (not recommended).")
        scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Load and prepare input well
    # ------------------------------------------------------------------
    print("\n[3/4] Loading and preparing input well data...")
    df, depth, feat_cols = load_raw_well(
        args.input, window_size=config["window_size"]
    )
    if df is None:
        print("[ERROR] Could not load the input well. Aborting.")
        return

    # Feature engineering
    df, all_feat = engineer_features(df, depth, feat_cols)

    # Create windows
    X, Y, aligned_depth, _, _ = create_windows(
        df, all_feat, depth, scaler=scaler, fit_scaler=False,
        window_size=config["window_size"],
    )
    print(f"  Created {len(X)} windows of size {config['window_size']}")

    if len(X) == 0:
        print("[ERROR] No windows created. The well may be too short. Aborting.")
        return

    # ------------------------------------------------------------------
    # Run prediction
    # ------------------------------------------------------------------
    print("\n[4/4] Running prediction...")
    preds, probs = predict_well(model, X, device, args.batch_size)

    # Optional post-processing
    if args.postprocess:
        preds = ensemble_postprocess(preds, probs, args.num_classes)
        print("  Applied ensemble post-processing")

    # ------------------------------------------------------------------
    # Build output DataFrame
    # ------------------------------------------------------------------
    output_df = pd.DataFrame()
    output_df["Depth"] = aligned_depth

    # Add ground-truth labels if available
    if Y is not None and len(Y) > 0:
        output_df["True_Label"] = Y

    output_df["Predicted_Label"] = preds

    # Add class probability columns
    for cls_idx in range(args.num_classes):
        label_name = DEFAULT_LABEL_NAMES.get(cls_idx, f"Class_{cls_idx}")
        output_df[f"Prob_{label_name}"] = probs[:, cls_idx]

    # Add predicted label names
    label_names = []
    for pred in preds:
        label_names.append(DEFAULT_LABEL_NAMES.get(int(pred), f"Class_{int(pred)}"))
    output_df["Predicted_Name"] = label_names

    # Save to CSV
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"\n  Predictions saved to {args.output}")
    print(f"  Total predictions: {len(preds)}")

    # Print class distribution
    unique, counts = np.unique(preds, return_counts=True)
    print("\n  Predicted class distribution:")
    for cls, count in zip(unique, counts):
        name = DEFAULT_LABEL_NAMES.get(int(cls), f"Class_{int(cls)}")
        pct = 100.0 * count / len(preds)
        print(f"    {name} (class {cls}): {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("  Prediction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
