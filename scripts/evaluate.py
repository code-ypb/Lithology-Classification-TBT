#!/usr/bin/env python3
"""Evaluation script for trained TCN-BiLSTM-Transformer model.

Loads a trained model from a checkpoint, evaluates it on test data,
generates comprehensive evaluation plots and metrics, and saves all
results to the specified output directory.

Usage:
    python scripts/evaluate.py --checkpoint output/best_model.pt --data_dir data/ --output_dir eval_output/
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Ensure project root is on sys.path for `src` imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.core.model import Model_TCN_BiLSTM_Transformer
from src.core.feature_engineering import engineer_features
from src.core.postprocess import ensemble_postprocess
from src.utils.data_loader import load_raw_well, create_windows, set_seed
from src.utils.metrics import evaluate_model, compute_comprehensive_metrics
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_roc_pr_curves,
    plot_depth_facies,
    plot_error_analysis,
    plot_postprocess_comparison,
    plot_integrated_gradients,
    plot_attention_weights,
    plot_saliency,
    plot_temporal_importance,
    plot_shap_summary,
)

# Default class names and colours
TARGET_NAMES = [
    "Medium Sandstone",
    "Mudstone",
    "Glutenite",
    "Siltstone",
    "Coarse Sandstone",
]
TARGET_COLORS = ["#FFD700", "#808080", "#FF4500", "#87CEEB", "#FF6347"]


def load_model_from_checkpoint(checkpoint_path, device):
    """Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Torch device to load the model onto.

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


def load_test_wells(data_dir, scaler, window_size, feature_names=None):
    """Load and prepare test well data from a directory.

    Args:
        data_dir: Directory containing well-log CSV files.
        scaler: Fitted StandardScaler.
        window_size: Sliding window size.
        feature_names: Expected feature column names.

    Returns:
        List of (X, Y, depth, well_name) tuples.
    """
    csv_paths = sorted(Path(data_dir).glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    test_wells = []

    for path in csv_paths:
        df, depth, feat_cols = load_raw_well(str(path), window_size=window_size)
        if df is None:
            print(f"[SKIP] Could not load {path.name}")
            continue

        # Feature engineering
        df, all_feat = engineer_features(df, depth, feat_cols)

        # Create windows with the provided scaler
        X, Y, aligned_depth, _, _ = create_windows(
            df, all_feat, depth, scaler=scaler, fit_scaler=False,
            window_size=window_size,
        )

        if X.shape[0] > 0:
            test_wells.append((X, Y, aligned_depth, path.stem))

    return test_wells


def evaluate_on_well(model, X, Y, device, batch_size=512):
    """Evaluate the model on a single well.

    Args:
        model: Trained model.
        X: Input features of shape (n_samples, window_size, n_features).
        Y: Ground-truth labels of shape (n_samples,).
        device: Torch device.
        batch_size: Batch size for inference.

    Returns:
        Tuple of (predictions, probabilities).
    """
    model.eval()
    all_logits = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            logits = model(xb)
            all_logits.append(logits)

    logits_all = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits_all, dim=-1).cpu().numpy()
    preds = probs.argmax(axis=1)

    return preds, probs


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained TCN-BiLSTM-Transformer model"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint (.pt file)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing test well CSV files")
    parser.add_argument("--output_dir", type=str, default="eval_output/",
                        help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for inference")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of lithology classes")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  TCN-BiLSTM-Transformer — Model Evaluation")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Device:     {device}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("\n[1/6] Loading model from checkpoint...")
    model, config, feature_names = load_model_from_checkpoint(args.checkpoint, device)
    num_classes = config["num_classes"]
    window_size = config["window_size"]
    print(f"  Model loaded — d_input={config['d_input']}, "
          f"num_classes={num_classes}, "
          f"window_size={window_size}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Load scaler
    # ------------------------------------------------------------------
    print("\n[2/6] Loading scaler...")
    import joblib
    checkpoint_dir = os.path.dirname(args.checkpoint)
    scaler_path = os.path.join(checkpoint_dir, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"  Scaler loaded from {scaler_path}")
    else:
        print(f"  [WARNING] Scaler not found at {scaler_path}. "
              "A new scaler will be fitted on test data (not recommended).")
        scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    print("\n[3/6] Loading test well data...")
    test_wells = load_test_wells(
        args.data_dir, scaler, window_size, feature_names
    )
    print(f"  Loaded {len(test_wells)} test wells")

    # ------------------------------------------------------------------
    # Evaluate each well
    # ------------------------------------------------------------------
    print("\n[4/6] Evaluating model on test wells...")
    all_results = []

    for X, Y, depth, well_name in test_wells:
        preds_raw, probs = evaluate_on_well(model, X, Y, device, args.batch_size)

        # Raw metrics
        metrics_raw = compute_comprehensive_metrics(Y, preds_raw, probs)

        # Post-processed metrics
        preds_post = ensemble_postprocess(preds_raw, probs, num_classes)
        metrics_post = compute_comprehensive_metrics(Y, preds_post, probs)

        result = {
            "well_name": well_name,
            "n_samples": len(Y),
            "raw": {k: v for k, v in metrics_raw.items() if k != "confusion_matrix"},
            "postprocessed": {k: v for k, v in metrics_post.items() if k != "confusion_matrix"},
        }
        all_results.append(result)

        print(f"\n  Well: {well_name} ({len(Y)} samples)")
        print(f"    Raw — accuracy={metrics_raw['accuracy']:.4f}  "
              f"macro_f1={metrics_raw['macro_f1']:.4f}  "
              f"weighted_f1={metrics_raw['weighted_f1']:.4f}")
        print(f"    Post — accuracy={metrics_post['accuracy']:.4f}  "
              f"macro_f1={metrics_post['macro_f1']:.4f}  "
              f"weighted_f1={metrics_post['weighted_f1']:.4f}")

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    print("\n[5/6] Generating evaluation plots...")

    # Use the first well for detailed plots
    if test_wells:
        X, Y, depth, well_name = test_wells[0]
        preds_raw, probs = evaluate_on_well(model, X, Y, device, args.batch_size)
        preds_post = ensemble_postprocess(preds_raw, probs, num_classes)
        metrics = compute_comprehensive_metrics(Y, preds_post, probs)

        try:
            plot_confusion_matrix(
                metrics["confusion_matrix"], TARGET_NAMES, TARGET_COLORS,
                output_dir=args.output_dir,
            )
            plot_per_class_metrics(
                metrics["per_class"], TARGET_NAMES, TARGET_COLORS,
                output_dir=args.output_dir,
            )
            plot_roc_pr_curves(
                Y, probs, TARGET_NAMES, TARGET_COLORS, num_classes,
                output_dir=args.output_dir,
            )
            plot_depth_facies(
                depth, Y, preds_post, TARGET_NAMES, TARGET_COLORS,
                num_classes, output_dir=args.output_dir,
            )
            plot_error_analysis(
                Y, preds_post, TARGET_NAMES, num_classes,
                output_dir=args.output_dir,
            )
            # Compute median-filtered predictions for postprocess comparison
            from scipy.ndimage import median_filter
            preds_med = median_filter(preds_raw, size=3, mode="nearest").astype(np.int64)

            plot_postprocess_comparison(
                Y, preds_raw, preds_med, preds_post, probs, TARGET_NAMES,
                TARGET_COLORS, num_classes, output_dir=args.output_dir,
            )
        except Exception as e:
            print(f"  [WARNING] Could not generate some evaluation plots: {e}")

        # Interpretability plots
        print("\n[6/6] Generating interpretability plots...")
        try:
            plot_integrated_gradients(
                model, X, Y, feature_names, device, window_size,
                num_classes, TARGET_NAMES, TARGET_COLORS,
                output_dir=args.output_dir,
            )
            plot_attention_weights(
                model, X, Y, device, window_size,
                output_dir=args.output_dir,
            )
            plot_saliency(
                model, X, Y, feature_names, device,
                output_dir=args.output_dir,
            )
            plot_temporal_importance(
                model, X, Y, device, window_size,
                output_dir=args.output_dir,
            )
            plot_shap_summary(
                model, X, Y, feature_names, device, window_size,
                num_classes, TARGET_NAMES, TARGET_COLORS,
                output_dir=args.output_dir,
            )
        except Exception as e:
            print(f"  [WARNING] Could not generate interpretability plots: {e}")
    else:
        print("  No test wells available for plotting.")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Aggregate summary
    if all_results:
        avg_raw_acc = np.mean([r["raw"]["accuracy"] for r in all_results])
        avg_raw_mf1 = np.mean([r["raw"]["macro_f1"] for r in all_results])
        avg_post_acc = np.mean([r["postprocessed"]["accuracy"] for r in all_results])
        avg_post_mf1 = np.mean([r["postprocessed"]["macro_f1"] for r in all_results])

        print(f"\n  Aggregate Results (across {len(all_results)} wells):")
        print(f"    Raw — accuracy={avg_raw_acc:.4f}  macro_f1={avg_raw_mf1:.4f}")
        print(f"    Post — accuracy={avg_post_acc:.4f}  macro_f1={avg_post_mf1:.4f}")

    print("\n" + "=" * 60)
    print("  Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
