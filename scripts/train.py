#!/usr/bin/env python3
"""Training script for TCN-BiLSTM-Transformer lithology identification.

Supports:
1. Optuna hyperparameter optimization
2. Leave-one-well-out cross-validation
3. Final training with best hyperparameters
4. Complete interpretability analysis

Usage:
    python scripts/train.py --data_dir data/ --output_dir output/ --n_trials 30
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

# Ensure project root is on sys.path for `src` imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.core.model import Model_TCN_BiLSTM_Transformer
from src.core.loss import ClassBalancedFocalLoss
from src.core.feature_engineering import engineer_features
from src.core.postprocess import ensemble_postprocess
from src.core.augmentation import (
    oversample_thin_layers,
    mixup_data,
    mixup_criterion,
    create_weighted_sampler,
)
from src.utils.data_loader import load_raw_well, create_windows, set_seed
from src.utils.metrics import evaluate_model, compute_comprehensive_metrics
from src.utils.visualization import (
    plot_training_process,
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
    plot_cv_results,
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


# ---------------------------------------------------------------------------
# Helper: load all wells from a directory
# ---------------------------------------------------------------------------

def load_all_wells(data_dir, window_size=41):
    """Load all well CSV files from *data_dir*.

    Returns:
        well_data: list of (X, Y, depth) tuples
        well_names: list of well name strings
        feature_names: list of feature column names
        scaler: fitted StandardScaler
    """
    csv_paths = sorted(Path(data_dir).glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    well_names = [p.stem for p in csv_paths]
    raw_wells = []
    feature_names = None

    for path in csv_paths:
        df, depth, feat_cols = load_raw_well(str(path))
        if df is None:
            print(f"[SKIP] Could not load {path.name}")
            continue

        # Feature engineering
        df, all_feat = engineer_features(df, depth, feat_cols)
        if feature_names is None:
            feature_names = all_feat

        raw_wells.append((df, depth, all_feat, path.stem))

    if not raw_wells:
        raise RuntimeError("No wells loaded successfully.")

    # Fit a global scaler on all wells combined (numeric features only)
    numeric_feat = [f for f in raw_wells[0][2] if raw_wells[0][0][f].dtype in ('float64', 'float32', 'int64', 'int32')]
    all_features_combined = np.vstack(
        [df[numeric_feat].values.astype(np.float32) for df, _, _, _ in raw_wells]
    )
    scaler = StandardScaler()
    scaler.fit(all_features_combined)

    # Create windows for each well
    well_data = []
    loaded_names = []
    for df, depth, feat, name in raw_wells:
        X, Y, aligned_depth, _, _ = create_windows(
            df, feat, depth, scaler=scaler, fit_scaler=False, window_size=window_size
        )
        if X.shape[0] > 0:
            well_data.append((X, Y, aligned_depth))
            loaded_names.append(name)

    return well_data, loaded_names, feature_names, scaler


# ---------------------------------------------------------------------------
# Helper: train one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=True, mixup_alpha=0.3):
    """Train the model for one epoch.

    Args:
        model: The model to train.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Torch device.
        use_mixup: Whether to apply mixup augmentation.
        mixup_alpha: Mixup beta distribution parameter.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        if use_mixup and mixup_alpha > 0:
            xb_mixed, y_a, y_b, lam = mixup_data(xb, yb, alpha=mixup_alpha)
            logits = model(xb_mixed)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            logits = model(xb)
            loss = criterion(logits, yb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Helper: validate
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, X_val, Y_val, criterion, device, batch_size=512):
    """Evaluate the model on validation data.

    Returns:
        Tuple of (val_loss, predictions, probabilities).
    """
    model.eval()
    X_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_val, dtype=torch.long, device=device)

    all_logits = []
    for start in range(0, len(X_t), batch_size):
        xb = X_t[start : start + batch_size]
        logits = model(xb)
        all_logits.append(logits)

    logits_all = torch.cat(all_logits, dim=0)

    val_loss = criterion(logits_all, Y_t).item()
    probs = torch.softmax(logits_all, dim=-1).cpu().numpy()
    preds = probs.argmax(axis=1)

    return val_loss, preds, probs


# ---------------------------------------------------------------------------
# Helper: full training loop with early stopping
# ---------------------------------------------------------------------------

def train_model(
    model,
    X_train,
    Y_train,
    X_val,
    Y_val,
    criterion,
    device,
    num_epochs=100,
    batch_size=512,
    lr=0.001,
    patience=15,
    use_mixup=True,
    mixup_alpha=0.3,
    verbose=True,
):
    """Train a model with early stopping.

    Returns:
        Dict with training history and best model state.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    sampler = create_weighted_sampler(Y_train, num_classes=model.fc.out_features)
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                               drop_last=False)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(num_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=use_mixup, mixup_alpha=mixup_alpha,
        )
        val_loss, preds_val, probs_val = validate(model, X_val, Y_val, criterion, device, batch_size)
        scheduler.step()

        # Compute validation metrics
        from sklearn.metrics import accuracy_score, f1_score
        val_acc = accuracy_score(Y_val, preds_val)
        val_f1 = f1_score(Y_val, preds_val, average="macro")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        elapsed = time.time() - t0

        if verbose and (epoch % 5 == 0 or epoch == num_epochs - 1):
            print(f"  Epoch {epoch + 1:3d}/{num_epochs} — "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"time={elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1}")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return {"history": history, "best_state": best_state, "best_val_loss": best_val_loss}


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def optuna_objective(trial, wells, well_names, device, num_classes, window_size):
    """Optuna objective function for hyperparameter optimization.

    Performs leave-one-well-out CV on a subset of wells and returns
    the mean macro F1 score.
    """
    tcn_channels = trial.suggest_categorical(
        "tcn_channels",
        ["[32, 64]", "[32, 64, 128]", "[64, 128]"],
    )
    tcn_channels = json.loads(tcn_channels)

    lstm_hidden = trial.suggest_categorical("lstm_hidden", [64, 128, 256])
    lstm_layers = trial.suggest_int("lstm_layers", 1, 2)
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    trans_fwd = trial.suggest_categorical("trans_fwd", [64, 128, 256])
    trans_layers = trial.suggest_int("trans_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

    n_wells = len(wells)
    fold_f1s = []

    for fold_idx in range(n_wells):
        X_val, Y_val, _ = wells[fold_idx]
        train_parts = [wells[j] for j in range(n_wells) if j != fold_idx]
        X_train = np.concatenate([w[0] for w in train_parts], axis=0)
        Y_train = np.concatenate([w[1] for w in train_parts], axis=0)

        d_input = X_train.shape[2]
        model = Model_TCN_BiLSTM_Transformer(
            d_input=d_input,
            num_classes=num_classes,
            window_size=window_size,
            tcn_channels=tcn_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            nhead=nhead,
            trans_fwd=trans_fwd,
            trans_layers=trans_layers,
            dropout=dropout,
        ).to(device)

        class_counts = np.bincount(Y_train, minlength=num_classes)
        criterion = ClassBalancedFocalLoss(
            samples_per_class=class_counts.tolist(),
            num_classes=num_classes,
        )

        result = train_model(
            model, X_train, Y_train, X_val, Y_val,
            criterion, device,
            num_epochs=30,
            batch_size=batch_size,
            lr=lr,
            patience=8,
            verbose=False,
        )

        _, preds, _ = validate(model, X_val, Y_val, criterion, device, batch_size)
        from sklearn.metrics import f1_score
        fold_f1 = f1_score(Y_val, preds, average="macro", zero_division=0)
        fold_f1s.append(fold_f1)

    return np.mean(fold_f1s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train TCN-BiLSTM-Transformer for lithology identification"
    )
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Directory containing well-log CSV files")
    parser.add_argument("--output_dir", type=str, default="output/",
                        help="Directory to save outputs")
    parser.add_argument("--n_trials", type=int, default=0,
                        help="Number of Optuna trials (0 = skip Optuna)")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="Maximum number of training epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--window_size", type=int, default=41,
                        help="Sliding window size")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of lithology classes")
    parser.add_argument("--blind_well", type=str, default=None,
                        help="Name of the blind/test well (excluded from training)")
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  TCN-BiLSTM-Transformer Lithology Identification — Training")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")

    # ------------------------------------------------------------------
    # Step 1: Load all well data
    # ------------------------------------------------------------------
    print("\n[1/9] Loading well data...")
    well_data, well_names, feature_names, scaler = load_all_wells(
        args.data_dir, window_size=args.window_size
    )
    print(f"  Loaded {len(well_data)} wells: {well_names}")
    print(f"  Feature dimension: {len(feature_names)}")

    # Save scaler
    import joblib
    scaler_path = os.path.join(args.output_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to {scaler_path}")

    # ------------------------------------------------------------------
    # Step 2: Optuna hyperparameter optimization (optional)
    # ------------------------------------------------------------------
    best_params = None
    if args.n_trials > 0:
        print(f"\n[2/9] Running Optuna optimization ({args.n_trials} trials)...")
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            study = optuna.create_study(direction="maximize",
                                          sampler=optuna.samplers.TPESampler(seed=args.seed))
            study.optimize(
                lambda trial: optuna_objective(
                    trial, well_data, well_names, device, args.num_classes, args.window_size
                ),
                n_trials=args.n_trials,
                show_progress_bar=True,
            )

            best_params = study.best_params
            # Parse tcn_channels from string
            if "tcn_channels" in best_params:
                best_params["tcn_channels"] = json.loads(best_params["tcn_channels"])

            print(f"  Best macro F1: {study.best_value:.4f}")
            print(f"  Best params: {json.dumps(best_params, indent=2)}")

            # Save Optuna results
            optuna_path = os.path.join(args.output_dir, "optuna_best_params.json")
            with open(optuna_path, "w") as f:
                json.dump(best_params, f, indent=2)
        except ImportError:
            print("  [WARNING] Optuna not installed. Skipping hyperparameter optimization.")
            best_params = None
    else:
        print("\n[2/9] Skipping Optuna optimization (n_trials=0)")

    # Default hyperparameters if Optuna was skipped
    if best_params is None:
        best_params = {
            "tcn_channels": [32, 64],
            "lstm_hidden": 128,
            "lstm_layers": 1,
            "nhead": 4,
            "trans_fwd": 128,
            "trans_layers": 2,
            "dropout": 0.2,
            "lr": 0.001,
            "batch_size": 512,
        }

    # ------------------------------------------------------------------
    # Step 3: Leave-one-well-out cross-validation
    # ------------------------------------------------------------------
    print("\n[3/9] Running leave-one-well-out cross-validation...")
    n_wells = len(well_data)
    cv_results = []

    for fold_idx in range(n_wells):
        X_val, Y_val, depth_val = well_data[fold_idx]
        train_parts = [well_data[j] for j in range(n_wells) if j != fold_idx]
        X_train = np.concatenate([w[0] for w in train_parts], axis=0)
        Y_train = np.concatenate([w[1] for w in train_parts], axis=0)

        # Oversample thin layers
        X_train_aug, Y_train_aug = oversample_thin_layers(
            X_train.reshape(len(X_train), -1),
            Y_train,
            num_classes=args.num_classes,
        )
        X_train_aug = X_train_aug.reshape(-1, args.window_size, X_train.shape[2])

        d_input = X_train.shape[2]
        model = Model_TCN_BiLSTM_Transformer(
            d_input=d_input,
            num_classes=args.num_classes,
            window_size=args.window_size,
            tcn_channels=best_params.get("tcn_channels", [32, 64]),
            lstm_hidden=best_params.get("lstm_hidden", 128),
            lstm_layers=best_params.get("lstm_layers", 1),
            nhead=best_params.get("nhead", 4),
            trans_fwd=best_params.get("trans_fwd", 128),
            trans_layers=best_params.get("trans_layers", 2),
            dropout=best_params.get("dropout", 0.2),
        ).to(device)

        class_counts = np.bincount(Y_train_aug, minlength=args.num_classes)
        criterion = ClassBalancedFocalLoss(
            samples_per_class=class_counts.tolist(),
            num_classes=args.num_classes,
        )

        print(f"\n  Fold {fold_idx + 1}/{n_wells} — "
              f"val well: {well_names[fold_idx]} — "
              f"train: {len(X_train_aug)}, val: {len(X_val)}")

        train_result = train_model(
            model, X_train_aug, Y_train_aug, X_val, Y_val,
            criterion, device,
            num_epochs=args.n_epochs,
            batch_size=best_params.get("batch_size", 512),
            lr=best_params.get("lr", 0.001),
            patience=15,
        )

        _, preds, probs = validate(model, X_val, Y_val, criterion, device)
        metrics = compute_comprehensive_metrics(Y_val, preds, probs)
        metrics["fold"] = fold_idx
        metrics["val_well"] = well_names[fold_idx]
        cv_results.append(metrics)

        print(f"    accuracy={metrics['accuracy']:.4f}  "
              f"macro_f1={metrics['macro_f1']:.4f}  "
              f"weighted_f1={metrics['weighted_f1']:.4f}")

    # CV summary
    avg_acc = np.mean([r["accuracy"] for r in cv_results])
    avg_mf1 = np.mean([r["macro_f1"] for r in cv_results])
    avg_wf1 = np.mean([r["weighted_f1"] for r in cv_results])
    print(f"\n  CV Average — accuracy={avg_acc:.4f}  "
          f"macro_f1={avg_mf1:.4f}  weighted_f1={avg_wf1:.4f}")

    # Save CV results
    cv_path = os.path.join(args.output_dir, "cv_results.json")
    # Convert numpy arrays to lists for JSON serialization
    cv_results_serializable = []
    for r in cv_results:
        r_copy = {k: v for k, v in r.items() if k != "confusion_matrix"}
        cv_results_serializable.append(r_copy)
    with open(cv_path, "w") as f:
        json.dump(cv_results_serializable, f, indent=2)

    # Plot CV results
    try:
        cv_plot_data = {
            "accuracy": [r.get("accuracy", 0) for r in cv_results],
            "f1": [r.get("macro_f1", 0) for r in cv_results],
        }
        plot_cv_results(cv_plot_data, well_names, TARGET_NAMES, args.num_classes,
                        output_dir=args.output_dir)
    except Exception as e:
        print(f"  [WARNING] Could not plot CV results: {e}")

    # ------------------------------------------------------------------
    # Step 4: Final training on all training wells
    # ------------------------------------------------------------------
    print("\n[4/9] Training final model on all wells...")

    # Determine if there is a blind well to exclude
    blind_idx = None
    if args.blind_well and args.blind_well in well_names:
        blind_idx = well_names.index(args.blind_well)
        print(f"  Excluding blind well: {args.blind_well}")

    if blind_idx is not None:
        train_parts = [well_data[j] for j in range(n_wells) if j != blind_idx]
    else:
        train_parts = well_data

    X_train_all = np.concatenate([w[0] for w in train_parts], axis=0)
    Y_train_all = np.concatenate([w[1] for w in train_parts], axis=0)

    # Oversample thin layers
    X_train_aug, Y_train_aug = oversample_thin_layers(
        X_train_all.reshape(len(X_train_all), -1),
        Y_train_all,
        num_classes=args.num_classes,
    )
    X_train_aug = X_train_aug.reshape(-1, args.window_size, X_train_all.shape[2])

    d_input = X_train_all.shape[2]
    final_model = Model_TCN_BiLSTM_Transformer(
        d_input=d_input,
        num_classes=args.num_classes,
        window_size=args.window_size,
        tcn_channels=best_params.get("tcn_channels", [32, 64]),
        lstm_hidden=best_params.get("lstm_hidden", 128),
        lstm_layers=best_params.get("lstm_layers", 1),
        nhead=best_params.get("nhead", 4),
        trans_fwd=best_params.get("trans_fwd", 128),
        trans_layers=best_params.get("trans_layers", 2),
        dropout=best_params.get("dropout", 0.2),
    ).to(device)

    class_counts = np.bincount(Y_train_aug, minlength=args.num_classes)
    criterion = ClassBalancedFocalLoss(
        samples_per_class=class_counts.tolist(),
        num_classes=args.num_classes,
    )

    # Use last fold's validation set if no blind well
    if blind_idx is not None:
        X_val_final, Y_val_final, _ = well_data[blind_idx]
    else:
        X_val_final, Y_val_final, _ = well_data[-1]

    final_result = train_model(
        final_model, X_train_aug, Y_train_aug, X_val_final, Y_val_final,
        criterion, device,
        num_epochs=args.n_epochs,
        batch_size=best_params.get("batch_size", 512),
        lr=best_params.get("lr", 0.001),
        patience=15,
    )

    # Save best model
    model_path = os.path.join(args.output_dir, "best_model.pt")
    torch.save({
        "model_state_dict": final_result["best_state"],
        "model_config": {
            "d_input": d_input,
            "num_classes": args.num_classes,
            "window_size": args.window_size,
            "tcn_channels": best_params.get("tcn_channels", [32, 64]),
            "lstm_hidden": best_params.get("lstm_hidden", 128),
            "lstm_layers": best_params.get("lstm_layers", 1),
            "nhead": best_params.get("nhead", 4),
            "trans_fwd": best_params.get("trans_fwd", 128),
            "trans_layers": best_params.get("trans_layers", 2),
            "dropout": best_params.get("dropout", 0.2),
        },
        "best_params": best_params,
        "feature_names": feature_names,
    }, model_path)
    print(f"  Model saved to {model_path}")

    # Plot training process
    try:
        hist = final_result["history"]
        plot_training_process(
            hist.get("train_loss", []),
            hist.get("val_acc", []),
            hist.get("val_f1", []),
            output_dir=args.output_dir,
        )
    except Exception as e:
        print(f"  [WARNING] Could not plot training process: {e}")

    # ------------------------------------------------------------------
    # Step 5: Blind well test with post-processing
    # ------------------------------------------------------------------
    print("\n[5/9] Evaluating on blind well...")
    if blind_idx is not None:
        X_blind, Y_blind, depth_blind = well_data[blind_idx]

        _, preds_raw, probs = validate(
            final_model, X_blind, Y_blind, criterion, device
        )

        metrics_raw = compute_comprehensive_metrics(Y_blind, preds_raw, probs)
        print(f"  Raw — accuracy={metrics_raw['accuracy']:.4f}  "
              f"macro_f1={metrics_raw['macro_f1']:.4f}")

        # Post-processing
        preds_post = ensemble_postprocess(preds_raw, probs, args.num_classes)
        metrics_post = compute_comprehensive_metrics(Y_blind, preds_post, probs)
        print(f"  Post-processed — accuracy={metrics_post['accuracy']:.4f}  "
              f"macro_f1={metrics_post['macro_f1']:.4f}")

        # Save blind well results
        blind_results = {
            "raw": {k: v for k, v in metrics_raw.items() if k != "confusion_matrix"},
            "postprocessed": {k: v for k, v in metrics_post.items() if k != "confusion_matrix"},
        }
        blind_path = os.path.join(args.output_dir, "blind_well_results.json")
        with open(blind_path, "w") as f:
            json.dump(blind_results, f, indent=2)

        # Plots
        try:
            plot_confusion_matrix(metrics_raw["confusion_matrix"], TARGET_NAMES,
                                   TARGET_COLORS, output_dir=args.output_dir,
                                   name_suffix="_raw")
            plot_confusion_matrix(metrics_post["confusion_matrix"], TARGET_NAMES,
                                   TARGET_COLORS, output_dir=args.output_dir,
                                   name_suffix="_post")
            plot_depth_facies(depth_blind, Y_blind, preds_post, TARGET_NAMES,
                              TARGET_COLORS, args.num_classes, output_dir=args.output_dir)
            # Compute median-filtered predictions for postprocess comparison
            from scipy.ndimage import median_filter as _median_filter
            preds_med = _median_filter(preds_raw, size=3, mode="nearest").astype(np.int64)

            plot_postprocess_comparison(
                Y_blind, preds_raw, preds_med, preds_post, probs, TARGET_NAMES,
                TARGET_COLORS, args.num_classes, output_dir=args.output_dir
            )
        except Exception as e:
            print(f"  [WARNING] Could not generate blind well plots: {e}")
    else:
        print("  No blind well specified. Skipping blind well evaluation.")

    # ------------------------------------------------------------------
    # Step 6: Comprehensive evaluation on validation set
    # ------------------------------------------------------------------
    print("\n[6/9] Generating evaluation plots...")
    _, preds_final, probs_final = validate(
        final_model, X_val_final, Y_val_final, criterion, device
    )
    metrics_final = compute_comprehensive_metrics(Y_val_final, preds_final, probs_final)

    try:
        plot_per_class_metrics(metrics_final, TARGET_NAMES, TARGET_COLORS,
                                output_dir=args.output_dir)
        plot_roc_pr_curves(Y_val_final, probs_final, TARGET_NAMES,
                           TARGET_COLORS, args.num_classes,
                           output_dir=args.output_dir)
        plot_error_analysis(Y_val_final, preds_final, TARGET_NAMES,
                            args.num_classes, output_dir=args.output_dir)
    except Exception as e:
        print(f"  [WARNING] Could not generate evaluation plots: {e}")

    # ------------------------------------------------------------------
    # Step 7: Interpretability analysis
    # ------------------------------------------------------------------
    print("\n[7/9] Running interpretability analysis...")
    try:
        plot_integrated_gradients(final_model, X_val_final, Y_val_final,
                                   feature_names, device, args.window_size,
                                   args.num_classes, TARGET_NAMES, TARGET_COLORS,
                                   output_dir=args.output_dir)
        plot_attention_weights(final_model, X_val_final, Y_val_final, device,
                                args.window_size, output_dir=args.output_dir)
        plot_saliency(final_model, X_val_final, Y_val_final, feature_names,
                      device, output_dir=args.output_dir)
        plot_temporal_importance(final_model, X_val_final, Y_val_final, device,
                                 args.window_size, output_dir=args.output_dir)
        plot_shap_summary(final_model, X_val_final, Y_val_final,
                          feature_names, device, args.window_size,
                          args.num_classes, TARGET_NAMES, TARGET_COLORS,
                          output_dir=args.output_dir)
    except Exception as e:
        print(f"  [WARNING] Could not complete interpretability analysis: {e}")

    # ------------------------------------------------------------------
    # Step 8: (Removed unused architecture/gradient plots)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Step 9: Save final results
    # ------------------------------------------------------------------
    print("\n[9/9] Saving final results...")
    final_results = {
        "best_params": best_params,
        "cv_average": {
            "accuracy": float(avg_acc),
            "macro_f1": float(avg_mf1),
            "weighted_f1": float(avg_wf1),
        },
        "feature_names": feature_names,
        "num_wells": len(well_data),
        "window_size": args.window_size,
        "num_classes": args.num_classes,
        "d_input": int(d_input),
    }

    results_path = os.path.join(args.output_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"  Results saved to {results_path}")

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
