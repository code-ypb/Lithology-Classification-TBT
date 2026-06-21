"""
Visualization module for well-log lithology identification.

Provides publication-quality plotting utilities for training curves,
confusion matrices, per-class metrics, ROC/PR curves, depth profiles,
error analysis, gradient flow, model architecture, interpretability
(IG, attention, saliency, temporal, permutation), post-processing
comparison, comprehensive metrics dashboards, and cross-validation
results.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


# ===================================================================
# 1. save_fig
# ===================================================================
def save_fig(
    fig: plt.Figure,
    name: str,
    output_dir: str | Path,
    dpi: int = 300,
) -> None:
    """Save a figure to *output_dir* and close it.

    Args:
        fig: Matplotlib Figure object.
        name: File name (without directory). Extension defaults to ``.png``.
        output_dir: Destination directory. Created if it does not exist.
        dpi: Resolution in dots per inch.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not name.lower().endswith((".png", ".pdf", ".svg", ".eps")):
        name += ".png"
    fig.savefig(output_dir / name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# 2. plot_training_process
# ===================================================================
def plot_training_process(
    loss_h: list[float],
    acc_h: list[float],
    f1_h: list[float],
    output_dir: str | Path,
) -> None:
    """Plot training loss, accuracy, and macro-F1 over epochs.

    Args:
        loss_h: Loss values per epoch.
        acc_h: Accuracy values per epoch.
        f1_h: Macro-F1 values per epoch.
        output_dir: Directory to save the figure.
    """
    epochs = range(1, len(loss_h) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(epochs, loss_h, "o-", color="#e74c3c", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, acc_h, "s-", color="#2ecc71", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training Accuracy")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, f1_h, "D-", color="#3498db", markersize=3)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].set_title("Training Macro F1")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "training_process", output_dir)


# ===================================================================
# 3. plot_confusion_matrix
# ===================================================================
def plot_confusion_matrix(
    cm: np.ndarray,
    target_names: list[str],
    target_colors: list[str | tuple],
    output_dir: str | Path,
    name_suffix: str = "",
) -> None:
    """Plot confusion matrix in three panels: raw, recall-norm, precision-norm.

    Args:
        cm: Confusion matrix of shape (n_classes, n_classes).
        target_names: Class names.
        target_colors: Colours for each class (used for annotation only).
        output_dir: Directory to save the figure.
        name_suffix: Optional suffix appended to the file name.
    """
    n = cm.shape[0]
    labels = target_names[:n]

    # Normalised variants
    cm_recall = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    cm_prec = cm.astype(float) / (cm.sum(axis=0, keepdims=True) + 1e-12)

    titles = ["Raw Counts", "Normalised by Recall (Row)", "Normalised by Precision (Col)"]
    matrices = [cm, cm_recall, cm_prec]
    fmts = ["d", ".2f", ".2f"]

    fig, axes = plt.subplots(1, 3, figsize=(6 * n, 5))
    for ax, mat, title, fmt in zip(axes, matrices, titles, fmts):
        sns.heatmap(
            mat, ax=ax, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            square=True, cbar_kws={"shrink": 0.8},
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

    fig.tight_layout()
    fname = "confusion_matrix"
    if name_suffix:
        fname += f"_{name_suffix}"
    save_fig(fig, fname, output_dir)


# ===================================================================
# 4. plot_per_class_metrics
# ===================================================================
def plot_per_class_metrics(
    report_dict: dict,
    target_names: list[str],
    target_colors: list[str | tuple],
    output_dir: str | Path,
    name_suffix: str = "",
) -> None:
    """Grouped bar chart of Precision / Recall / F1 per class.

    Args:
        report_dict: Dictionary with per-class metrics (as returned by
            :func:`compute_comprehensive_metrics` under the ``per_class`` key).
        target_names: Class names.
        target_colors: Bar colours for each class.
        output_dir: Directory to save the figure.
        name_suffix: Optional suffix appended to the file name.
    """
    n = len(target_names)
    metrics_keys = ["precision", "recall", "f1_score"]
    metrics_labels = ["Precision", "Recall", "F1"]

    values = {k: [] for k in metrics_keys}
    for name in target_names:
        cls_info = report_dict.get(name, {})
        for k in metrics_keys:
            values[k].append(cls_info.get(k, 0.0))

    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5))
    for i, (k, label) in enumerate(zip(metrics_keys, metrics_labels)):
        bars = ax.bar(x + i * width, values[k], width, label=label, alpha=0.85)
        for bar, val in zip(bars, values[k]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(target_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fname = "per_class_metrics"
    if name_suffix:
        fname += f"_{name_suffix}"
    save_fig(fig, fname, output_dir)


# ===================================================================
# 5. plot_roc_pr_curves
# ===================================================================
def plot_roc_pr_curves(
    Y_test: np.ndarray,
    Y_proba: np.ndarray,
    target_names: list[str],
    target_colors: list[str | tuple],
    num_classes: int,
    output_dir: str | Path,
    name_suffix: str = "",
) -> None:
    """Plot ROC and Precision-Recall curves side by side.

    Args:
        Y_test: Ground-truth labels, shape (n_samples,).
        Y_proba: Predicted probabilities, shape (n_samples, num_classes).
        target_names: Class names.
        target_colors: Colour for each class curve.
        num_classes: Number of classes.
        output_dir: Directory to save the figure.
        name_suffix: Optional suffix appended to the file name.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- ROC ---
    ax_roc = axes[0]
    for i in range(num_classes):
        binary_true = (Y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, Y_proba[:, i])
        roc_auc_val = auc(fpr, tpr)
        color = target_colors[i] if i < len(target_colors) else None
        label = f"{target_names[i]} (AUC={roc_auc_val:.3f})" if i < len(target_names) else f"Class {i}"
        ax_roc.plot(fpr, tpr, color=color, label=label, lw=1.5)
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves (One-vs-Rest)")
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.grid(True, alpha=0.3)

    # --- PR ---
    ax_pr = axes[1]
    for i in range(num_classes):
        binary_true = (Y_test == i).astype(int)
        prec, rec, _ = precision_recall_curve(binary_true, Y_proba[:, i])
        pr_auc_val = auc(rec, prec)
        color = target_colors[i] if i < len(target_colors) else None
        label = f"{target_names[i]} (AUC={pr_auc_val:.3f})" if i < len(target_names) else f"Class {i}"
        ax_pr.plot(rec, prec, color=color, label=label, lw=1.5)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves")
    ax_pr.legend(loc="lower left", fontsize=8)
    ax_pr.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = "roc_pr_curves"
    if name_suffix:
        fname += f"_{name_suffix}"
    save_fig(fig, fname, output_dir)


# ===================================================================
# 6. plot_depth_facies
# ===================================================================
def plot_depth_facies(
    depth: np.ndarray,
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    target_names: list[str],
    target_colors: list[str | tuple],
    num_classes: int,
    output_dir: str | Path,
    Y_proba: np.ndarray | None = None,
    name_suffix: str = "",
) -> None:
    """Depth plot with true / predicted / match / confidence panels.

    Args:
        depth: Depth values, shape (n_samples,).
        Y_true: Ground-truth labels, shape (n_samples,).
        Y_pred: Predicted labels, shape (n_samples,).
        target_names: Class names.
        target_colors: Colour for each class.
        num_classes: Number of classes.
        output_dir: Directory to save the figure.
        Y_proba: Predicted probabilities, shape (n_samples, num_classes).
            If provided, a confidence panel is drawn.
        name_suffix: Optional suffix appended to the file name.
    """
    n_panels = 4 if Y_proba is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 10), sharey=True)

    color_map = {i: target_colors[i] for i in range(min(num_classes, len(target_colors)))}
    match = (Y_true == Y_pred).astype(int)

    panel_data = [
        (Y_true, "True Facies"),
        (Y_pred, "Predicted Facies"),
        (match, "Match (1=correct)"),
    ]
    if Y_proba is not None:
        confidence = np.max(Y_proba, axis=1)
        panel_data.append((confidence, "Confidence"))

    for ax, (data, title) in zip(axes, panel_data):
        if data.dtype in (np.float64, np.float32, float):
            ax.scatter(data, depth, c=data, cmap="viridis", s=1, marker="s")
        else:
            colors_arr = np.array(
                [color_map.get(int(v), "#888888") for v in data]
            )
            for cls_id in range(num_classes):
                mask = data == cls_id
                if not np.any(mask):
                    continue
                ax.scatter(
                    np.full(np.sum(mask), cls_id), depth[mask],
                    c=color_map.get(cls_id, "#888888"), s=1, marker="s",
                    label=target_names[cls_id] if cls_id < len(target_names) else str(cls_id),
                )
        ax.set_title(title)
        ax.set_xlabel("Class" if title != "Confidence" else "Probability")

    axes[0].set_ylabel("Depth")
    axes[0].invert_yaxis()

    handles = [
        mpatches.Patch(color=color_map.get(i, "#888888"), label=target_names[i] if i < len(target_names) else str(i))
        for i in range(num_classes)
    ]
    axes[-1].legend(handles=handles, loc="lower right", fontsize=7)

    fig.tight_layout()
    fname = "depth_facies"
    if name_suffix:
        fname += f"_{name_suffix}"
    save_fig(fig, fname, output_dir)


# ===================================================================
# 7. plot_error_analysis
# ===================================================================
def plot_error_analysis(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    target_names: list[str],
    num_classes: int,
    output_dir: str | Path,
    name_suffix: str = "",
) -> None:
    """Horizontal bar chart of top misclassification pairs.

    Args:
        Y_true: Ground-truth labels, shape (n_samples,).
        Y_pred: Predicted labels, shape (n_samples,).
        target_names: Class names.
        num_classes: Number of classes.
        output_dir: Directory to save the figure.
        name_suffix: Optional suffix appended to the file name.
    """
    cm = confusion_matrix(Y_true, Y_pred, labels=list(range(num_classes)))
    np.fill_diagonal(cm, 0)

    pairs: list[tuple[int, int, int]] = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                pairs.append((i, j, int(cm[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    top_k = min(15, len(pairs))
    if top_k == 0:
        return

    top_pairs = pairs[:top_k]
    labels = [
        f"{target_names[i] if i < len(target_names) else i} → "
        f"{target_names[j] if j < len(target_names) else j}"
        for i, j, _ in top_pairs
    ]
    counts = [c for _, _, c in top_pairs]

    fig, ax = plt.subplots(figsize=(8, max(3, top_k * 0.4)))
    y_pos = np.arange(top_k)
    ax.barh(y_pos, counts, color="#e74c3c", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Misclassification Count")
    ax.set_title("Top Misclassification Pairs (True → Predicted)")
    ax.grid(axis="x", alpha=0.3)

    for idx, c in enumerate(counts):
        ax.text(c + 0.5, idx, str(c), va="center", fontsize=9)

    fig.tight_layout()
    fname = "error_analysis"
    if name_suffix:
        fname += f"_{name_suffix}"
    save_fig(fig, fname, output_dir)


# ===================================================================
# 8. plot_integrated_gradients
# ===================================================================
def _integrated_gradients_compute(
    model: torch.nn.Module,
    x_input: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor | None = None,
    steps: int = 50,
) -> torch.Tensor:
    """Compute Integrated Gradients for a single input.

    Args:
        model: PyTorch model.
        x_input: Input tensor of shape (1, seq_len, n_features).
        target_class: Target class index.
        baseline: Baseline input (zeros if None).
        steps: Number of interpolation steps.

    Returns:
        Tensor of shape (seq_len, n_features) with IG attributions.
    """
    if baseline is None:
        baseline = torch.zeros_like(x_input)
    scaled_inputs = [baseline + (float(i) / steps) * (x_input - baseline) for i in range(steps + 1)]
    scaled_inputs = torch.cat(scaled_inputs, dim=0)
    scaled_inputs.requires_grad_(True)
    scaled_inputs.retain_grad()

    logits = model(scaled_inputs)
    targets = logits[:, target_class].sum()
    targets.backward()

    grads = scaled_inputs.grad  # (steps+1, seq_len, n_features)
    avg_grads = grads.mean(dim=0)  # (seq_len, n_features)
    ig = (x_input - baseline) * avg_grads
    return ig.squeeze(0)  # (seq_len, n_features)


def plot_integrated_gradients(
    model: torch.nn.Module,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    all_features_names: list[str],
    device: torch.device | str,
    window_size: int,
    num_classes: int,
    target_names: list[str],
    target_colors: list[str | tuple],
    output_dir: str | Path,
) -> None:
    """Integrated Gradients feature importance bar chart.

    Computes IG for a random subset of test samples, averages the
    absolute attributions per feature, and displays a grouped bar
    chart (one group per class).

    Args:
        model: Trained PyTorch model.
        X_test: Test features, shape (n_samples, seq_len, n_features).
        Y_test: Test labels, shape (n_samples,).
        all_features_names: Feature names.
        device: Torch device.
        window_size: Sequence length of each window.
        num_classes: Number of classes.
        target_names: Class names.
        target_colors: Colours for each class.
        output_dir: Directory to save the figure.
    """
    model.eval()
    n_features = len(all_features_names)
    n_samples = min(100, len(X_test))
    indices = np.random.choice(len(X_test), n_samples, replace=False)

    # Accumulate per-class IG
    ig_accum = np.zeros((num_classes, n_features))
    counts = np.zeros(num_classes)

    for idx in indices:
        x_np = X_test[idx]
        y = int(Y_test[idx])
        x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
        x_tensor.requires_grad_(True)

        with torch.enable_grad():
            ig = _integrated_gradients_compute(model, x_tensor, y, steps=30)

        ig_accum[y] += ig.abs().mean(dim=0).detach().cpu().numpy()
        counts[y] += 1

    for c in range(num_classes):
        if counts[c] > 0:
            ig_accum[c] /= counts[c]

    # Plot grouped bar chart
    x = np.arange(n_features)
    width = 0.8 / num_classes
    fig, ax = plt.subplots(figsize=(max(8, n_features * 0.7), 5))

    for c in range(num_classes):
        color = target_colors[c] if c < len(target_colors) else None
        label = target_names[c] if c < len(target_names) else f"Class {c}"
        ax.bar(x + c * width, ig_accum[c], width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x + width * (num_classes - 1) / 2)
    ax.set_xticklabels(all_features_names, rotation=45, ha="right")
    ax.set_ylabel("Mean |IG Attribution|")
    ax.set_title("Integrated Gradients Feature Importance")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "integrated_gradients", output_dir)


# ===================================================================
# 11. plot_attention_weights
# ===================================================================
def plot_attention_weights(
    model: torch.nn.Module,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    device: torch.device | str,
    window_size: int,
    output_dir: str | Path,
) -> None:
    """Mean self-attention heatmap across test samples.

    Expects the model to have an ``get_attention_weights`` method that
    returns a tensor of shape ``(batch, n_heads, seq_len, seq_len)``.
    If not available, falls back to a placeholder message.

    Args:
        model: Trained model with ``get_attention_weights`` method.
        X_test: Test features, shape (n_samples, seq_len, n_features).
        Y_test: Test labels, shape (n_samples,).
        device: Torch device.
        window_size: Sequence length.
        output_dir: Directory to save the figure.
    """
    model.eval()
    if not hasattr(model, "get_attention_weights"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Model does not expose get_attention_weights()",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.axis("off")
        save_fig(fig, "attention_weights_unavailable", output_dir)
        return

    n_samples = min(200, len(X_test))
    attn_sum = None
    count = 0

    for start in range(0, n_samples, 32):
        end = min(start + 32, n_samples)
        x_batch = torch.tensor(X_test[start:end], dtype=torch.float32, device=device)
        with torch.no_grad():
            attn = model.get_attention_weights(x_batch)  # (B, H, S, S)
        attn = attn.mean(dim=1).cpu().numpy()  # (B, S, S)
        if attn_sum is None:
            attn_sum = attn.sum(axis=0)
        else:
            attn_sum += attn.sum(axis=0)
        count += attn.shape[0]

    attn_mean = attn_sum / count

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(attn_mean, ax=ax, cmap="YlOrRd", square=True,
                xticklabels=window_size // 4, yticklabels=window_size // 4,
                cbar_kws={"shrink": 0.8, "label": "Attention Weight"})
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title("Mean Self-Attention Weights")

    fig.tight_layout()
    save_fig(fig, "attention_weights", output_dir)


# ===================================================================
# 12. plot_saliency
# ===================================================================
def plot_saliency(
    model: torch.nn.Module,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    all_features_names: list[str],
    device: torch.device | str,
    output_dir: str | Path,
) -> None:
    """Saliency / gradient sensitivity bar chart.

    Computes the mean absolute gradient of the output logit w.r.t. each
    input feature across a subset of test samples.

    Args:
        model: Trained PyTorch model.
        X_test: Test features, shape (n_samples, seq_len, n_features).
        Y_test: Test labels, shape (n_samples,).
        all_features_names: Feature names.
        device: Torch device.
        output_dir: Directory to save the figure.
    """
    model.eval()
    n_features = len(all_features_names)
    n_samples = min(200, len(X_test))
    saliency = np.zeros(n_features)

    for idx in range(n_samples):
        x_tensor = torch.tensor(X_test[idx], dtype=torch.float32, device=device).unsqueeze(0)
        x_tensor.requires_grad_(True)
        y = int(Y_test[idx])

        with torch.enable_grad():
            logits = model(x_tensor)
            logits[0, y].backward()

        grad = x_tensor.grad.abs().mean(dim=1).squeeze(0).cpu().numpy()  # (n_features,)
        saliency += grad

    saliency /= n_samples

    fig, ax = plt.subplots(figsize=(max(8, n_features * 0.6), 5))
    x = np.arange(n_features)
    ax.bar(x, saliency, color="#8e44ad", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(all_features_names, rotation=45, ha="right")
    ax.set_ylabel("Mean |Gradient|")
    ax.set_title("Saliency / Gradient Sensitivity")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "saliency", output_dir)


# ===================================================================
# 13. plot_temporal_importance
# ===================================================================
def plot_temporal_importance(
    model: torch.nn.Module,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    device: torch.device | str,
    window_size: int,
    output_dir: str | Path,
) -> None:
    """Bar chart of gradient magnitude across window positions.

    Shows which temporal positions in the input window contribute most
    to the model's prediction.

    Args:
        model: Trained PyTorch model.
        X_test: Test features, shape (n_samples, seq_len, n_features).
        Y_test: Test labels, shape (n_samples,).
        device: Torch device.
        window_size: Sequence length.
        output_dir: Directory to save the figure.
    """
    model.eval()
    n_samples = min(200, len(X_test))
    temporal_grad = np.zeros(window_size)

    for idx in range(n_samples):
        x_tensor = torch.tensor(X_test[idx], dtype=torch.float32, device=device).unsqueeze(0)
        x_tensor.requires_grad_(True)
        y = int(Y_test[idx])

        with torch.enable_grad():
            logits = model(x_tensor)
            logits[0, y].backward()

        # Mean absolute gradient across features per time step
        grad = x_tensor.grad.abs().mean(dim=2).squeeze(0).cpu().numpy()  # (seq_len,)
        temporal_grad += grad

    temporal_grad /= n_samples

    fig, ax = plt.subplots(figsize=(max(8, window_size * 0.15), 5))
    ax.bar(range(window_size), temporal_grad, color="#e67e22", alpha=0.85)
    ax.set_xlabel("Window Position")
    ax.set_ylabel("Mean |Gradient|")
    ax.set_title("Temporal Importance (Gradient Magnitude per Position)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "temporal_importance", output_dir)


# ===================================================================
# 14. plot_postprocess_comparison
# ===================================================================
def plot_postprocess_comparison(
    Y_test: np.ndarray,
    preds_raw: np.ndarray,
    preds_med: np.ndarray,
    preds_ensemble: np.ndarray,
    Y_proba: np.ndarray,
    target_names: list[str],
    target_colors: list[str | tuple],
    num_classes: int,
    output_dir: str | Path,
) -> None:
    """3-panel normalised confusion matrices for raw / median / ensemble.

    Args:
        Y_test: Ground-truth labels, shape (n_samples,).
        preds_raw: Raw model predictions, shape (n_samples,).
        preds_med: Median-filtered predictions, shape (n_samples,).
        preds_ensemble: Ensemble predictions, shape (n_samples,).
        Y_proba: Predicted probabilities, shape (n_samples, num_classes).
        target_names: Class names.
        target_colors: Colours for each class.
        num_classes: Number of classes.
        output_dir: Directory to save the figure.
    """
    labels = list(range(num_classes))
    preds_list = [preds_raw, preds_med, preds_ensemble]
    titles = ["Raw Predictions", "Median Filtered", "Ensemble"]

    fig, axes = plt.subplots(1, 3, figsize=(6 * num_classes, 5))
    for ax, preds, title in zip(axes, preds_list, titles):
        cm = confusion_matrix(Y_test, preds, labels=labels)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
        sns.heatmap(
            cm_norm, ax=ax, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=target_names[:num_classes],
            yticklabels=target_names[:num_classes],
            square=True, cbar_kws={"shrink": 0.8},
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

    fig.tight_layout()
    save_fig(fig, "postprocess_comparison", output_dir)


# ===================================================================
# 16. plot_cv_results
# ===================================================================
def plot_cv_results(
    cv_results: dict[str, list[float]],
    well_names: list[str],
    target_names: list[str],
    num_classes: int,
    output_dir: str | Path,
) -> None:
    """Cross-validation results: accuracy/F1 per fold, comparison bar, F1 heatmap.

    Args:
        cv_results: Dictionary with keys ``"accuracy"`` and ``"f1"``,
            each mapping to a list of per-fold scores. May also contain
            ``"per_class_f1"`` with shape (n_folds, num_classes).
        well_names: Name of each fold / well.
        target_names: Class names.
        num_classes: Number of classes.
        output_dir: Directory to save the figure.
    """
    accs = cv_results.get("accuracy", [])
    f1s = cv_results.get("f1", [])
    n_folds = len(accs)

    fig, axes = plt.subplots(1, 3, figsize=(6 * max(n_folds, 1), 5))

    # --- Panel 1: Accuracy per fold ---
    ax = axes[0]
    x = np.arange(n_folds)
    ax.bar(x, accs, color="#3498db", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(well_names[:n_folds], rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy per Fold")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(accs):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

    # --- Panel 2: F1 per fold ---
    ax = axes[1]
    ax.bar(x, f1s, color="#e74c3c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(well_names[:n_folds], rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 per Fold")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(f1s):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

    # --- Panel 3: Per-class F1 heatmap ---
    ax = axes[2]
    per_class_f1 = cv_results.get("per_class_f1", None)
    if per_class_f1 is not None and len(per_class_f1) > 0:
        per_class_f1 = np.array(per_class_f1)  # (n_folds, num_classes)
        sns.heatmap(
            per_class_f1, ax=ax, annot=True, fmt=".2f", cmap="YlGn",
            xticklabels=target_names[:per_class_f1.shape[1]],
            yticklabels=well_names[:per_class_f1.shape[0]],
            vmin=0, vmax=1, cbar_kws={"shrink": 0.8},
        )
        ax.set_xlabel("Class")
        ax.set_ylabel("Fold")
        ax.set_title("Per-Class F1 Heatmap")
    else:
        ax.text(0.5, 0.5, "No per-class F1 data", ha="center", va="center",
                fontsize=12, transform=ax.transAxes)
        ax.axis("off")
        ax.set_title("Per-Class F1 Heatmap")

    fig.tight_layout()
    save_fig(fig, "cv_results", output_dir)


# ===================================================================
# 17. plot_shap_summary
# ===================================================================
def plot_shap_summary(
    model: torch.nn.Module,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    all_features_names: list[str],
    device: torch.device | str,
    window_size: int,
    num_classes: int,
    target_names: list[str],
    target_colors: list[str | tuple],
    output_dir: str | Path,
    max_samples: int = 200,
) -> None:
    """SHAP feature importance summary with per-class bar chart and beeswarm.

    Uses a model wrapper that extracts the centre-position logits for SHAP's
    KernelExplainer. Produces two figures:
    1. A grouped bar chart of mean |SHAP value| per feature per class.
    2. A beeswarm plot showing SHAP value distributions for the top features.

    Args:
        model: Trained PyTorch model.
        X_test: Test data, shape (n_samples, seq_len, n_features).
        Y_test: Test labels, shape (n_samples,).
        all_features_names: Feature column names.
        device: Torch device.
        window_size: Sequence length.
        num_classes: Number of classes.
        target_names: Class names.
        target_colors: Colours for each class.
        output_dir: Directory to save figures.
        max_samples: Maximum number of samples for SHAP computation.
    """
    import shap

    model.eval()
    n_samples = min(max_samples, len(X_test))
    indices = np.random.choice(len(X_test), n_samples, replace=False)

    # Extract centre-position features for SHAP
    centre = window_size // 2
    X_centre = X_test[indices, centre, :]  # (n_samples, n_features)
    Y_centre = Y_test[indices]

    # Model wrapper: takes (n_samples, n_features), returns centre logits
    class CentreLogitsWrapper:
        def __init__(self, model, window_size, device):
            self.model = model
            self.window_size = window_size
            self.device = device

        def __call__(self, x):
            x = np.asarray(x, dtype=np.float32)
            n = x.shape[0]
            # Broadcast centre features to full window
            x_full = np.broadcast_to(
                x[:, np.newaxis, :], (n, self.window_size, x.shape[1])
            ).copy()
            with torch.no_grad():
                x_tensor = torch.tensor(x_full, dtype=torch.float32, device=self.device)
                logits = self.model(x_tensor)  # (n, num_classes)
            return logits.cpu().numpy()

    wrapper = CentreLogitsWrapper(model, window_size, device)

    # Use a subset as background
    bg_size = min(50, n_samples)
    bg_indices = np.random.choice(n_samples, bg_size, replace=False)
    background = X_centre[bg_indices]

    # Compute SHAP values
    explainer = shap.KernelExplainer(wrapper, background)
    shap_values = explainer.shap_values(X_centre, nsamples=100)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        # List of (n_samples, n_features) arrays, one per class
        shap_array = np.stack(shap_values, axis=-1)  # (n_samples, n_features, n_classes)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_array = shap_values  # already (n_samples, n_features, n_classes)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        # Single output: (n_samples, n_features) -> add class dimension
        shap_array = shap_values[:, :, np.newaxis]
    else:
        print(f"  [SHAP] Unexpected shap_values format: {type(shap_values)}")
        return

    # --- Figure 1: Grouped bar chart of mean |SHAP| per class ---
    mean_abs_shap = np.abs(shap_array).mean(axis=0)  # (n_features, n_classes)
    n_features = len(all_features_names)
    x = np.arange(n_features)
    width = 0.8 / num_classes
    fig, ax = plt.subplots(figsize=(max(8, n_features * 0.7), 5))

    for c in range(num_classes):
        color = target_colors[c] if c < len(target_colors) else None
        label = target_names[c] if c < len(target_names) else f"Class {c}"
        ax.bar(x + c * width, mean_abs_shap[:, c], width,
               label=label, color=color, alpha=0.85)

    ax.set_xticks(x + width * (num_classes - 1) / 2)
    ax.set_xticklabels(all_features_names, rotation=45, ha="right")
    ax.set_ylabel("Mean |SHAP Value|")
    ax.set_title("SHAP Feature Importance by Class")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "shap_summary", output_dir)

    # --- Figure 2: Beeswarm plot for overall importance ---
    # Use the max SHAP value across classes for overall importance
    overall_shap = np.abs(shap_array).max(axis=-1)  # (n_samples, n_features)
    fig2, ax2 = plt.subplots(figsize=(max(8, n_features * 0.6), 6))

    # Sort features by mean importance
    sorted_idx = np.argsort(overall_shap.mean(axis=0))[::-1]
    sorted_names = [all_features_names[i] for i in sorted_idx]

    # Create beeswarm-like scatter
    for rank, fi in enumerate(sorted_idx):
        vals = overall_shap[:, fi]
        # Jitter y positions for beeswarm effect
        y_jitter = np.random.normal(rank, 0.15, size=len(vals))
        ax2.scatter(vals, y_jitter, s=8, alpha=0.5, color="#3498db")

    ax2.set_yticks(range(n_features))
    ax2.set_yticklabels(sorted_names)
    ax2.invert_yaxis()
    ax2.set_xlabel("|SHAP Value|")
    ax2.set_title("SHAP Beeswarm Summary")
    ax2.grid(axis="x", alpha=0.3)
    fig2.tight_layout()
    save_fig(fig2, "shap_beeswarm", output_dir)
