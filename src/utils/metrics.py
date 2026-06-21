"""
Evaluation metrics module for well-log lithology identification.

Provides model inference helpers and comprehensive classification metric
computation including per-class statistics, confusion matrix, and
area-under-curve measures.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    X_data: np.ndarray,
    Y_data: np.ndarray,
    device: torch.device | str = "cpu",
    batch_size: int = 512,
    use_amp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a PyTorch model on numpy data.

    Runs inference in batches, optionally using automatic mixed precision
    (AMP) for faster evaluation on compatible GPUs.

    Args:
        model: A trained PyTorch model that outputs raw logits of shape
            (batch_size, n_classes).
        X_data: Input features as a numpy array of shape
            (n_samples, seq_len, n_features).
        Y_data: Ground-truth labels as a numpy array of shape (n_samples,).
        device: Torch device string or object. Defaults to "cpu".
        batch_size: Mini-batch size for inference. Defaults to 512.
        use_amp: Whether to use automatic mixed precision. Defaults to False.

    Returns:
        tuple: (predictions, probabilities)
            - predictions: np.ndarray of shape (n_samples,) with predicted
              class indices.
            - probabilities: np.ndarray of shape (n_samples, n_classes) with
              softmax probabilities.
    """
    model.eval()
    model.to(device)

    n_samples = X_data.shape[0]
    n_classes = None  # determined from first batch output

    all_probs: list[np.ndarray] = []

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        x_batch = torch.tensor(
            X_data[start:end], dtype=torch.float32, device=device
        )

        if use_amp and device != "cpu":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x_batch)
        else:
            logits = model(x_batch)

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        if n_classes is None:
            n_classes = probs.shape[1]
        all_probs.append(probs)

    probabilities = np.concatenate(all_probs, axis=0)
    predictions = np.argmax(probabilities, axis=1)

    return predictions, probabilities


def compute_comprehensive_metrics(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    Y_proba: np.ndarray | None = None,
    target_names: list[str] | None = None,
) -> dict:
    """Compute comprehensive classification metrics.

    Calculates overall accuracy, macro/weighted F1, precision, recall,
    balanced accuracy, Cohen's kappa, MCC, per-class metrics, confusion
    matrix, and (if probabilities are provided) ROC-AUC and PR-AUC.

    Args:
        Y_true: Ground-truth labels, shape (n_samples,).
        Y_pred: Predicted labels, shape (n_samples,).
        Y_proba: Predicted class probabilities, shape (n_samples, n_classes).
            If provided, ROC-AUC and PR-AUC are computed. Defaults to None.
        target_names: Optional list of class names for per-class reporting.
            If None, class indices are used as names.

    Returns:
        dict: Dictionary with the following keys:
            - "accuracy": Overall accuracy
            - "balanced_accuracy": Balanced accuracy
            - "macro_f1": Macro-averaged F1 score
            - "weighted_f1": Weighted F1 score
            - "macro_precision": Macro-averaged precision
            - "weighted_precision": Weighted precision
            - "macro_recall": Macro-averaged recall
            - "weighted_recall": Weighted recall
            - "cohen_kappa": Cohen's kappa coefficient
            - "mcc": Matthews correlation coefficient
            - "confusion_matrix": Confusion matrix (2D array)
            - "per_class": Dict mapping class identifier to per-class metrics
            - "roc_auc_ovr": ROC-AUC one-vs-rest (if Y_proba given)
            - "roc_auc_ovo": ROC-AUC one-vs-one (if Y_proba given)
            - "pr_auc": Dict of per-class PR-AUC (if Y_proba given)
    """
    # Determine class set
    classes = sorted(set(Y_true.tolist()) | set(Y_pred.tolist()))
    n_classes = len(classes)

    if target_names is None:
        target_names = [str(c) for c in classes]

    # ------------------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------------------
    metrics: dict = {
        "accuracy": float(accuracy_score(Y_true, Y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(Y_true, Y_pred)),
        "macro_f1": float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(Y_true, Y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(Y_true, Y_pred, average="weighted", zero_division=0)),
        "macro_recall": float(recall_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "weighted_recall": float(recall_score(Y_true, Y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(Y_true, Y_pred)),
        "mcc": float(matthews_corrcoef(Y_true, Y_pred)),
        "confusion_matrix": confusion_matrix(Y_true, Y_pred, labels=classes),
    }

    # ------------------------------------------------------------------
    # Per-class metrics
    # ------------------------------------------------------------------
    per_class_f1 = f1_score(Y_true, Y_pred, average=None, labels=classes, zero_division=0)
    per_class_precision = precision_score(Y_true, Y_pred, average=None, labels=classes, zero_division=0)
    per_class_recall = recall_score(Y_true, Y_pred, average=None, labels=classes, zero_division=0)

    per_class_dict: dict = {}
    for idx, cls in enumerate(classes):
        name = target_names[idx] if idx < len(target_names) else str(cls)
        per_class_dict[name] = {
            "precision": float(per_class_precision[idx]),
            "recall": float(per_class_recall[idx]),
            "f1_score": float(per_class_f1[idx]),
            "support": int(np.sum(Y_true == cls)),
        }
    metrics["per_class"] = per_class_dict

    # ------------------------------------------------------------------
    # AUC metrics (require probabilities)
    # ------------------------------------------------------------------
    if Y_proba is not None:
        try:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(Y_true, Y_proba, multi_class="ovr", average="macro", labels=classes)
            )
        except ValueError:
            metrics["roc_auc_ovr"] = None

        try:
            metrics["roc_auc_ovo"] = float(
                roc_auc_score(Y_true, Y_proba, multi_class="ovo", average="macro", labels=classes)
            )
        except ValueError:
            metrics["roc_auc_ovo"] = None

        # Per-class PR-AUC
        from sklearn.metrics import average_precision_score

        pr_auc_dict: dict = {}
        for idx, cls in enumerate(classes):
            name = target_names[idx] if idx < len(target_names) else str(cls)
            binary_true = (Y_true == cls).astype(int)
            try:
                pr_auc_dict[name] = float(average_precision_score(binary_true, Y_proba[:, idx]))
            except ValueError:
                pr_auc_dict[name] = None
        metrics["pr_auc"] = pr_auc_dict

    return metrics
