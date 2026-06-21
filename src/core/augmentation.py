"""Data augmentation module for lithology identification.

Provides oversampling for thin/minority layers, mixup data augmentation,
and weighted sampling utilities for balanced mini-batch training.
"""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def oversample_thin_layers(X, Y, num_classes, min_samples_per_class=500, noise_std=0.02):
    """Oversample minority classes by adding Gaussian noise.

    For classes with fewer than *min_samples_per_class* samples, new
    samples are created by duplicating existing ones and adding
    zero-mean Gaussian noise with standard deviation *noise_std*.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    Y : np.ndarray of shape (n_samples,)
        Integer class labels.
    num_classes : int
        Total number of classes.
    min_samples_per_class : int, optional
        Target minimum number of samples per class.  Default is 500.
    noise_std : float, optional
        Standard deviation of the Gaussian noise added to duplicated
        samples.  Default is 0.02.

    Returns
    -------
    X_aug : np.ndarray
        Augmented feature matrix.
    Y_aug : np.ndarray
        Augmented label vector.
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int64)

    X_list = [X]
    Y_list = [Y]

    for cls in range(num_classes):
        cls_mask = Y == cls
        cls_count = cls_mask.sum()
        if cls_count == 0 or cls_count >= min_samples_per_class:
            continue

        n_needed = min_samples_per_class - cls_count
        cls_indices = np.where(cls_mask)[0]

        # Randomly pick indices to duplicate (with replacement)
        dup_indices = np.random.choice(cls_indices, size=n_needed, replace=True)
        X_dup = X[dup_indices].copy()
        X_dup += np.random.normal(0, noise_std, size=X_dup.shape).astype(np.float32)

        X_list.append(X_dup)
        Y_list.append(np.full(n_needed, cls, dtype=np.int64))

    X_aug = np.concatenate(X_list, axis=0)
    Y_aug = np.concatenate(Y_list, axis=0)

    return X_aug, Y_aug


def mixup_data(x, y, alpha=0.3):
    """Mixup data augmentation.

    Creates convex combinations of randomly paired samples and their
    labels.  See Zhang et al., *mixup: Beyond Empirical Risk
    Minimization* (2018).

    Parameters
    ----------
    x : torch.Tensor of shape (batch, ...)
        Input features.
    y : torch.Tensor
        Labels (integer class indices).
    alpha : float, optional
        Beta distribution parameter.  Default is 0.3.

    Returns
    -------
    mixed_x : torch.Tensor
        Mixed input features.
    y_a : torch.Tensor
        Labels from the first element of the pair.
    y_b : torch.Tensor
        Labels from the second element of the pair.
    lam : float
        Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a = y
    y_b = y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute the mixup loss.

    The loss is a convex combination of the criterion evaluated on both
    label sets: ``lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)``.

    Parameters
    ----------
    criterion : callable
        Loss function (e.g. ``nn.CrossEntropyLoss()``).
    pred : torch.Tensor
        Model predictions.
    y_a : torch.Tensor
        Labels from the first element of the mixup pair.
    y_b : torch.Tensor
        Labels from the second element of the mixup pair.
    lam : float
        Mixing coefficient.

    Returns
    -------
    torch.Tensor
        Scalar mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def create_weighted_sampler(Y, num_classes):
    """Create a WeightedRandomSampler for balanced mini-batch sampling.

    Each sample's weight is inversely proportional to its class
    frequency: ``weight = 1 / (class_count + eps)``.

    Parameters
    ----------
    Y : array-like of int
        Integer class labels for the full dataset.
    num_classes : int
        Total number of classes.

    Returns
    -------
    WeightedRandomSampler
        Sampler that can be passed to a DataLoader.
    """
    Y = np.asarray(Y, dtype=np.int64)
    eps = 1e-8

    # Count samples per class
    class_counts = np.zeros(num_classes, dtype=np.float64)
    for cls in range(num_classes):
        class_counts[cls] = (Y == cls).sum()

    # Per-sample weight = 1 / (class_count + eps)
    sample_weights = np.array(
        [1.0 / (class_counts[label] + eps) for label in Y],
        dtype=np.float64,
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler
