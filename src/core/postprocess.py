"""Post-processing module for lithology prediction smoothing.

Provides CRF-like, adaptive-median, and ensemble post-processing methods
to refine raw lithology predictions by leveraging local context and
prediction confidence.
"""

import numpy as np
from scipy.ndimage import median_filter


def crf_postprocess(preds, Y_proba, num_classes, window=5, confidence_weight=0.7):
    """CRF-like post-processing combining local voting with probability confidence.

    For each position, a weighted combination of the average class confidence
    and the vote counts in a local window is computed.  The class with the
    highest combined score is selected.

    Parameters
    ----------
    preds : array-like of int
        Raw predicted class labels (1-D).
    Y_proba : array-like of shape (n_samples, num_classes)
        Predicted class probabilities.
    num_classes : int
        Number of lithology classes.
    window : int, optional
        Half-width of the local voting window.  Default is 5.
    confidence_weight : float, optional
        Weight for the confidence term (0–1).  The voting term receives
        ``1 - confidence_weight``.  Default is 0.7.

    Returns
    -------
    np.ndarray of int
        Smoothed class predictions.
    """
    preds = np.asarray(preds, dtype=np.int64)
    Y_proba = np.asarray(Y_proba, dtype=np.float64)
    n = len(preds)
    result = preds.copy()

    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        local_preds = preds[lo:hi]
        local_proba = Y_proba[lo:hi]

        # Average confidence in the window
        avg_conf = local_proba.mean(axis=0)

        # Vote counts in the window
        vote_counts = np.zeros(num_classes, dtype=np.float64)
        for cls in local_preds:
            vote_counts[cls] += 1
        # Normalise votes
        vote_sum = vote_counts.sum()
        if vote_sum > 0:
            vote_counts /= vote_sum

        # Weighted combination
        combined = confidence_weight * avg_conf + (1 - confidence_weight) * vote_counts
        result[i] = np.argmax(combined)

    return result


def adaptive_median_filter(preds, Y_proba, num_classes, min_confidence=0.5, kernel_size=5):
    """Apply median filter only to low-confidence predictions.

    Positions where the maximum predicted probability is at least
    *min_confidence* are left unchanged; all others are replaced by the
    local median.

    Parameters
    ----------
    preds : array-like of int
        Raw predicted class labels (1-D).
    Y_proba : array-like of shape (n_samples, num_classes)
        Predicted class probabilities.
    num_classes : int
        Number of lithology classes.
    min_confidence : float, optional
        Confidence threshold below which a prediction is smoothed.
        Default is 0.5.
    kernel_size : int, optional
        Kernel size for the median filter.  Default is 5.

    Returns
    -------
    np.ndarray of int
        Adaptively smoothed class predictions.
    """
    preds = np.asarray(preds, dtype=np.int64)
    Y_proba = np.asarray(Y_proba, dtype=np.float64)

    max_conf = Y_proba.max(axis=1)
    smoothed = median_filter(preds, size=kernel_size, mode="nearest")
    result = np.where(max_conf >= min_confidence, preds, smoothed)

    return result.astype(np.int64)


def ensemble_postprocess(preds_raw, Y_proba, num_classes, kernel_size=3):
    """Ensemble of median filter + CRF + adaptive median filter.

    For each position, the majority vote among the three methods is taken.
    If no majority exists, the class with the highest probability is used.

    Parameters
    ----------
    preds_raw : array-like of int
        Raw predicted class labels (1-D).
    Y_proba : array-like of shape (n_samples, num_classes)
        Predicted class probabilities.
    num_classes : int
        Number of lithology classes.
    kernel_size : int, optional
        Kernel size for the base median filter.  Default is 3.

    Returns
    -------
    np.ndarray of int
        Ensemble-smoothed class predictions.
    """
    preds_raw = np.asarray(preds_raw, dtype=np.int64)
    Y_proba = np.asarray(Y_proba, dtype=np.float64)

    # Method 1: simple median filter
    med_preds = median_filter(preds_raw, size=kernel_size, mode="nearest").astype(np.int64)

    # Method 2: CRF-like post-processing
    crf_preds = crf_postprocess(preds_raw, Y_proba, num_classes, window=5, confidence_weight=0.7)

    # Method 3: adaptive median filter
    ada_preds = adaptive_median_filter(
        preds_raw, Y_proba, num_classes, min_confidence=0.5, kernel_size=5
    )

    # Majority vote among the three methods
    n = len(preds_raw)
    result = np.empty(n, dtype=np.int64)

    for i in range(n):
        votes = np.array([med_preds[i], crf_preds[i], ada_preds[i]])
        counts = np.bincount(votes, minlength=num_classes)
        max_count = counts.max()
        # Majority requires at least 2 out of 3
        if max_count >= 2:
            result[i] = counts.argmax()
        else:
            # No majority — fall back to probability argmax
            result[i] = Y_proba[i].argmax()

    return result
