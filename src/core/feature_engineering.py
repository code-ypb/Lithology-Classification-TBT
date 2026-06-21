"""Feature engineering module for well-log lithology identification.

Provides functions to derive physically meaningful and statistical features
from raw well-log curves, as well as utilities to map derived features back
to their parent raw logging curves and aggregate importance scores.
"""

import numpy as np
import pandas as pd


def engineer_features(
    df,
    depth_valid,
    feature_cols=None,
):
    """Engineer features from raw well-log curves for lithology identification.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw well-log curves.
    depth_valid : array-like
        1-D array of depth values corresponding to each row in *df*.
    feature_cols : list of str, optional
        Column names of the raw logging curves to use.  Defaults to
        ``['CAL', 'DEN', 'DT', 'GR', 'RT', 'SP']``.

    Returns
    -------
    df : pd.DataFrame
        The input DataFrame augmented with all engineered features.
    all_feature_names : list of str
        Names of all feature columns (excluding the target column).
    """
    if feature_cols is None:
        feature_cols = ["CAL", "DEN", "DT", "GR", "RT", "SP"]

    df = df.copy()
    depth_valid = np.asarray(depth_valid, dtype=np.float64)

    # ------------------------------------------------------------------
    # 1. Relative depth (normalised to [0, 1])
    # ------------------------------------------------------------------
    depth_min, depth_max = depth_valid.min(), depth_valid.max()
    if depth_max - depth_min > 0:
        df["Relative_Depth"] = (depth_valid - depth_min) / (depth_max - depth_min)
    else:
        df["Relative_Depth"] = 0.0

    # ------------------------------------------------------------------
    # 2. Vshale from GR — linear interpolation between P5 and P95
    # ------------------------------------------------------------------
    if "GR" in df.columns:
        gr_p5 = df["GR"].quantile(0.05)
        gr_p95 = df["GR"].quantile(0.95)
        denom = gr_p95 - gr_p5
        if denom > 0:
            df["Vshale"] = ((df["GR"] - gr_p5) / denom).clip(0, 1)
        else:
            df["Vshale"] = 0.0
    else:
        df["Vshale"] = 0.0

    # ------------------------------------------------------------------
    # 3. Uranium anomaly flag — high GR + low DEN
    # ------------------------------------------------------------------
    if "GR" in df.columns and "DEN" in df.columns:
        gr_threshold = df["GR"].quantile(0.75)
        den_threshold = df["DEN"].quantile(0.25)
        df["Uranium_Anomaly_Flag"] = (
            (df["GR"] > gr_threshold) & (df["DEN"] < den_threshold)
        ).astype(int)
    else:
        df["Uranium_Anomaly_Flag"] = 0

    # ------------------------------------------------------------------
    # 4. Slopes — differenced and smoothed
    # ------------------------------------------------------------------
    for col in ["GR", "SP"]:
        if col in df.columns:
            diff = df[col].diff().fillna(0)
            df[f"{col}_slope"] = diff.rolling(window=3, center=True, min_periods=1).mean()

    # ------------------------------------------------------------------
    # 5. First-order differences for each raw feature
    # ------------------------------------------------------------------
    for col in feature_cols:
        if col in df.columns:
            df[f"{col}_diff"] = df[col].diff().fillna(0)

    # ------------------------------------------------------------------
    # 6. Rolling standard deviations (windows 5 and 20)
    # ------------------------------------------------------------------
    roll_std_cols = ["GR", "RT", "DEN"]
    for col in roll_std_cols:
        if col in df.columns:
            df[f"{col}_roll_std_5"] = (
                df[col].rolling(window=5, center=True, min_periods=1).std().fillna(0)
            )
            df[f"{col}_roll_std_20"] = (
                df[col].rolling(window=20, center=True, min_periods=1).std().fillna(0)
            )

    # ------------------------------------------------------------------
    # 7. Cross-curve features
    # ------------------------------------------------------------------
    eps = 1e-8
    if "GR" in df.columns and "RT" in df.columns:
        df["GR_RT_prod"] = df["GR"] * np.log1p(df["RT"])
        df["GR_RT_ratio"] = df["GR"] / (np.log1p(df["RT"]) + eps)

    if "DEN" in df.columns and "DT" in df.columns:
        df["DEN_DT_ratio"] = df["DEN"] / (df["DT"] + eps)

    if "SP" in df.columns and "GR" in df.columns:
        df["SP_GR_diff"] = df["SP"] - df["GR"]

    # ------------------------------------------------------------------
    # 8. Rolling means (windows 5 and 20)
    # ------------------------------------------------------------------
    roll_mean_cols = ["GR", "RT", "DEN", "DT"]
    for col in roll_mean_cols:
        if col in df.columns:
            df[f"{col}_roll_mean_5"] = (
                df[col].rolling(window=5, center=True, min_periods=1).mean().bfill().ffill()
            )
            df[f"{col}_roll_mean_20"] = (
                df[col].rolling(window=20, center=True, min_periods=1).mean().bfill().ffill()
            )

    # ------------------------------------------------------------------
    # 9. Absolute differences (lags 3 and 7)
    # ------------------------------------------------------------------
    abs_diff_cols = ["GR", "DEN"]
    for col in abs_diff_cols:
        if col in df.columns:
            df[f"{col}_abs_diff_3"] = (df[col] - df[col].shift(3)).abs().fillna(0)
            df[f"{col}_abs_diff_7"] = (df[col] - df[col].shift(7)).abs().fillna(0)

    # ------------------------------------------------------------------
    # Collect all feature column names (exclude target if present)
    # ------------------------------------------------------------------
    exclude = {"label", "LITHO", "LITHOLOGY", "DEPTH", "Depth", "Well", "well_name", "Facies"}
    all_feature_names = [
        c for c in df.columns
        if c not in exclude and c not in feature_cols and df[c].dtype in ('float64', 'float32', 'int64', 'int32')
    ] + list(feature_cols)

    # De-duplicate while preserving order
    seen = set()
    unique_features = []
    for name in all_feature_names:
        if name not in seen:
            seen.add(name)
            unique_features.append(name)

    return df, unique_features


def map_feature_to_raw(feature_name, feature_cols):
    """Map a derived feature name back to its parent raw logging curve(s).

    Parameters
    ----------
    feature_name : str
        Name of the derived feature.
    feature_cols : list of str
        List of raw logging-curve column names.

    Returns
    -------
    list of str
        Parent raw feature names that contribute to *feature_name*.
    """
    parents = []

    # Direct raw feature
    if feature_name in feature_cols:
        parents.append(feature_name)
        return parents

    # Relative_Depth — depends on depth, no single raw curve
    if feature_name == "Relative_Depth":
        return []

    # Vshale — derived from GR
    if feature_name == "Vshale":
        if "GR" in feature_cols:
            parents.append("GR")
        return parents

    # Uranium_Anomaly_Flag — from GR and DEN
    if feature_name == "Uranium_Anomaly_Flag":
        for col in ["GR", "DEN"]:
            if col in feature_cols:
                parents.append(col)
        return parents

    # Slopes — single parent
    if feature_name.endswith("_slope"):
        raw = feature_name.replace("_slope", "")
        if raw in feature_cols:
            parents.append(raw)
        return parents

    # First-order differences
    if feature_name.endswith("_diff") and not feature_name.startswith("SP_GR"):
        raw = feature_name.replace("_diff", "")
        if raw in feature_cols:
            parents.append(raw)
        return parents

    # Rolling standard deviations
    for suffix in ("_roll_std_5", "_roll_std_20"):
        if feature_name.endswith(suffix):
            raw = feature_name[: -len(suffix)]
            if raw in feature_cols:
                parents.append(raw)
            return parents

    # Rolling means
    for suffix in ("_roll_mean_5", "_roll_mean_20"):
        if feature_name.endswith(suffix):
            raw = feature_name[: -len(suffix)]
            if raw in feature_cols:
                parents.append(raw)
            return parents

    # Absolute differences
    for suffix in ("_abs_diff_3", "_abs_diff_7"):
        if feature_name.endswith(suffix):
            raw = feature_name[: -len(suffix)]
            if raw in feature_cols:
                parents.append(raw)
            return parents

    # Cross-curve features
    cross_map = {
        "GR_RT_prod": ["GR", "RT"],
        "GR_RT_ratio": ["GR", "RT"],
        "DEN_DT_ratio": ["DEN", "DT"],
        "SP_GR_diff": ["SP", "GR"],
    }
    if feature_name in cross_map:
        for col in cross_map[feature_name]:
            if col in feature_cols:
                parents.append(col)
        return parents

    # Fallback: try to match a raw prefix
    for raw in feature_cols:
        if feature_name.startswith(raw):
            parents.append(raw)
            break

    return parents


def aggregate_to_raw_features(importance_scores, all_features_names, feature_cols):
    """Aggregate per-feature importance scores to raw logging-curve level.

    Parameters
    ----------
    importance_scores : array-like
        Importance score for each feature in *all_features_names*.
    all_features_names : list of str
        Names of all engineered features (same length as *importance_scores*).
    feature_cols : list of str
        List of raw logging-curve column names.

    Returns
    -------
    raw_importance : dict
        Mapping from raw feature name to aggregated importance.
    raw_detail : dict
        Nested mapping: ``{raw_feature: {derived_feature: score, ...}, ...}``.
    """
    importance_scores = np.asarray(importance_scores, dtype=np.float64)

    raw_importance = {col: 0.0 for col in feature_cols}
    raw_detail = {col: {} for col in feature_cols}

    for feat_name, score in zip(all_features_names, importance_scores):
        parents = map_feature_to_raw(feat_name, feature_cols)
        if not parents:
            continue
        share = score / len(parents)
        for parent in parents:
            raw_importance[parent] += share
            raw_detail[parent][feat_name] = raw_detail[parent].get(feat_name, 0.0) + share

    return raw_importance, raw_detail
