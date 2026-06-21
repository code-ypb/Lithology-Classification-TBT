"""
Data loading and preprocessing module for well-log lithology identification.

Provides utilities for loading well-log CSV files, handling missing values,
creating sliding windows, and preparing datasets for the
TCN-BiLSTM-Transformer lithology classification model.
"""

import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Module-level defaults (can be overridden via function arguments)
# ---------------------------------------------------------------------------
FEATURE_COLS = ["CAL", "DEN", "DT", "GR", "RT", "SP"]
TARGET_COL = "Facies"
DEPTH_COL = "Depth"
WINDOW_SIZE = 41
NUM_CLASSES = 5
MISSING_VALUE_MARKERS = [-999.25, -999, -9999, -999.99]


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all relevant libraries.

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_raw_well(
    file_path: str,
    feature_cols: list | None = None,
    target_col: str = TARGET_COL,
    depth_col: str = DEPTH_COL,
    window_size: int = WINDOW_SIZE,
):
    """Load a single well CSV file and perform basic preprocessing.

    Steps:
        1. Read the CSV file.
        2. Replace known missing-value markers with NaN.
        3. Interpolate linearly and forward/backward-fill remaining NaNs.
        4. Drop rows where the target column is still missing.
        5. Ensure target column is integer type.
        6. Return (DataFrame, depth_array, feature_cols) or
           (None, None, None) if too few valid samples remain.

    Args:
        file_path: Path to the well-log CSV file.
        feature_cols: List of feature column names. Defaults to FEATURE_COLS.
        target_col: Name of the target/label column. Defaults to TARGET_COL.
        depth_col: Name of the depth column. Defaults to DEPTH_COL.
        window_size: Minimum number of samples required (used to reject
            wells that are too short after cleaning). Defaults to WINDOW_SIZE.

    Returns:
        tuple: (df, depth_valid, feature_cols) where *df* is the cleaned
        DataFrame, *depth_valid* is a numpy array of valid depth values, and
        *feature_cols* is the list of feature column names actually used.
        Returns (None, None, None) if the well has too few valid samples.
    """
    if feature_cols is None:
        feature_cols = list(FEATURE_COLS)

    # 1. Read CSV
    try:
        df = pd.read_csv(file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        print(f"[WARNING] Could not read {file_path}: {exc}")
        return None, None, None

    # 2. Replace missing-value markers with NaN
    for marker in MISSING_VALUE_MARKERS:
        df.replace(marker, np.nan, inplace=True)

    # 3. Interpolate and fill remaining missing values
    df = df.infer_objects(copy=False)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    df[numeric_cols] = df[numeric_cols].bfill()
    df[numeric_cols] = df[numeric_cols].ffill()

    # 4. Drop rows with missing target
    df.dropna(subset=[target_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < window_size:
        print(
            f"[WARNING] Well '{file_path}' has only {len(df)} valid rows "
            f"(minimum {window_size} required). Skipping."
        )
        return None, None, None

    # 5. Ensure target column is integer type
    df[target_col] = df[target_col].astype(int)
    df.reset_index(drop=True, inplace=True)

    if len(df) < window_size:
        print(
            f"[WARNING] Well '{file_path}' has only {len(df)} rows after "
            f"cleaning (minimum {window_size} required). Skipping."
        )
        return None, None, None

    # Extract valid depth array
    if depth_col in df.columns:
        depth_valid = df[depth_col].values
    else:
        depth_valid = np.arange(len(df), dtype=float)

    return df, depth_valid, feature_cols


def _engineer_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Apply feature engineering to create additional log-derived features.

    Derived features:
        - GR_norm: Normalised GR (min-max within well)
        - DEN_DT: Density × sonic product (acoustic impedance proxy)
        - GR_RT: GR × log10(RT) (lithology-resistivity cross)
        - CAL_GR: CAL × GR (borehole-lithology cross)

    Args:
        df: Input DataFrame.
        feature_cols: Original feature column names.

    Returns:
        Tuple of (DataFrame with new columns, updated feature column list).
    """
    engineered_cols = list(feature_cols)

    if "GR" in df.columns:
        gr_min, gr_max = df["GR"].min(), df["GR"].max()
        df["GR_norm"] = (df["GR"] - gr_min) / (gr_max - gr_min + 1e-8)
        engineered_cols.append("GR_norm")

    if "DEN" in df.columns and "DT" in df.columns:
        df["DEN_DT"] = df["DEN"] * df["DT"]
        engineered_cols.append("DEN_DT")

    if "GR" in df.columns and "RT" in df.columns:
        df["GR_RT"] = df["GR"] * np.log10(df["RT"].abs() + 1e-8)
        engineered_cols.append("GR_RT")

    if "CAL" in df.columns and "GR" in df.columns:
        df["CAL_GR"] = df["CAL"] * df["GR"]
        engineered_cols.append("CAL_GR")

    return df, engineered_cols


def create_windows(
    df: pd.DataFrame,
    all_features_names: list[str],
    depth_valid: np.ndarray,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
    window_size: int = WINDOW_SIZE,
):
    """Create sliding windows from well data for sequence models.

    Each window is of length *window_size* and the label is the target value
    at the centre position.  Windows are created only where the full window
    fits within the data (i.e. no padding).

    Args:
        df: Cleaned DataFrame containing features and target.
        all_features_names: Ordered list of feature column names to use.
        depth_valid: Array of depth values aligned with df rows.
        scaler: An existing StandardScaler, or None to create a new one.
        fit_scaler: If True, fit the scaler on this data before transforming.
            If False, the scaler must already be fitted.
        window_size: Length of each sliding window. Defaults to WINDOW_SIZE.

    Returns:
        tuple: (X, Y, aligned_depth, all_features_names, scaler)
            - X: np.ndarray of shape (n_windows, window_size, n_features)
            - Y: np.ndarray of shape (n_windows,)
            - aligned_depth: np.ndarray of depth values at window centres
            - all_features_names: list of feature names used
            - scaler: the (possibly newly fitted) StandardScaler
    """
    target_col = TARGET_COL

    # Standardise features
    features = df[all_features_names].values.astype(np.float32)
    if scaler is None:
        scaler = StandardScaler()
        fit_scaler = True

    if fit_scaler:
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)

    targets = df[target_col].values.astype(np.int64)

    # Create sliding windows
    half_w = window_size // 2
    n_samples = len(features)
    n_windows = n_samples - window_size + 1

    if n_windows <= 0:
        print(
            f"[WARNING] Not enough samples ({n_samples}) to create windows "
            f"of size {window_size}."
        )
        return (
            np.array([], dtype=np.float32).reshape(0, window_size, len(all_features_names)),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            all_features_names,
            scaler,
        )

    X = np.zeros((n_windows, window_size, len(all_features_names)), dtype=np.float32)
    Y = np.zeros(n_windows, dtype=np.int64)
    aligned_depth = np.zeros(n_windows, dtype=np.float64)

    for i in range(n_windows):
        X[i] = features[i : i + window_size]
        Y[i] = targets[i + half_w]
        aligned_depth[i] = depth_valid[i + half_w]

    return X, Y, aligned_depth, all_features_names, scaler


def prepare_well_data(
    well_paths: list[str],
    well_names: list[str] | None = None,
    use_feature_engineering: bool = True,
):
    """High-level function to load and prepare all well data.

    Loads each well via :func:`load_raw_well`, optionally applies feature
    engineering, fits a global StandardScaler on the combined training data,
    and creates sliding windows for every well.

    Args:
        well_paths: List of file paths to well-log CSV files.
        well_names: Optional list of human-readable names for each well.
            If None, the stem of each file path is used.
        use_feature_engineering: Whether to apply derived feature columns.
            Defaults to True.

    Returns:
        tuple: (well_data, metadata)
            - well_data: list of (X, Y, depth) tuples, one per well
            - metadata: dict with keys
                * 'well_names': list of well name strings
                * 'feature_names': list of feature column names used
                * 'scaler': the fitted StandardScaler
                * 'n_classes': number of target classes
                * 'window_size': window size used
    """
    if well_names is None:
        from pathlib import Path

        well_names = [Path(p).stem for p in well_paths]

    # ------------------------------------------------------------------
    # Stage 1: Load all wells
    # ------------------------------------------------------------------
    raw_wells: list[tuple] = []  # (df, depth, feature_cols, name)
    for path, name in zip(well_paths, well_names):
        df, depth, feat_cols = load_raw_well(path)
        if df is None:
            continue
        raw_wells.append((df, depth, feat_cols, name))

    if not raw_wells:
        print("[ERROR] No wells could be loaded successfully.")
        return [], {
            "well_names": [],
            "feature_names": list(FEATURE_COLS),
            "scaler": None,
            "n_classes": NUM_CLASSES,
            "window_size": WINDOW_SIZE,
        }

    # ------------------------------------------------------------------
    # Stage 2: Feature engineering (optional)
    # ------------------------------------------------------------------
    all_features_names = list(FEATURE_COLS)
    if use_feature_engineering:
        processed_wells = []
        for df, depth, feat_cols, name in raw_wells:
            df, all_features_names = _engineer_features(df, feat_cols)
            processed_wells.append((df, depth, all_features_names, name))
        raw_wells = processed_wells

    # ------------------------------------------------------------------
    # Stage 3: Fit global scaler on all wells combined
    # ------------------------------------------------------------------
    global_scaler = StandardScaler()
    all_features_combined = np.vstack(
        [df[all_features_names].values.astype(np.float32) for df, *_ in raw_wells]
    )
    global_scaler.fit(all_features_combined)

    # ------------------------------------------------------------------
    # Stage 4: Create windows for each well
    # ------------------------------------------------------------------
    well_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    loaded_names: list[str] = []

    for df, depth, feat_names, name in raw_wells:
        X, Y, aligned_depth, _, _ = create_windows(
            df, feat_names, depth, scaler=global_scaler, fit_scaler=False
        )
        if X.shape[0] > 0:
            well_data.append((X, Y, aligned_depth))
            loaded_names.append(name)

    metadata = {
        "well_names": loaded_names,
        "feature_names": all_features_names,
        "scaler": global_scaler,
        "n_classes": NUM_CLASSES,
        "window_size": WINDOW_SIZE,
    }

    return well_data, metadata
