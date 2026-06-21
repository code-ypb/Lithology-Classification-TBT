import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    optuna = None


def extract_center_features(X_windowed, Y_windowed):
    """Extract center-point features from windowed data for ML baselines.

    Traditional ML models operate on flat feature vectors, not on sequences.
    This helper extracts the feature vector at the center of each window so
    that SVM / XGBoost can be used as single-point classifiers.

    Args:
        X_windowed: (n_samples, window_size, n_features) array.
        Y_windowed: (n_samples,) array.

    Returns:
        X_center: (n_samples, n_features) - features at window center.
        Y: (n_samples,) - labels.
    """
    window_size = X_windowed.shape[1]
    center_idx = window_size // 2
    X_center = X_windowed[:, center_idx, :]
    Y = Y_windowed
    return X_center, Y


def train_svm(X_train, Y_train, X_val, Y_val, n_trials=15, seed=42):
    """Train SVM with Optuna hyperparameter optimization.

    Uses single-point features (window center) rather than sequences.
    Searches over C (log-uniform) and kernel (rbf / linear).

    Args:
        X_train: Training features, shape (n_train, n_features).
        Y_train: Training labels, shape (n_train,).
        X_val: Validation features, shape (n_val, n_features).
        Y_val: Validation labels, shape (n_val,).
        n_trials: Number of Optuna trials.
        seed: Random seed for reproducibility.

    Returns:
        best_model: Trained SVC with the best hyperparameters.
        best_params: Dictionary of the best hyperparameters found.
        val_accuracy: Classification accuracy on the validation set.
    """
    if optuna is None:
        raise ImportError(
            "optuna is required for SVM hyperparameter optimization. "
            "Install it with: pip install optuna"
        )

    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])

        clf = SVC(C=C, kernel=kernel, random_state=seed)
        clf.fit(X_train, Y_train)
        preds = clf.predict(X_val)
        return accuracy_score(Y_val, preds)

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_model = SVC(
        C=best_params["C"],
        kernel=best_params["kernel"],
        random_state=seed,
    )
    best_model.fit(X_train, Y_train)
    val_accuracy = accuracy_score(Y_val, best_model.predict(X_val))

    return best_model, best_params, val_accuracy


def train_xgboost(X_train, Y_train, X_val, Y_val, n_trials=15, seed=42):
    """Train XGBoost with Optuna hyperparameter optimization.

    Uses single-point features (window center).
    Searches over n_estimators, learning_rate, and max_depth.

    Args:
        X_train: Training features, shape (n_train, n_features).
        Y_train: Training labels, shape (n_train,).
        X_val: Validation features, shape (n_val, n_features).
        Y_val: Validation labels, shape (n_val,).
        n_trials: Number of Optuna trials.
        seed: Random seed for reproducibility.

    Returns:
        best_model: Trained XGBClassifier with the best hyperparameters.
        best_params: Dictionary of the best hyperparameters found.
        val_accuracy: Classification accuracy on the validation set.
    """
    if optuna is None:
        raise ImportError(
            "optuna is required for XGBoost hyperparameter optimization. "
            "Install it with: pip install optuna"
        )

    num_classes = len(np.unique(np.concatenate([Y_train, Y_val])))

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 10)

        clf = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_class=num_classes,
            objective="multi:softmax" if num_classes > 2 else "binary:logistic",
            use_label_encoder=False,
            eval_metric="mlogloss" if num_classes > 2 else "logloss",
            random_state=seed,
            verbosity=0,
        )
        clf.fit(X_train, Y_train)
        preds = clf.predict(X_val)
        return accuracy_score(Y_val, preds)

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_model = XGBClassifier(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        num_class=num_classes,
        objective="multi:softmax" if num_classes > 2 else "binary:logistic",
        use_label_encoder=False,
        eval_metric="mlogloss" if num_classes > 2 else "logloss",
        random_state=seed,
        verbosity=0,
    )
    best_model.fit(X_train, Y_train)
    val_accuracy = accuracy_score(Y_val, best_model.predict(X_val))

    return best_model, best_params, val_accuracy
