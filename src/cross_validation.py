"""Repeated stratified CV comparing Golub weighted voting and logistic regression."""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import golub

SEED = 34
N_SPLITS = 5
N_REPEATS = 3
N_FEATURES = 10

def _metrics(y_true, y_pred, y_score) -> dict:
    """Compute evaluation metrics for a single fold.

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        True labels for the test set.
    y_pred : pd.Series or np.ndarray
        Predicted labels for the test set.
    y_score : pd.Series or np.ndarray
        Predicted scores (e.g., probabilities) for the test set.

    Returns
    -------
    dict
        Dictionary containing confusion matrix, AUC, and balanced accuracy.
    """

    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "auc": roc_auc_score(y_true, y_score),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }

def perform_cross_validation(X: pd.DataFrame, y: pd.Series, seed: int = SEED) -> dict:
    """Evaluate both models on the same splits; return raw per-fold results only.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix as a pandas DataFrame.
        y : pd.Series
            Target vector as a pandas Series.
        seed : int, optional
            Random seed for reproducibility (default is SEED).

        Returns
        -------
        dict: dict
            Dictionary containing per-fold results for both models.
    """

    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)

    folds = []
    for fold_number, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        golub_model = golub.fit(X_train, y_train, N_FEATURES)
        golub_score = golub.score(golub_model, X_test)
        golub_pred = (golub_score >= 0).astype(int)

        logistic_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=seed),
        )
        logistic_model.fit(X_train, y_train)
        logistic_score = logistic_model.predict_proba(X_test)[:, 1]

        folds.append(
            {
                "fold": fold_number,
                "golub": _metrics(y_test, golub_pred, golub_score),
                "logistic": _metrics(y_test, logistic_model.predict(X_test), logistic_score),
                "selected_features": list(golub_model["weights"].index),
                # Raw material for ROC curves, score histograms and weight tables.
                "y_true": y_test.to_numpy(),
                "scores": {"golub": golub_score.to_numpy(), "logistic": logistic_score},
                "weights": golub_model["weights"],
                "midpoints": golub_model["midpoints"],
            }
        )

    return {"seed": seed, "folds": folds, "columns": list(X.columns)}

def average_metrics(result: dict, model: str) -> dict:
    """Mean of each metric over all folds.

    Parameters
    ----------
    result : dict
        Dictionary containing per-fold results for both models.
    model : str
        Name of the model for which to average metrics.

    Returns
    -------
    dict : dict
        Dictionary containing the mean of each metric over all folds.
    """

    runs = [fold[model] for fold in result["folds"]]
    return {key: np.mean([run[key] for run in runs], axis=0) for key in runs[0]}


def selection_frequency(result: dict) -> pd.Series:
    """Number of folds in which each of the 30 features was selected.

    Parameters
    ----------
    result : dict
        Dictionary containing per-fold results for both models.

    Returns
    -------
    pd.Series
        Series containing the count of folds in which each feature was selected.
    """

    counts = pd.Series(0, index=result["columns"], dtype=int)
    for fold in result["folds"]:
        counts[fold["selected_features"]] += 1
    return counts.sort_values(ascending=False, kind="stable")


def pairwise_jaccard(result: dict) -> list[float]:
    """Jaccard similarity between every pair of selected feature sets.

    Parameters
    ----------
    result : dict
        Dictionary containing per-fold results for both models.

    Returns
    -------
    list[float]
        List containing the Jaccard similarity between every pair of selected feature sets.
    """

    sets = [set(fold["selected_features"]) for fold in result["folds"]]
    return [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
