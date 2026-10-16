"""Weighted-voting classifier of Golub et al. (1999), adapted to tabular features."""
import pandas as pd


def fit(X_train: pd.DataFrame, y_train: pd.Series, n_features: int = 10) -> dict:
    """Estimate weights, selected features and midpoints from the training data only.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    n_features : int, optional
        Number of top features to select based on their weights, by default 10.

    Returns
    -------
    dict : dict
        Dictionary containing the selected feature weights and midpoints.
    """
    grouped = X_train.groupby(y_train)
    mean = grouped.mean()
    std = grouped.std()  # pandas default (ddof=1): sample standard deviation

    weights = (mean.loc[1] - mean.loc[0]) / (std.loc[1] + std.loc[0])

    # Stable sort so that ties are broken by column order.
    selected = weights.abs().sort_values(ascending=False, kind="stable").index[:n_features]
    midpoints = (mean.loc[1] + mean.loc[0]) / 2

    return {"weights": weights[selected], "midpoints": midpoints[selected]}


def score(model: dict, X: pd.DataFrame) -> pd.Series:
    """Vote V(x) = sum_j w_j * (x_j - midpoint_j) over the selected features.

    Parameters
    ----------
    model : dict
        Dictionary containing the selected feature weights and midpoints.
    X : pd.DataFrame
        Feature matrix for which to compute the vote.

    Returns
    -------
    pd.Series : pd.Series
        Series containing the vote for each sample.
    """
    selected = model["weights"].index
    return ((X[selected] - model["midpoints"]) * model["weights"]).sum(axis=1)


def predict(model: dict, X: pd.DataFrame) -> pd.Series:
    """Predict malignant (1) when V(x) >= 0, benign (0) otherwise.

    Parameters
    ----------
    model : dict
        Dictionary containing the selected feature weights and midpoints.
    X : pd.DataFrame
        Feature matrix for which to make predictions.

    Returns
    -------
    pd.Series : pd.Series
        Series containing the predicted class for each sample.
    """
    return (score(model, X) >= 0).astype(int)
