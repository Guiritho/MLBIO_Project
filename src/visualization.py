"""Console report, tables and figures for the WDBC weighted-voting study."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import cross_validation as cv

# Visualization settings and constants
MODELS = {"golub": "Golub weighted voting", "logistic": "Logistic regression"}
COLORS = {"golub": "tab:blue", "logistic": "tab:orange"}
CLASS_NAMES = {0: "benign", 1: "malignant"}
EXAMPLE_FOLD = 0  # the arbitrary "example model": first of the 15


# --------------------------------------------------------------------------- #
# Tables and matrices for visualization
# --------------------------------------------------------------------------- #
def per_fold_table(result: dict) -> pd.DataFrame:
    """Create a table with one row per fold and model, containing AUC and balanced accuracy."""
    rows = []
    for fold in result["folds"]:
        for model, label in MODELS.items():
            m = fold[model]
            rows.append(
                {
                    "fold": fold["fold"],
                    "repeat": (fold["fold"] - 1) // cv.N_SPLITS + 1,
                    "model": label,
                    "auc": m["auc"],
                    "balanced_accuracy": m["balanced_accuracy"],
                }
            )
    return pd.DataFrame(rows)

def summary_table(result: dict) -> pd.DataFrame:
    """Mean and standard deviation over the 15 folds for each model."""
    metrics = ["auc", "balanced_accuracy"]
    grouped = per_fold_table(result).groupby("model", sort=False)[metrics]
    return grouped.agg(["mean", "std"])

def descriptive_table(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Per-class mean/std | Descriptive only: never used for selection."""
    benign, malignant = X[y == 0], X[y == 1]

    return pd.DataFrame(
        {
            "mean_benign": benign.mean(),
            "std_benign": benign.std(),
            "mean_malignant": malignant.mean(),
            "std_malignant": malignant.std(),
        })

def selection_matrix(result: dict) -> pd.DataFrame:
    """Boolean matrix (folds x features) | features ordered by selection frequency."""
    order = cv.selection_frequency(result).index
    matrix = pd.DataFrame(False, index=[f["fold"] for f in result["folds"]], columns=order)
    for fold in result["folds"]:
        matrix.loc[fold["fold"], fold["selected_features"]] = True
    return matrix

def jaccard_matrix(result: dict) -> np.ndarray:
    """Symmetric fold x fold Jaccard similarity of the selected sets."""
    sets = [set(f["selected_features"]) for f in result["folds"]]
    return np.array([[len(a & b) / len(a | b) for b in sets] for a in sets])


# --------------------------------------------------------------------------- #
# Console report
# --------------------------------------------------------------------------- #
def print_dataset_summary(X: pd.DataFrame, y: pd.Series) -> None:
    counts = y.value_counts().sort_index()
    print(f"Dataset: {X.shape[0]} patients, {X.shape[1]} features")
    for label, n in counts.items():
        print(f"  {CLASS_NAMES[label]} ({label}): {n} ({n / len(y):.1%})")
    print(f"  missing values: {int(X.isna().sum().sum())}")

def print_report(result: dict) -> None:
    print(f"Personal seed: {result['seed']}")

    example = result["folds"][EXAMPLE_FOLD]
    print(f"\n--- Example model (fold {example['fold']}) ---")
    for model, label in MODELS.items():
        m = example[model]
        print(f"{label}: AUC={m['auc']:.4f}  balanced accuracy={m['balanced_accuracy']:.4f}")
        print(f"Confusion matrix (rows = true, columns = predicted):\n{m['confusion_matrix']}")
    print("\nGolub weights of the example model (selected features):")
    print(example["weights"].round(4).to_string())

    print(f"\n--- Average over {len(result['folds'])} models ---")
    for model, label in MODELS.items():
        m = cv.average_metrics(result, model)
        print(f"{label}: AUC={m['auc']:.4f}  balanced accuracy={m['balanced_accuracy']:.4f}")
        print(f"Mean confusion matrix (average over folds, not patient counts):\n{m['confusion_matrix'].round(2)}")
    print("\nMean and standard deviation over folds:")
    print(summary_table(result).round(4).to_string())

    print("\n--- Per-fold results ---")
    print(per_fold_table(result).round(4).to_string(index=False))

    print("\n--- Feature selection stability ---")
    print(f"Mean pairwise Jaccard: {np.mean(cv.pairwise_jaccard(result)):.4f}")
    print("Selection frequency (folds out of 15, zeros included):")
    print(cv.selection_frequency(result).to_string())
    print("\nSelected features per fold:")
    for fold in result["folds"]:
        print(f"  Fold {fold['fold']:>2}: {', '.join(fold['selected_features'])}")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def plot_class_balance(y: pd.Series, path: Path) -> None:
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4, 3.5))
    bars = ax.bar([CLASS_NAMES[i] for i in counts.index], counts.values, color=["tab:green", "tab:red"])
    ax.bar_label(bars)
    ax.set_ylabel("Patients")
    ax.set_title("Diagnosis distribution")
    _save(fig, path)

def plot_correlation(X: pd.DataFrame, path: Path) -> None:
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Feature correlation")
    _save(fig, path)

def plot_example_confusion(result: dict, path: Path) -> None:
    example = result["folds"][EXAMPLE_FOLD]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, (model, label) in zip(axes, MODELS.items()):
        cm = example[model]["confusion_matrix"]
        ax.imshow(cm, cmap="Blues")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, int(value), ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1], ["benign", "malignant"])
        ax.set_yticks([0, 1], ["benign", "malignant"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{label}\nAUC={example[model]['auc']:.3f}, "
                     f"BA={example[model]['balanced_accuracy']:.3f}")
    _save(fig, path)

def plot_score_distribution(result: dict, path: Path) -> None:
    """Golub vote V(x) per true class on the example fold; the decision threshold is 0."""
    example = result["folds"][EXAMPLE_FOLD]
    scores, y_true = example["scores"]["golub"], example["y_true"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color in zip((0, 1), ("tab:green", "tab:red")):
        ax.hist(scores[y_true == label], bins=25, alpha=0.6, color=color, label=CLASS_NAMES[label])
    ax.axvline(0, color="black", linestyle="--", label="threshold V(x) = 0")
    ax.set_xlabel("Weighted vote V(x)")
    ax.set_ylabel("Patients")
    ax.legend()
    ax.set_title(f"Golub scores, fold {example['fold']}")
    _save(fig, path)

def plot_metric_comparison(result: dict, path: Path) -> None:
    table = per_fold_table(result)
    metrics = ["auc", "balanced_accuracy",]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4))
    for ax, metric in zip(axes, metrics):
        data = [table.loc[table["model"] == label, metric] for label in MODELS.values()]
        ax.boxplot(data, tick_labels=["Golub", "Logistic"])
        ax.set_title(metric.replace("_", " "))
    fig.suptitle("Performance over the 15 validation folds")
    _save(fig, path)

def plot_paired_differences(result: dict, path: Path) -> None:
    """Logistic minus Golub on identical splits; points are individual folds."""
    table = per_fold_table(result)
    golub_rows = table[table["model"] == MODELS["golub"]].set_index("fold")
    logistic_rows = table[table["model"] == MODELS["logistic"]].set_index("fold")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, metric in zip(axes, ("auc", "balanced_accuracy")):
        diff = logistic_rows[metric] - golub_rows[metric]
        ax.axhline(0, color="black", linestyle="--")
        ax.scatter(np.ones(len(diff)) + np.linspace(-0.15, 0.15, len(diff)), diff, color="tab:gray")
        ax.hlines(diff.mean(), 0.8, 1.2, color="tab:red", label="mean")
        ax.set_xticks([])
        ax.set_title(f"{metric.replace('_', ' ')}: logistic - Golub")
        ax.legend()
    _save(fig, path)

def plot_roc_curves(result: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, (model, label) in zip(axes, MODELS.items()):
        for index, fold in enumerate(result["folds"]):
            fpr, tpr, _ = roc_curve(fold["y_true"], fold["scores"][model])
            is_example = index == EXAMPLE_FOLD
            ax.plot(fpr, tpr, color=COLORS[model], alpha=1.0 if is_example else 0.25,
                    linewidth=2.5 if is_example else 1,
                    label=f"example fold ({fold['fold']})" if is_example else None)
        ax.plot([0, 1], [0, 1], color="gray", linestyle=":")
        ax.set_xlabel("False positive rate")
        ax.set_title(label)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("True positive rate")
    _save(fig, path)

def plot_selection_frequency(result: dict, path: Path) -> None:
    freq = cv.selection_frequency(result)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(freq.index[::-1], freq.values[::-1], color="tab:blue")
    ax.set_xlim(0, len(result["folds"]))
    ax.set_xlabel("Folds in which the feature was selected (max 15)")
    ax.set_title("Golub feature selection frequency")
    _save(fig, path)

def plot_selection_heatmap(result: dict, path: Path) -> None:
    matrix = selection_matrix(result)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.set_ylabel("Fold")
    ax.set_title("Selected features per fold (dark = selected)")
    _save(fig, path)

def plot_jaccard(result: dict, path: Path) -> None:
    matrix = jaccard_matrix(result)
    pairs = cv.pairwise_jaccard(result)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    im = axes[0].imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=axes[0], label="Jaccard")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Fold")
    axes[0].set_title("Pairwise Jaccard of selected sets")
    axes[1].hist(pairs, bins=15, color="tab:blue")
    axes[1].axvline(np.mean(pairs), color="red", linestyle="--", label=f"mean = {np.mean(pairs):.3f}")
    axes[1].set_xlabel("Jaccard similarity")
    axes[1].set_ylabel("Pairs of folds")
    axes[1].legend()
    _save(fig, path)

def plot_stability_vs_performance(result: dict, path: Path) -> None:
    """Does a fold whose selected set resembles the others also perform better?"""
    matrix = jaccard_matrix(result)
    typicality = (matrix.sum(axis=1) - 1) / (len(matrix) - 1)  # mean Jaccard to the other folds
    balanced = [f["golub"]["balanced_accuracy"] for f in result["folds"]]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(typicality, balanced, color="tab:blue")
    ax.set_xlabel("Mean Jaccard with the other folds")
    ax.set_ylabel("Golub balanced accuracy")
    ax.set_title("Feature-set typicality vs performance")
    _save(fig, path)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def save_all(X: pd.DataFrame, y: pd.Series, result: dict, out_dir: Path) -> None:
    """Write every table (CSV) and figure (PNG) under out_dir/tables and out_dir/figures."""
    tables, figures = out_dir / "tables", out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    descriptive_table(X, y).to_csv(tables / "descriptive_statistics.csv")
    per_fold_table(result).to_csv(tables / "per_fold_metrics.csv", index=False)
    summary_table(result).to_csv(tables / "performance_summary.csv")
    cv.selection_frequency(result).rename("folds_selected").to_csv(tables / "selection_frequency.csv")
    selection_matrix(result).to_csv(tables / "selection_matrix.csv")
    result["folds"][EXAMPLE_FOLD]["weights"].rename("weight").to_csv(tables / "example_fold_weights.csv")

    plot_class_balance(y, figures / "class_balance.png")
    plot_correlation(X, figures / "feature_correlation.png")
    plot_example_confusion(result, figures / "example_confusion_matrices.png")
    plot_score_distribution(result, figures / "golub_score_distribution.png")
    plot_metric_comparison(result, figures / "metric_comparison.png")
    plot_paired_differences(result, figures / "paired_differences.png")
    plot_roc_curves(result, figures / "roc_curves.png")
    plot_selection_frequency(result, figures / "selection_frequency.png")
    plot_selection_heatmap(result, figures / "selection_heatmap.png")
    plot_jaccard(result, figures / "jaccard.png")
    plot_stability_vs_performance(result, figures / "stability_vs_performance.png")
