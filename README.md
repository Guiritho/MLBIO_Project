# MLBIO Project: Golub weighted voting on WDBC

Applicability study of the weighted-voting classifier of Golub et al. (1999) on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset (569 patients, 30 features extracted from fine-needle-aspirate images), compared with logistic regression.

## Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)

## Run the whole pipeline

From the repository root, one command installs the locked environment (from `pyproject.toml` and `uv.lock`) and produces every table and figure:

```bash
uv run python src/main.py
```

The console report is printed and all outputs are written to `results/`.

## What the pipeline does

- Loads `data/wdbc.csv` (malignant = 1, benign = 0) and checks the diagnosis labels.
- Evaluates two models on the **same 15 splits**: `RepeatedStratifiedKFold` with 5 folds, 3 repeats and personal seed `34`.
  - **Golub weighted voting**: weights, the ten selected features and the midpoints are estimated on each training fold only; malignant if `V(x) >= 0`.
  - **Logistic regression** on all 30 features: `StandardScaler` fitted on each training fold only, `LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)`, threshold 0.5.
- Reports the confusion matrix, AUC and balanced accuracy for an example model (fold 1) and their averages over the 15 models.
- Records the features selected in every fold: selection frequency of all 30 features and mean pairwise Jaccard similarity of the 15 selected sets.

## Outputs

| Path | Content |
|---|---|
| `results/tables/` | Descriptive statistics, per-fold metrics, performance summary, selection frequency and matrix, example-fold weights (CSV) |
| `results/figures/` | Class balance, feature correlation, confusion matrices, metric comparison, paired differences, ROC curves, score distribution, selection frequency and heatmap, Jaccard, stability vs performance (PNG) |

## Code layout

| File | Role |
|---|---|
| `src/main.py` | Entry point: loads data, runs the evaluation, writes outputs |
| `src/data.py` | Loading and feature/target preparation |
| `src/golub.py` | Weighted-voting fit, score and prediction |
| `src/cross_validation.py` | Splits, both models, per-fold metrics, stability measures |
| `src/visualization.py` | Console report, tables and figures |

## Documentation

- [Data dictionary](data/DATA_DICTIONARY.md) and [manifest](data/MANIFEST.json)

## References

- Golub, T. R. et al. (1999). Molecular classification of cancer: class discovery and class prediction by gene expression monitoring. *Science*, 286(5439), 531-537.
- Wolberg, W., Mangasarian, O., Street, N., and Street, W. Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. [DOI: 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B).
- Wolberg, W. H., Street, W. N., and Mangasarian, O. L. (1994). *Cancer Letters*, 77(2-3), 163-171. [DOI: 10.1016/0304-3835(94)90099-X](https://doi.org/10.1016/0304-3835(94)90099-X).
- Machine learning tooling: scikit-learn, pandas, NumPy, matplotlib. No external code was reused.
