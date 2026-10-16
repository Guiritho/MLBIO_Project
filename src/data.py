"""Loading and preparation of the WDBC table."""
import pandas as pd

# Constants and metadata for the WDBC dataset.
METADATA_COLUMNS = ["sample_id", "diagnosis"]
# Mapping of diagnosis labels to binary values.
LABELS = {"benign": 0, "malignant": 1}

def load_csv(file_path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)

def split_features_and_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the 30 numerical features and the binary target.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing the WDBC data with metadata and diagnosis columns.

    Returns
    -------
    X, y : tuple[pd.DataFrame, pd.Series]
        A tuple containing the feature matrix (X) and the binary target vector (y).
    """
    unknown = set(dataframe["diagnosis"]) - set(LABELS)
    if unknown:
        raise ValueError(f"Unexpected diagnosis values: {unknown}")

    X = dataframe.drop(columns=METADATA_COLUMNS)
    y = dataframe["diagnosis"].map(LABELS).rename("diagnosis_binary")

    return X, y
