import data
import cross_validation
import visualization

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FILE_PATH_CSV = ROOT / "data" / "wdbc.csv"
RESULTS_DIR = ROOT / "results"

def main():

    DATAFRAME = data.load_csv(FILE_PATH_CSV)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = data.split_features_and_target(DATAFRAME)

    result = cross_validation.perform_cross_validation(X, y)
    visualization.print_dataset_summary(X, y)
    visualization.print_report(result)
    visualization.save_all(X, y, result, RESULTS_DIR)

if __name__ == "__main__":
    main()
