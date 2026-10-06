# File that handles the csv processing to prepare data for the project (like pandas, etc)

import pandas as pd

def load_csv(file_path):
    return pd.read_csv(file_path)
