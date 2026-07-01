

import pandas as pd


REQUIRED_COLUMNS = {"review", "sentiment"}


def load_data(file_path):
   
    dataframe = pd.read_csv(file_path)

    if dataframe.empty:
        raise ValueError("The dataset is empty.")

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    print(f"Number of rows: {dataframe.shape[0]}")
    print(f"Number of columns: {dataframe.shape[1]}")
    print("Sentiment class distribution:")
    print(dataframe["sentiment"].value_counts())

    return dataframe
