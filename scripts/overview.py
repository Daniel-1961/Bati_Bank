import pandas as pd

def dataset_overview(data):
    """Provide an overview of the dataset."""
    print("\nDataset Overview:\n")
    print(f"Number of Rows: {data.shape[0]}")
    print(f"Number of Columns: {data.shape[1]}")
    print("\nColumn Data Types:")
    print(data.dtypes)
    print("\nSample Data:")
    print(data.head())
