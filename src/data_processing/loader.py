from __future__ import annotations
import pandas as pd
from pathlib import Path

def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load dataset from the given path.

    Args:
        path: The path to the dataset file.

    Returns:
        A pandas DataFrame containing the loaded dataset.
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")   
    if dataset_path.suffix.lower() == ".csv":
        return pd.read_csv(dataset_path)
    if dataset_path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(dataset_path)
    raise ValueError(f"Unsupported file format: {dataset_path.suffix}. Supported formats are .csv, .xlsx, .xls.")