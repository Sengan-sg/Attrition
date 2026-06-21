from __future__ import annotations

import pandas as pd

def summarize_dataset(frame: pd.DataFrame) -> dict:
    """Summarize the dataset by providing basic statistics and information.

    Args:
        frame: The input DataFrame.
    Returns:
        A dictionary containing summary statistics and information about the dataset.
        """
    return {
        "shape": frame.shape,
        "columns": frame.columns.tolist(),
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "missing_values": frame.isnull().sum().to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),

    }