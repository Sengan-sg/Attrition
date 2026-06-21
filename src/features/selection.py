from __future__ import annotations
from typing import Iterable

import numpy as np
import pandas as pd

def select_features(
        frame: pd.DataFrame,
        target_column: str,
        selected_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """return a subset of columns from *frame*, always including Attrition column."""
    if selected_columns is None:
        return frame
    columns = list(selected_columns)
    if target_column not in columns:
        columns.append(target_column)
    return frame[columns].copy()

class VarianceSelector:
    """Drop numeric colmns whose variance falls below threshold"""

    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = threshold 
        self._keep: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "VarianceSelector":
        numeric = frame.select_dtypes(include=["number"])
        variances = numeric.var()
        low_var_cols = set(variances[variances < self.threshold].index)
        self._keep = [c for c in self._keep if c in frame.columns]
        return self
    
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in self._keep if c in frame.columns]
        return frame[keep].copy()
    
    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)
    

class CorrelationSelector:
    """Drop one column from each highly correlated numeric pair using pearson correlation"""

    def __init__(self, threshold: float = 0.90) -> None:
        self.threshold = threshold
        self._drop: list[str] = []
    
    def fit(self, frame: pd.DataFrame) -> "CorrelationSelector":
        numeric = frame.select_dtypes(include=["number"])
        corr = numeric.corr().abs()
        #Upper triangle only
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        self._drop = [col for col in upper.columns if (upper[col] > self.threshold.any())]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        to_drop = [c for c in self._drop if c in frame.columns]
        return frame.drop(columns=to_drop).copy()

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)