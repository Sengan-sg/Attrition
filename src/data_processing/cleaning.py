from __future__ import annotations

import pandas as pd

# Columns that are constant in dataset - carry zero information and can be dropped
_CONSTANT_COLUMNS: frozenset[str] = frozenset({"StandardHours", "Over18", "EmployeeCount", "EmployeeNumber"})

#Pure identifier columns - carry no information for modeling and can be dropped
_ID_COLUMNS: frozenset[str] = frozenset({"EmployeeNumber"})

# Numeric columns Known to produce outliers that should be capped
_OUTLIER_COLUMNS: list[str] = [
    "MonthlyIncome", 
    "TotalWorkingYears"
]


def clean_raw_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Pre-split structural cleanup.

    Safe to call on the full dataset before the train-test split because no 
    statistics are computed from the data - operations are purely structural.

    Steps
    _____
    1. Normalise column names (strip whitespace, space -> underscore)
    2. Drop exact duplicate rows.
    3. Remove constant columns (EmployeeCount, Over18, StandardHours) and the employee ID column.
    4. Encode the target column ''Attrition'' (Yes -> 1, No -> 0)
    5. Encode ''OverTime'' (Yes -> 1, No -> 0) - binary, no statistical choice.

    Args:
        frame: The input DataFrame to be cleaned.
        """
    
    cleaned = frame.copy()

    # Normalise column names
    cleaned_columns = cleaned.columns.str.strip().str.replace(" ", "_", regex=False)

    # Drop duplicate rows
    cleaned = cleaned.drop_duplicates()

    # Remove constant /identifier columns that are present
    cols_to_drop = (_CONSTANT_COLUMNS | _ID_COLUMNS) & set(cleaned_columns)
    cleaned = cleaned.drop(columns=list(cols_to_drop))

    # Encode target: Yes -> 1, No -> 0
    if "Attrition" in cleaned_columns:
        cleaned["Attrition"] = ( cleaned["Attrition"].map({"Yes": 1, "No": 0}).astype("Int64") )

    # Encode Overtime: Yes -> 1, No -> 0
    if "OverTime" in cleaned_columns:
        cleaned["OverTime"] = ( cleaned["OverTime"].map({"Yes": 1, "No": 0}).astype("Int64") )
    
    return cleaned

class DataCleaner:
    """ STateful outlier capper fitted exclusively on trained data.
    
    Compute IQR-based lower/upper bounds from ''X_train'' during ''fit'' and 
    applies those same bounds when transforming any split, preventing leakage
    of test-set statistics into the model.

    Usage (after train-test split)::

    cleaner = DataCleaner()
    X_train_cleaned = cleaner.fit_transform(X_train)
    X_test_cleaned = cleaner.transform(X_test)
    """

    def __init__(
            self,
            outlier_columns: list[str] | None = None,
            iqr_multiplier: float = 1.5,
    ) -> None:
        self.outlier_columns = outlier_columns or _OUTLIER_COLUMNS
        self.iqr_multiplier = iqr_multiplier
        self._bounds: dict[str, tuple[float, float]] = {}
    
    def fit(self, frame: pd.DataFrame) -> DataCleaner:
        """Compute IQR-based outlier bounds from the training data."""
        self._bounds = {}
        for col in self.outlier_columns:
            if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col]):
                q1 = frame[col].quantile(0.25)
                q3 = frame[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - self.iqr_multiplier * iqr
                upper_bound = q3 + self.iqr_multiplier * iqr
                self._bounds[col] = (lower_bound, upper_bound)
        return self
    
    def transform(self, frame: pd.DataFrame)   ->   pd.DataFrame:
        """Apply the outlier capping to the specified columns using the fitted bounds."""
        result = frame.copy()
        for col, (lower_bound, upper_bound) in self._bounds.items():
            if col in result.columns:
                result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
        return result
    
    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Convenience method to fit and transform in one step."""
        return self.fit(frame).transform(frame)

