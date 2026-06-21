from __future__ import annotations
import pandas as pd

# Ordinal order: more travel -> higher value (risk proxy)
_TRAVEL_ORDINAL: dict[str, int] = {
    "Non-Travel": 0,
    "Travel_Rarely": 1,
    "Travel_Frequently": 2,
}

# Binary gender mapping
_GENDER_MAP: dict[str, int] = {"Male": 1, "Female": 0}


def encode_ordinal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode ordinal features in the DataFrame. Meaningful order
    '''BusinessTravel''' is encoded as Non-Travel < Travel_Rarely < Travel_Frequently.
    Args:
        frame: The input DataFrame.

    Returns:
        A DataFrame with ordinal features encoded as integers.
    """
    result = frame.copy()
    if 'BusinessTravel' in result.columns:
        result['BusinessTravelEncoded'] = (
            result['BusinessTravel'].map(_TRAVEL_ORDINAL).astype("Int64")
        )
    return result

def encode_binary_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode binary features in the DataFrame."""
    result = frame.copy()
    if "Gender" in result.columns:
        result["GenderEncoded"] = (
            result["Gender"].map(_GENDER_MAP).astype("Int64")
        )
    return result

def create_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create new features derived from existing ones to capture additional information."""
    result = frame.copy()
    # Example derived feature: Age multiplied by YearsAtCompany
    if {"YearsAtCompany", "TotalWorkingYears"}.issubset(result.columns):
        result["TenureRatio"] = result["YearsAtCompany"] / (result["TotalWorkingYears"] + 1)
    
    if {"TotalWorkingYears", "NumCompaniesWorked"}.issubset(result.columns):
        result["YearsPerCompany"] = result["TotalWorkingYears"] / (result["NumCompaniesWorked"] + 1)

    if {"MonthlyIncome", "JobLevel"}.issubset(result.columns):
        result["IncomePerLevel"] = result["MonthlyIncome"] / (result["JobLevel"] + 1)
    
    sat_cols = [
        c for c in ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction"]
        if c in result.columns
    ]

    if sat_cols:
        result["AvgSatisfaction"] = result[sat_cols].mean(axis=1)
    
    if {"YearsSinceLastPromotion", "YearsAtCompany"}.issubset(result.columns):
        result["PromotionLag"] = result["YearsSinceLastPromotion"] / (result["YearsAtCompany"] + 1)

    return result





def engineering_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create new features based on existing ones to enhance model performance.

    Args:
        frame: The input DataFrame.

    Returns:
        A DataFrame with new engineered features.
    """
    frame = encode_ordinal_features(frame)
    frame = encode_binary_features(frame)
    frame = create_derived_features(frame)
    return frame