from __future__ import annotations
from scipy import stats
import pandas as pd


def compare_numeric_groups(frame: pd.DataFrame, feature: str, group_column: str) -> dict:
    """Compare numeric feature across different groups using statistical tests.

    Args:
        frame: The input DataFrame.
        feature: The name of the numeric feature to compare.
        group_column: The name of the column containing group labels.

    Returns:
        A dictionary containing the results of the statistical tests.
    """
    groups = [group.dropna().values for _, group in frame.groupby(group_column)[feature]]
    if len(groups) != 2:
        return {"test": "unsupported", "reason": "This helper currently supports only two groups for comparison."}
    
    t_stat, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=False)
    
    return {"test": "welch_ttest", "t_statistic":float(t_stat), "p_value": float(p_value)}