from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def create_eda_overview(frame: pd.DataFrame, target_column: str, output_dir: str | Path) -> dict[str, Path]:
    """Create EDA report for the given DataFrame and target column.

    Args:
        frame: The input DataFrame.
        target_column: The name of the target column.
        output_dir: The directory where the EDA report will be saved.
    Returns:
        A dictionary containing paths to the generated EDA report files.
        """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}

    numeric_frame = frame.select_dtypes(include=['number'])
    if not numeric_frame.empty:
        corr_path = output_path / 'correlation_heatmap.png'
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_frame.corr(numeric_only=True), annot=False, cmap='coolwarm', fmt='.2f')
        plt.tight_layout()
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        artifacts['correlation_heatmap'] = corr_path
    
    if target_column in frame.columns:
        target_dist_path = output_path / f'{target_column}_distribution.png'
        plt.figure(figsize=(8, 6))
        sns.countplot(x=target_column, data=frame)
        plt.title(f'Distribution of {target_column}')
        plt.tight_layout()
        plt.savefig(target_dist_path, dpi=300, bbox_inches='tight')
        artifacts['target_distribution'] = target_dist_path

    #Missing value bar chart
    missing_pct = frame.isnull().mean() * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    if not missing_pct.empty:
        missing_path = output_path / 'missing_values.png'
        plt.figure(figsize=(12, 6))
        missing_pct.plot(kind="bar", color="steelblue")
        plt.axhline(y=40, color="red", linestyle="--", label="40 % threshold")
        plt.ylabel("Missing (%)")
        plt.title("Percentage of Missing Values by Column")
        plt.legend()
        plt.tight_layout()
        plt.savefig(missing_path, dpi=150)
        plt.close()
        artifacts["missing_values"] = missing_path

#outlier box plot for salary and tenure columns
    outlier_cols = [
        col for col in [
            "MonthlyIncome", "TotalWorkingYears", "NumCompaniesWorked", "TrainingTimesLastYear"
            ] 
        if col in frame.columns
                    ]
    if outlier_cols:
        outlier_path = output_path / 'outliers.png'
        fig, axes = plt.subplots(1, len(outlier_cols), figsize=(4 * len(outlier_cols), 5))
        if len(outlier_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, outlier_cols):
            sns.boxplot(y=frame[col], color="lightcoral", ax=ax)
            ax.set_title(col)
        plt.suptitle("Outlier detection - Box plot")
        plt.tight_layout()
        plt.savefig(outlier_path, dpi=150)
        plt.close()
        artifacts["outlier_boxplots"] = outlier_path

    #Key categorical columns vs target (Attrition rate breakdown)
    cat_cols = [
        c for c in ["Department",  "EducationField",  "JobRole",  "MaritalStatus"]
        if c in frame.columns and c != target_column
    ]
    for col in cat_cols:
        cat_path = output_path / f"{col}_vs_{target_column}.png"
        fig, axes = plt.subplots(1,2, figsize=(14, 5))

        #Raw Counts
        ct_counts = pd.crosstab(frame[col], frame[target_column])
        ct_counts.plot(kind="bar", ax=axes[0])
        axes[0].set_title(f"{col} - Count by {target_column}")
        axes[0].set_xlabel("")
        axes[0].tick_params(axis="x", rotation=45)

        #Normalised (attrition rate per category)
        ct_norm = pd.crosstab(frame[col], frame[target_column], normalize="index") * 100
        ct_norm.plot(kind="bar", stacked=True, ax=axes[1])
        axes[1].set_ylabel("Percentage (%)")
        axes[1].set_title(f"{col} - Attrition rate(%)")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(cat_path, dpi=150)
        plt.close()
        artifacts[f"{col}_vs_attrition"] = cat_path

    return artifacts