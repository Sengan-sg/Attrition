from __future__ import annotations
import pandas as pd
from sklearn.model_selection import cross_validate

def validate_model(model, features: pd.DataFrame, target: pd.Series, scoring: list[str] | None = None) -> dict:
    scores = cross_validate(
        model,
        features,
        target,
        cv=5,
        scoring=scoring or ["accuracy", "f1", "roc_auc"],
        return_train_Score=False,
        n_jobs=-1
    )
    return {key: float(value.mean()) for key, value in scores.items() if key.startswith("test_")}