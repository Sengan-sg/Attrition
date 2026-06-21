from __future__ import annotations
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

@dataclass(slots=True)
class EvaluationResult:
    accuracy: float
    roc_auc: float | None
    confusion: list[list[int]]
    report: dict

def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> EvaluationResult:
    """Evaluate the given model on the test data.

    Args:
        model: The trained model to evaluate.
        x_test: The test features.
        y_test: The true labels for the test data.

    Returns:
        An EvaluationResult object containing accuracy, ROC AUC, confusion matrix, and classification report.
    """
    predictions = model.predict(x_test)
    probabilites = None
    if hasattr(model, "predict_proba"):
        probabilites = model.predict_proba(x_test)[:, 1]


    return EvaluationResult(
        accuracy=accuracy_score(y_test, predictions),
          roc_auc=roc_auc_score(y_test, probabilites) if probabilites is not None else None,
          confusion=confusion_matrix(y_test, predictions).tolist(),
          report=classification_report(y_test, predictions, output_dict=True)
    )