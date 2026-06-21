from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.analysis.eda import create_eda_overview
from src.configs.settings import Config
from src.data_processing.cleaning import DataCleaner, clean_raw_data
from src.data_processing.loader import load_dataset
from src.evaluation.metrics import evaluate_model
from src.features.engineering import engineering_features
from src.features.selection import CorrelationSelector, VarianceSelector, select_features
from src.models.training import build_pipeline, save_model
from src.utils import ensure_directory, get_logger

logger = get_logger(__name__)

def run_pipeline(dataset_path: str | Path, output_model_path: str | Path | None = None):
    config = Config()

    #Load RAW data
    frame = load_dataset(dataset_path)
    logger.info("Loading dataset: %d rows, %d columns", *frame.shape)

    # Pre-split structural cleanup
    frame = clean_raw_data(frame)
    logger.info("After structural cleanup: %d rows, %d columns", *frame.shape)

    #EDA on full dataset
    create_eda_overview(frame, config.model.target_column, output_dir=config.paths.reports / "eda")

    # Train / Test split

    target = config.model.target_column
    X = frame.drop(columns=[target])
    y = frame[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.model.test_size,
        random_state=config.model.random_state,
        stratify=y,
        )
    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

    # Post split outlier capping

    cleaner = DataCleaner()
    X_train = cleaner.fit_transform(X_train)
    X_test = cleaner.transform(X_test)

    # Feature engineering

    X_train = engineering_features(X_train)
    X_test = engineering_features(X_test)

    # Feature selection
    var_selector = VarianceSelector(threshold=0.01)
    X_train = var_selector.fit_transform(X_train)
    X_test = var_selector.transform(X_test)

    corr_selector = CorrelationSelector(threshold=0.90)
    X_train = corr_selector.fit_transform(X_train)
    X_test = corr_selector.transform(X_test)

    logger.info("Features after selection: %d", X_train.shape[1])


    # Build sklearn pipeline

    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    pipeline = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )


    # Stratified K-fold cross-validaiton
    kfold = StratifiedKFold(
        n_split=5, shuffle=True, random_state=config.model.random_state
    )

    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=kfold,
        scoring=["accuracy", "roc_auc", "f1"],
        return_train_score=True,
    )
    logger.info(
       "5-Fold CV | Accuracy %.4f +- %.4f | ROC-AUC %.4f +_ %.4f | F1 %.4f +_ %.4f",
       cv_results["test_accuracy"].mean(), cv_results["test_accuracy"].std(),
       cv_results["test_roc_auc"].mean(), cv_results["test_roc_auc"].std(),
       cv_results["test_roc_auc"].mean(), cv_results["test_roc_auc"].std(),
    )

    #Final fit on full training set
    pipeline.fit(X_train, y_train)
    eval_result = evaluate_model(pipeline, X_test, y_test)
    logger.info(
        "Hold-out test  |Accuracy %.4f | ROC-AUC %.4f",
        eval_result.accuracy,
        eval_result.roc_auc or 0.0
    )

    if output_model_path is not None:
        save_model(pipeline, output_model_path)
    
    logger.info("Pipeline completed succesfully")
    return pipeline

def main():
    config = Config()
    ensure_directory(config.paths.models)
    dataset_path = config.paths.data_raw / "employee_attrition.csv"
    model_path = config.paths.models / "attrition_models.joblib"
    run_pipeline(dataset_path=dataset_path, output_model_path=model_path)

if __name__=="__main__":
    main() 