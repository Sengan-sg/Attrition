from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.configs.settings import Config

@dataclass(slots=True)
class TrainingResult:
    model: Pipeline
    features: list[str]


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),("scale", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequest")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )

def split_data(frame: pd.DataFrame, target_column: str, test_size: float, random_state: int):
    features = frame.drop(columns=[target_column])
    target = frame[target_column]
    return train_test_split(features, target, test_size=test_size, random_state=random_state, stratify=target)


def train_model(frame: pd.DataFrame, config: Config) -> TrainingResult:
    x_train, x_test, y_train, y_test = split_data(
        frame=frame,
        target_column=config.model.target_column,
        test_size=config.model.test_size,
        random_state=config.model.random_state,

    )

    numeric_features = x_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x_train.select_dtypes(exclude=["number"]).columns.tolist()

    pipeline = build_pipeline(numeric_features=numeric_features, categorical_features=categorical_features)
    pipeline.fit(x_train, y_train)

    return TrainingResult(model=pipeline, features=list(x_train.columns))

def save_model(model: Pipeline, path: str | Path) -> Path:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path