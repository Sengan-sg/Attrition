from __future__ import annotations
from pathlib import Path
import joblib

def load_model(path: str | Path):
    return joblib.load(Path(path))

def predict(model, records):
    return model.predict(records)