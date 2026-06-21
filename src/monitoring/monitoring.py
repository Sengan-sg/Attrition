from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MonitoringSnapshot:
    timestamp: datetime
    accuracy: float | None = None
    data_drift_score: float | None = None
    retraining_required: bool = False

def should_retrain(snapshot: MonitoringSnapshot, accuracy_threshold: float = 0.8, drift_threshold: float = 0.2) -> bool:
    if snapshot.accuracy is not None and snapshot.accuracy < accuracy_threshold:
        return True
    if snapshot.data_drift_score is not None and snapshot.data_drift_Score > drift_threshold:
        return True
    return snapshot.retraining_required
