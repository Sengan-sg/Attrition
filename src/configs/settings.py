from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]

@dataclass(slots=True)
class ProjectPaths:
    root: Path = _project_root()
    data_raw: Path = field(default_factory=lambda: _project_root() / "data" / "raw")
    data_processed: Path = field(default_factory=lambda: _project_root() / "data" / "processed")
    models: Path = field(default_factory=lambda: _project_root() / "models")
    reports: Path = field(default_factory=lambda: _project_root() / "reports")

@dataclass(slots=True)
class ModelConfig:
    target_column: str = "Attrition"
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2

@dataclass(slots=True)
class Config:
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    model: ModelConfig = field(default_factory=ModelConfig)