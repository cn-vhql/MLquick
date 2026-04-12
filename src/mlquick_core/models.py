from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DataProfile:
    rows: int
    columns: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    missing_summary: dict[str, int]
    duplicate_rows: int
    candidate_targets: list[str]
    recommended_tasks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    task_type: str
    target_column: str | None = None
    train_size: float = 0.7
    feature_columns: list[str] | None = None
    training_mode: str = "auto_compare"
    manual_model_id: str | None = None
    cluster_count: int = 3
    preprocess_text: bool = False
    text_columns: list[str] | None = None
    remove_stopwords: bool = True
    min_word_length: int = 2


@dataclass
class ModelMetadata:
    model_name: str
    task_type: str
    created_at: str
    model_path: str
    metrics: dict[str, Any]
    target_column: str | None = None
    feature_columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class TrainingResult:
    metadata: ModelMetadata
    comparison: pd.DataFrame | None = None
    predictions_preview: pd.DataFrame | None = None
    cluster_stats: pd.DataFrame | None = None
    summary_lines: list[str] = field(default_factory=list)
    plot_paths: list[str] = field(default_factory=list)


@dataclass
class PredictionResult:
    metadata: ModelMetadata
    predictions: pd.DataFrame
    output_path: Path | None = None
