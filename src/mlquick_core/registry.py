from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .models import ModelMetadata


class ModelRegistry:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.predictions_dir = self.base_dir / "predictions"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def generate_model_name(self, task_type: str) -> str:
        return f"{task_type}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def metadata_path(self, model_name: str) -> Path:
        return self.models_dir / f"{model_name}.json"

    def model_path(self, model_name: str) -> Path:
        return self.models_dir / model_name

    def save_metadata(self, metadata: ModelMetadata) -> None:
        self.metadata_path(metadata.model_name).write_text(
            json.dumps(asdict(metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_metadata(self, model_name: str) -> ModelMetadata:
        payload = json.loads(self.metadata_path(model_name).read_text(encoding="utf-8"))
        return ModelMetadata(**payload)

    def list_models(self) -> list[ModelMetadata]:
        items: list[ModelMetadata] = []
        for path in sorted(self.models_dir.glob("*.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                items.append(ModelMetadata(**payload))
            except (json.JSONDecodeError, TypeError):
                continue
        return items
