from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .models import PredictionResult
from .registry import ModelRegistry
from .text import preprocess_text_column


class PredictionService:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def predict(self, model_name: str, data: pd.DataFrame) -> PredictionResult:
        metadata = self.registry.load_metadata(model_name)
        model_root = str(Path(metadata.model_path).with_suffix(""))

        if metadata.task_type == "classification":
            from pycaret.classification import load_model, predict_model

            prepared = self._prepare_supervised_features(data, metadata)
            model = load_model(model_root)
            predictions = predict_model(model, data=prepared)
        elif metadata.task_type == "regression":
            from pycaret.regression import load_model, predict_model

            prepared = self._prepare_supervised_features(data, metadata)
            model = load_model(model_root)
            predictions = predict_model(model, data=prepared)
        elif metadata.task_type == "clustering":
            from pycaret.clustering import load_model

            required_columns = metadata.feature_columns
            missing = [column for column in required_columns if column not in data.columns]
            if missing:
                raise ValueError(f"待预测数据缺少聚类特征列: {', '.join(missing)}")
            model = load_model(model_root)
            prepared = data[required_columns].copy()
            labels = model.predict(prepared)
            predictions = prepared.copy()
            predictions["Cluster"] = labels
        else:
            raise ValueError(f"不支持的任务类型: {metadata.task_type}")

        output_path = (
            self.registry.predictions_dir
            / f"{model_name}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        predictions.to_csv(output_path, index=False, encoding="utf-8-sig")
        return PredictionResult(metadata=metadata, predictions=predictions, output_path=output_path)

    def _prepare_supervised_features(self, data: pd.DataFrame, metadata) -> pd.DataFrame:
        required_columns = metadata.feature_columns
        missing = [column for column in required_columns if column not in data.columns]
        if missing:
            raise ValueError(f"待预测数据缺少特征列: {', '.join(missing)}")

        prepared = data[required_columns].copy()
        for column in metadata.text_columns:
            if column in prepared.columns:
                prepared[column] = preprocess_text_column(prepared[column])
        return prepared
