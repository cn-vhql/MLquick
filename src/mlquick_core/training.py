from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Callable
import warnings

import pandas as pd
from sklearn.metrics import silhouette_score

from .models import ModelMetadata, TrainingConfig, TrainingResult
from .registry import ModelRegistry
from .text import preprocess_text_column

ProgressCallback = Callable[[str], None]


class TrainingService:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def train(
        self,
        data: pd.DataFrame,
        config: TrainingConfig,
        progress: ProgressCallback | None = None,
    ) -> TrainingResult:
        if config.task_type == "classification":
            return self._train_classification(data, config, progress)
        if config.task_type == "regression":
            return self._train_regression(data, config, progress)
        if config.task_type == "clustering":
            return self._train_clustering(data, config, progress)
        raise ValueError(f"不支持的任务类型: {config.task_type}")

    def _emit(self, progress: ProgressCallback | None, message: str) -> None:
        if progress:
            progress(message)

    def _prepare_supervised_data(self, data: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
        if not config.target_column:
            raise ValueError("分类和回归任务必须选择目标列。")
        if not 0.1 <= config.train_size < 1.0:
            raise ValueError("训练集比例必须在 0.1 到 0.99 之间。")

        selected_columns = config.feature_columns or [col for col in data.columns if col != config.target_column]
        prepared = data[selected_columns + [config.target_column]].copy()

        if config.preprocess_text:
            text_columns = config.text_columns or prepared.select_dtypes(include="object").columns.tolist()
            text_columns = [col for col in text_columns if col != config.target_column]
            for column in text_columns:
                if column in prepared.columns:
                    prepared[column] = preprocess_text_column(
                        prepared[column],
                        remove_stopwords=config.remove_stopwords,
                        min_word_length=config.min_word_length,
                    )
        return prepared

    def _normalize_comparison_table(self, comparison: pd.DataFrame | None) -> pd.DataFrame:
        if comparison is None:
            return pd.DataFrame()
        normalized = comparison.copy()
        if normalized.index.name or normalized.index.dtype == "object":
            normalized = normalized.reset_index()
        return normalized

    def _extract_top_metrics(self, comparison: pd.DataFrame) -> dict[str, object]:
        if comparison.empty:
            return {}
        first_row = comparison.iloc[0].to_dict()
        return {key: value for key, value in first_row.items() if key not in {"index", "Model"}}

    def _detect_unavailable_supervised_models(self, task_type: str) -> list[str]:
        unavailable: list[str] = []

        # Detect whether optional gradient boosting backends are usable in current runtime
        # (especially in bundled onefile environments).
        try:
            import xgboost  # noqa: F401
        except Exception:
            unavailable.append("xgboost")
        try:
            import lightgbm  # noqa: F401
        except Exception:
            unavailable.append("lightgbm")

        # Keep only ids that PyCaret may use for this task.
        supported_ids = {"xgboost", "lightgbm"} if task_type in {"classification", "regression"} else set()
        return [model_id for model_id in unavailable if model_id in supported_ids]

    def _format_metrics(self, metrics: dict[str, object]) -> list[str]:
        lines: list[str] = []
        for key, value in metrics.items():
            lines.append(f"{key}: {value}")
        return lines

    def _resolve_supervised_fold(self, data: pd.DataFrame, target_column: str | None, task_type: str) -> int:
        row_count = len(data)
        if row_count < 3:
            raise ValueError("样本量过少，至少需要 3 条样本才能进行训练。")

        # PyCaret 默认 10 折，数据量不足时需要自动降级。
        fold_by_rows = min(10, row_count - 1)
        if task_type == "classification":
            if not target_column or target_column not in data.columns:
                raise ValueError("分类任务缺少有效目标列。")
            class_counts = data[target_column].value_counts(dropna=False)
            if class_counts.empty:
                raise ValueError("目标列为空，无法进行分类训练。")
            min_class_count = int(class_counts.min())
            if min_class_count < 2:
                raise ValueError("分类任务中每个类别至少需要 2 条样本。")
            return max(2, min(fold_by_rows, min_class_count))
        return max(2, fold_by_rows)

    def _resolve_classification_split(
        self,
        data: pd.DataFrame,
        target_column: str | None,
        requested_train_size: float,
    ) -> tuple[float, bool]:
        if not target_column or target_column not in data.columns:
            raise ValueError("分类任务缺少有效目标列。")
        n_rows = len(data)
        class_count = int(data[target_column].nunique(dropna=False))
        if n_rows < 2:
            raise ValueError("样本量过少，无法进行分类训练。")

        train_size = float(requested_train_size)
        train_size = max(0.1, min(0.95, train_size))

        # Prefer stratified split with integer-safe adjustment so both train/test
        # can contain all classes when feasible.
        min_rows_per_split = class_count
        if n_rows >= min_rows_per_split * 2:
            desired_test_rows = int(round((1.0 - train_size) * n_rows))
            desired_test_rows = max(min_rows_per_split, desired_test_rows)
            desired_test_rows = min(desired_test_rows, n_rows - min_rows_per_split)
            adjusted_train_size = 1.0 - (desired_test_rows / n_rows)
            return max(0.1, min(0.95, adjusted_train_size)), True

        # Not enough samples to guarantee both splits cover all classes.
        return max(0.5, train_size), False

    def _generate_plot_images(
        self,
        plot_model,
        model,
        model_name: str,
        plot_names: list[str],
        progress: ProgressCallback | None,
    ) -> list[str]:
        plot_dir = self.registry.models_dir / f"{model_name}_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        created_paths: list[str] = []
        cwd = Path.cwd()
        try:
            os.chdir(plot_dir)
            for plot_name in plot_names:
                existing = {path.resolve() for path in plot_dir.glob("*.png")}
                try:
                    self._emit(progress, f"正在生成图表: {plot_name}")
                    output = plot_model(model, plot=plot_name, save=True)
                except Exception as exc:  # pragma: no cover
                    self._emit(progress, f"图表 {plot_name} 生成失败: {exc}")
                    continue

                candidates: list[Path] = []
                if isinstance(output, str):
                    output_path = Path(output)
                    if not output_path.is_absolute():
                        output_path = plot_dir / output_path
                    if output_path.exists():
                        candidates.append(output_path.resolve())

                new_files = [path.resolve() for path in plot_dir.glob("*.png") if path.resolve() not in existing]
                candidates.extend(new_files)
                if not candidates:
                    continue

                latest = max(candidates, key=lambda path: path.stat().st_mtime)
                latest_text = str(latest)
                if latest_text not in created_paths:
                    created_paths.append(latest_text)
        finally:
            os.chdir(cwd)
        return created_paths

    def _train_classification(
        self,
        data: pd.DataFrame,
        config: TrainingConfig,
        progress: ProgressCallback | None,
    ) -> TrainingResult:
        from pycaret.classification import (
            compare_models,
            create_model,
            finalize_model,
            plot_model,
            predict_model,
            pull,
            save_model,
            setup,
        )

        prepared = self._prepare_supervised_data(data, config)
        effective_train_size, use_stratify = self._resolve_classification_split(
            prepared,
            config.target_column,
            config.train_size,
        )
        fold_count = self._resolve_supervised_fold(prepared, config.target_column, "classification")
        self._emit(progress, "正在初始化分类训练环境...")
        setup(
            data=prepared,
            target=config.target_column,
            session_id=123,
            normalize=True,
            train_size=effective_train_size,
            data_split_stratify=use_stratify,
            text_features=(config.text_columns or None) if config.preprocess_text else None,
            fold=fold_count,
            verbose=False,
            n_jobs=1,
        )
        model, comparison, train_mode_text = self._run_supervised_training(
            compare_models=compare_models,
            create_model=create_model,
            pull=pull,
            score_key="Accuracy",
            config=config,
            progress=progress,
        )
        preview = None
        try:
            preview = self._build_supervised_preview(
                holdout_predictions=predict_model(model),
                target_column=config.target_column,
                task_type="classification",
            )
        except Exception as exc:
            self._emit(progress, f"测试集预测预览生成失败，已跳过: {exc}")
        model = finalize_model(model)

        model_name = self.registry.generate_model_name("classification")
        save_model(model, str(self.registry.model_path(model_name)))
        plot_paths = self._generate_plot_images(
            plot_model=plot_model,
            model=model,
            model_name=model_name,
            plot_names=["auc", "confusion_matrix", "feature"],
            progress=progress,
        )

        metadata = ModelMetadata(
            model_name=model_name,
            task_type="classification",
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_path=str(self.registry.model_path(model_name).with_suffix(".pkl")),
            metrics=self._extract_top_metrics(comparison),
            target_column=config.target_column,
            feature_columns=[col for col in prepared.columns if col != config.target_column],
            text_columns=config.text_columns or [],
            notes=f"{train_mode_text}，并保存最终分类模型。",
        )
        self.registry.save_metadata(metadata)
        summary_lines = [
            f"已完成分类训练，模型名称：{model_name}",
            f"目标列：{config.target_column}",
            f"训练策略：{train_mode_text}",
            f"交叉验证折数：{fold_count}",
            f"训练集比例：{effective_train_size:.0%}",
            f"参与训练字段数：{len(metadata.feature_columns)}",
        ]
        if metadata.metrics:
            summary_lines.append("关键指标：")
            summary_lines.extend(self._format_metrics(metadata.metrics))
        if plot_paths:
            summary_lines.append(f"已生成图表数量: {len(plot_paths)}")
        return TrainingResult(
            metadata=metadata,
            comparison=comparison,
            predictions_preview=preview,
            summary_lines=summary_lines,
            plot_paths=plot_paths,
        )

    def _train_regression(
        self,
        data: pd.DataFrame,
        config: TrainingConfig,
        progress: ProgressCallback | None,
    ) -> TrainingResult:
        from pycaret.regression import (
            compare_models,
            create_model,
            finalize_model,
            plot_model,
            predict_model,
            pull,
            save_model,
            setup,
        )

        prepared = self._prepare_supervised_data(data, config)
        fold_count = self._resolve_supervised_fold(prepared, config.target_column, "regression")
        self._emit(progress, "正在初始化回归训练环境...")
        setup(
            data=prepared,
            target=config.target_column,
            train_size=config.train_size,
            text_features=(config.text_columns or None) if config.preprocess_text else None,
            fold=fold_count,
            verbose=False,
            n_jobs=1,
        )
        model, comparison, train_mode_text = self._run_supervised_training(
            compare_models=compare_models,
            create_model=create_model,
            pull=pull,
            score_key="R2",
            config=config,
            progress=progress,
        )
        preview = None
        try:
            preview = self._build_supervised_preview(
                holdout_predictions=predict_model(model),
                target_column=config.target_column,
                task_type="regression",
            )
        except Exception as exc:
            self._emit(progress, f"测试集预测预览生成失败，已跳过: {exc}")
        model = finalize_model(model)

        model_name = self.registry.generate_model_name("regression")
        save_model(model, str(self.registry.model_path(model_name)))
        plot_paths = self._generate_plot_images(
            plot_model=plot_model,
            model=model,
            model_name=model_name,
            plot_names=["residuals", "error", "feature"],
            progress=progress,
        )

        metadata = ModelMetadata(
            model_name=model_name,
            task_type="regression",
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_path=str(self.registry.model_path(model_name).with_suffix(".pkl")),
            metrics=self._extract_top_metrics(comparison),
            target_column=config.target_column,
            feature_columns=[col for col in prepared.columns if col != config.target_column],
            text_columns=config.text_columns or [],
            notes=f"{train_mode_text}，并保存最终回归模型。",
        )
        self.registry.save_metadata(metadata)
        summary_lines = [
            f"已完成回归训练，模型名称：{model_name}",
            f"目标列：{config.target_column}",
            f"训练策略：{train_mode_text}",
            f"交叉验证折数：{fold_count}",
            f"训练集比例：{config.train_size:.0%}",
            f"参与训练字段数：{len(metadata.feature_columns)}",
        ]
        if metadata.metrics:
            summary_lines.append("关键指标：")
            summary_lines.extend(self._format_metrics(metadata.metrics))
        if plot_paths:
            summary_lines.append(f"已生成图表数量: {len(plot_paths)}")
        return TrainingResult(
            metadata=metadata,
            comparison=comparison,
            predictions_preview=preview,
            summary_lines=summary_lines,
            plot_paths=plot_paths,
        )

    def _train_clustering(
        self,
        data: pd.DataFrame,
        config: TrainingConfig,
        progress: ProgressCallback | None,
    ) -> TrainingResult:
        from pycaret.clustering import assign_model, create_model, plot_model, save_model, setup

        selected_columns = config.feature_columns or data.select_dtypes(include="number").columns.tolist()
        numeric_data = data[selected_columns].select_dtypes(include="number").copy()
        if numeric_data.empty:
            raise ValueError("聚类任务当前仅支持数值型特征，请至少选择一个数值字段。")

        self._emit(progress, "正在初始化聚类训练环境...")
        setup(data=numeric_data, session_id=123, normalize=True, verbose=False)
        self._emit(progress, "正在训练聚类模型...")
        model = create_model("kmeans", num_clusters=config.cluster_count)
        clustered = assign_model(model)
        cluster_stats = clustered.groupby("Cluster").mean(numeric_only=True).round(4)

        metrics: dict[str, object] = {
            "cluster_count": config.cluster_count,
            "sample_count": len(numeric_data),
        }
        if len(clustered["Cluster"].unique()) > 1:
            metrics["silhouette"] = round(float(silhouette_score(numeric_data, clustered["Cluster"])), 4)

        model_name = self.registry.generate_model_name("clustering")
        save_model(model, str(self.registry.model_path(model_name)))
        plot_paths = self._generate_plot_images(
            plot_model=plot_model,
            model=model,
            model_name=model_name,
            plot_names=["elbow", "silhouette", "distribution"],
            progress=progress,
        )
        clustered.to_csv(
            self.registry.models_dir / f"{model_name}_clusters.csv",
            index=False,
            encoding="utf-8-sig",
        )

        metadata = ModelMetadata(
            model_name=model_name,
            task_type="clustering",
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_path=str(self.registry.model_path(model_name).with_suffix(".pkl")),
            metrics=metrics,
            feature_columns=numeric_data.columns.tolist(),
            notes="K-means 聚类模型，已同步导出聚类样本结果。",
        )
        self.registry.save_metadata(metadata)
        summary_lines = [
            f"已完成聚类训练，模型名称：{model_name}",
            f"特征数量：{len(numeric_data.columns)}",
            f"聚类数量：{config.cluster_count}",
        ]
        if metadata.metrics:
            summary_lines.append("关键指标：")
            summary_lines.extend(self._format_metrics(metadata.metrics))
        if plot_paths:
            summary_lines.append(f"已生成图表数量: {len(plot_paths)}")
        return TrainingResult(
            metadata=metadata,
            predictions_preview=clustered.head(50),
            cluster_stats=cluster_stats,
            summary_lines=summary_lines,
            plot_paths=plot_paths,
        )

    def _run_supervised_training(
        self,
        compare_models,
        create_model,
        pull,
        score_key: str,
        config: TrainingConfig,
        progress: ProgressCallback | None,
    ):
        if config.training_mode == "manual_single":
            if not config.manual_model_id:
                raise ValueError("手动模式下必须指定模型。")
            self._emit(progress, f"正在训练手动指定模型: {config.manual_model_id}")
            try:
                model = create_model(config.manual_model_id, verbose=False)
            except Exception as exc:
                message = str(exc)
                split_errors = (
                    "n_splits",
                    "number of members in each class",
                    "Cannot have number of splits",
                )
                if any(token in message for token in split_errors):
                    self._emit(progress, "样本过小，交叉验证自动降级为禁用后重试。")
                    model = create_model(config.manual_model_id, verbose=False, cross_validation=False)
                else:
                    raise ValueError(f"指定模型不可用或依赖缺失: {config.manual_model_id} ({exc})") from exc
            metrics_table = self._normalize_comparison_table(pull())
            summary_row = self._extract_summary_row(metrics_table)
            comparison = pd.DataFrame([summary_row.to_dict()])
            comparison.insert(0, "Model", str(config.manual_model_id))
            return model, comparison, f"手动指定模型({config.manual_model_id})"

        self._emit(progress, "正在计算模型排行榜...")
        # 保留 PyCaret 对比输出的完整明细（所有候选模型、全部指标列）。
        exclude_models = self._detect_unavailable_supervised_models(config.task_type)
        if exclude_models:
            self._emit(progress, f"检测到不可用模型依赖，已自动排除: {', '.join(exclude_models)}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_model = compare_models(
                sort=score_key,
                n_select=1,
                turbo=False,
                verbose=False,
                exclude=exclude_models or None,
            )
        comparison = self._normalize_comparison_table(pull())
        if comparison.empty:
            raise ValueError("PyCaret 模型对比结果为空，请检查数据或配置。")
        if not comparison.empty and "Model" in comparison.columns:
            leading_columns = ["Model"] + [col for col in comparison.columns if col != "Model"]
            comparison = comparison[leading_columns]
        return best_model, comparison, "自动对比候选模型"

    def _build_supervised_preview(
        self,
        holdout_predictions: pd.DataFrame | None,
        target_column: str | None,
        task_type: str,
    ) -> pd.DataFrame | None:
        if holdout_predictions is None or holdout_predictions.empty or not target_column:
            return None

        predicted_column = None
        for candidate in ("prediction_label", "Label", "prediction", "Predicted"):
            if candidate in holdout_predictions.columns:
                predicted_column = candidate
                break
        if predicted_column is None or target_column not in holdout_predictions.columns:
            return None

        preview = pd.DataFrame(
            {
                "实际值": holdout_predictions[target_column],
                "预测值": holdout_predictions[predicted_column],
            }
        )
        if task_type == "classification":
            for score_col in ("prediction_score", "Score", "prediction_score_1"):
                if score_col in holdout_predictions.columns:
                    preview["预测置信度"] = holdout_predictions[score_col]
                    break
        return preview.head(100)

    def _extract_summary_row(self, metrics_table: pd.DataFrame) -> pd.Series:
        if metrics_table is None or metrics_table.empty:
            raise ValueError("未获取到模型评估指标，请检查训练数据。")
        if "Fold" in metrics_table.columns:
            mean_rows = metrics_table[metrics_table["Fold"].astype(str) == "Mean"]
            if not mean_rows.empty:
                return mean_rows.iloc[0]
            std_rows = metrics_table[metrics_table["Fold"].astype(str) == "Std"]
            if not std_rows.empty:
                return std_rows.iloc[0]
        return metrics_table.iloc[-1]
