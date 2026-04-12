from __future__ import annotations

import pandas as pd

from .models import DataProfile


def profile_dataset(data: pd.DataFrame) -> DataProfile:
    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in data.columns if column not in numeric_columns]

    candidate_targets: list[str] = []
    warnings: list[str] = []
    row_count = len(data)

    for column in data.columns:
        unique_count = data[column].nunique(dropna=True)
        missing_ratio = float(data[column].isna().mean()) if row_count else 0.0

        if missing_ratio > 0.4:
            warnings.append(f"{column} 缺失率较高（{missing_ratio:.0%}），建议谨慎使用。")

        if unique_count == row_count and row_count > 20:
            warnings.append(f"{column} 很像唯一编号列，通常不建议直接参与建模。")
            continue

        if unique_count <= min(20, max(2, row_count // 5)):
            candidate_targets.append(column)
        elif pd.api.types.is_numeric_dtype(data[column]) and unique_count > min(20, row_count // 3):
            candidate_targets.append(column)

    recommended_tasks: list[str] = []
    if len(numeric_columns) >= 2:
        recommended_tasks.append("clustering")

    has_low_cardinality_target = any(
        data[column].nunique(dropna=True) <= min(20, max(2, row_count // 5))
        for column in data.columns
    )
    if has_low_cardinality_target:
        recommended_tasks.append("classification")

    has_numeric_target = any(
        pd.api.types.is_numeric_dtype(data[column])
        and data[column].nunique(dropna=True) > min(20, max(5, row_count // 3))
        for column in data.columns
    )
    if has_numeric_target:
        recommended_tasks.append("regression")

    return DataProfile(
        rows=row_count,
        columns=len(data.columns),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        missing_summary=data.isna().sum().to_dict(),
        duplicate_rows=int(data.duplicated().sum()),
        candidate_targets=candidate_targets,
        recommended_tasks=recommended_tasks,
        warnings=warnings,
    )
