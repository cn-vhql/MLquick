#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管理组件 - 处理数据加载、预处理和管理
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import warnings

class DataManager:
    """数据管理器"""

    def __init__(self):
        self.current_data = None
        self.data_info = {}

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        加载数据文件

        Args:
            file_path: 文件路径

        Returns:
            加载的DataFrame

        Raises:
            Exception: 文件加载失败时抛出异常
        """
        try:
            if file_path.endswith('.csv'):
                # 尝试不同的编码
                encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'iso-8859-1']

                for encoding in encodings:
                    try:
                        data = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise Exception("无法读取CSV文件，请检查文件编码")

            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                data = pd.read_excel(file_path, engine='openpyxl')
            else:
                raise Exception("不支持的文件格式，请使用CSV或Excel文件")

            # 基本数据清理
            data = self._basic_cleaning(data)

            # 更新数据信息
            self.current_data = data
            self._update_data_info(data)

            return data

        except Exception as e:
            raise Exception(f"数据加载失败: {str(e)}")

    def _basic_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        基本数据清理

        Args:
            data: 原始数据

        Returns:
            清理后的数据
        """
        # 去除完全为空的行和列
        data = data.dropna(how='all').dropna(axis=1, how='all')

        # 重置索引
        data = data.reset_index(drop=True)

        return data

    def _update_data_info(self, data: pd.DataFrame):
        """更新数据信息"""
        self.data_info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'numeric_columns': self.get_numeric_columns(data),
            'text_columns': self.get_text_columns(data),
            'missing_values': data.isnull().sum().to_dict()
        }

    def get_numeric_columns(self, data: Optional[pd.DataFrame] = None) -> List[str]:
        """获取数值型列名"""
        if data is None:
            data = self.current_data

        if data is None:
            return []

        return data.select_dtypes(include=[np.number]).columns.tolist()

    def get_text_columns(self, data: Optional[pd.DataFrame] = None) -> List[str]:
        """获取文本型列名"""
        if data is None:
            data = self.current_data

        if data is None:
            return []

        return data.select_dtypes(include=['object']).columns.tolist()

    def get_data_summary(self, data: Optional[pd.DataFrame] = None) -> dict:
        """获取数据摘要信息"""
        if data is None:
            data = self.current_data

        if data is None:
            return {}

        summary = {
            'shape': data.shape,
            'columns': list(data.columns),
            'numeric_columns': len(self.get_numeric_columns(data)),
            'text_columns': len(self.get_text_columns(data)),
            'missing_values': data.isnull().sum().sum(),
            'memory_usage': data.memory_usage(deep=True).sum()
        }

        return summary

    def validate_data_for_task(self, data: pd.DataFrame, task_type: str,
                              target_column: Optional[str] = None) -> Tuple[bool, str]:
        """
        验证数据是否适合特定任务

        Args:
            data: 数据
            task_type: 任务类型 ('classification', 'regression', 'clustering')
            target_column: 目标列名（分类和回归任务需要）

        Returns:
            (是否有效, 错误信息)
        """
        if data is None or data.empty:
            return False, "数据为空"

        if task_type in ['classification', 'regression']:
            if not target_column:
                return False, "请指定目标变量"

            if target_column not in data.columns:
                return False, f"目标列 '{target_column}' 不存在"

            # 检查目标列的缺失值
            if data[target_column].isnull().all():
                return False, f"目标列 '{target_column}' 全为缺失值"

            # 对于分类任务，检查目标变量的唯一值数量
            if task_type == 'classification':
                unique_values = data[target_column].nunique()
                if unique_values < 2:
                    return False, "分类任务的目标变量至少需要2个不同类别"
                if unique_values > 100:
                    return False, f"目标变量类别过多 ({unique_values} 个)，可能不适合分类任务"

            # 对于回归任务，检查目标变量是否为数值型
            elif task_type == 'regression':
                if not pd.api.types.is_numeric_dtype(data[target_column]):
                    return False, "回归任务的目标变量必须是数值型"

        elif task_type == 'clustering':
            # 聚类任务检查是否有足够的特征
            numeric_cols = self.get_numeric_columns(data)
            if len(numeric_cols) < 1:
                return False, "聚类任务至少需要1个数值型特征"

        return True, ""

    def get_sample_data(self, n_rows: int = 10) -> Optional[pd.DataFrame]:
        """获取样本数据"""
        if self.current_data is None:
            return None

        return self.current_data.head(n_rows)

    def get_column_info(self, column: str) -> dict:
        """获取列的详细信息"""
        if self.current_data is None or column not in self.current_data.columns:
            return {}

        col_data = self.current_data[column]

        info = {
            'name': column,
            'dtype': str(col_data.dtype),
            'non_null_count': col_data.count(),
            'null_count': col_data.isnull().sum(),
            'unique_count': col_data.nunique()
        }

        # 数值型列的额外信息
        if pd.api.types.is_numeric_dtype(col_data):
            info.update({
                'min': col_data.min(),
                'max': col_data.max(),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'median': col_data.median()
            })

        # 文本型列的额外信息
        elif pd.api.types.is_object_dtype(col_data):
            non_null_data = col_data.dropna()
            if len(non_null_data) > 0:
                avg_length = non_null_data.astype(str).str.len().mean()
                info.update({
                    'avg_length': avg_length,
                    'most_common': col_data.value_counts().head(5).to_dict()
                })

        return info

    def export_data(self, file_path: str, data: Optional[pd.DataFrame] = None):
        """导出数据到文件"""
        if data is None:
            data = self.current_data

        if data is None:
            raise Exception("没有可导出的数据")

        try:
            if file_path.endswith('.csv'):
                data.to_csv(file_path, index=False, encoding='utf-8-sig')
            elif file_path.endswith('.xlsx'):
                data.to_excel(file_path, index=False, engine='openpyxl')
            else:
                raise Exception("不支持的导出格式")
        except Exception as e:
            raise Exception(f"数据导出失败: {str(e)}")