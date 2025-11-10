#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习引擎组件 - 处理模型训练和预测
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
from typing import Dict, Any, Optional, List
import warnings

# 导入PyCaret
try:
    from pycaret.classification import setup as clf_setup, compare_models as clf_compare_models
    from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models
    from pycaret.clustering import setup as cluster_setup, create_model, assign_model
    from pycaret.classification import pull as clf_pull
    from pycaret.regression import pull as reg_pull
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    warnings.warn("PyCaret未安装，机器学习功能将不可用")

# 导入文本处理组件
from .text_processor import TextProcessor

class MLEngine:
    """机器学习引擎"""

    def __init__(self):
        self.text_processor = TextProcessor()
        self.current_model = None
        self.model_info = {}

    def train_model(self, data: pd.DataFrame, task_type: str,
                   target_variable: Optional[str] = None,
                   train_size: float = 0.7,
                   n_clusters: int = 3,
                   enable_text_processing: bool = False,
                   text_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        训练机器学习模型

        Args:
            data: 训练数据
            task_type: 任务类型 ('classification', 'regression', 'clustering')
            target_variable: 目标变量（分类和回归任务）
            train_size: 训练集比例
            n_clusters: 聚类数量
            enable_text_processing: 是否启用文本处理
            text_columns: 文本列名列表

        Returns:
            训练结果字典
        """
        if not PYCARET_AVAILABLE:
            raise Exception("PyCaret未安装，无法训练模型")

        start_time = time.time()

        try:
            # 处理文本特征
            processed_data = data.copy()
            text_processing_info = {}

            if enable_text_processing:
                processed_data, text_processing_info = self._process_text_features(
                    data, task_type, target_variable, text_columns
                )

            # 根据任务类型训练模型
            if task_type == "classification":
                result = self._train_classification_model(
                    processed_data, target_variable, train_size
                )
            elif task_type == "regression":
                result = self._train_regression_model(
                    processed_data, target_variable, train_size
                )
            elif task_type == "clustering":
                result = self._train_clustering_model(
                    processed_data, n_clusters
                )
            else:
                raise ValueError(f"不支持的任务类型: {task_type}")

            # 添加通用信息
            result.update({
                'training_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'duration_seconds': time.time() - start_time,
                'data_shape': data.shape,
                'text_processing_info': text_processing_info,
                'enable_text_processing': enable_text_processing
            })

            self.current_model = result['model']
            self.model_info = result

            return result

        except Exception as e:
            raise Exception(f"模型训练失败: {str(e)}")

    def _process_text_features(self, data: pd.DataFrame, task_type: str,
                             target_variable: Optional[str],
                             text_columns: Optional[List[str]]) -> tuple:
        """处理文本特征"""
        processed_data = data.copy()
        text_info = {"processed_columns": [], "method": "tfidf"}

        # 自动检测文本列
        if text_columns is None:
            text_columns = data.select_dtypes(include=['object']).columns.tolist()

            # 移除目标变量（如果存在）
            if target_variable and target_variable in text_columns:
                text_columns.remove(target_variable)

        # 处理每个文本列
        for col in text_columns:
            if col in data.columns:
                try:
                    # 预处理文本
                    processed_text = self.text_processor.preprocess_text_column(data[col])
                    processed_data[col] = processed_text

                    # 提取文本特征
                    text_features, feature_names = self.text_processor.extract_features(
                        processed_text, max_features=100
                    )

                    if text_features is not None:
                        # 将文本特征添加到数据中
                        text_features_df = pd.DataFrame(
                            text_features.toarray(),
                            columns=[f"{col}_{name}" for name in feature_names]
                        )
                        processed_data = pd.concat([processed_data, text_features_df], axis=1)
                        text_info["processed_columns"].append(col)

                except Exception as e:
                    warnings.warn(f"处理文本列 '{col}' 时出错: {str(e)}")

        return processed_data, text_info

    def _train_classification_model(self, data: pd.DataFrame, target_variable: str,
                                  train_size: float) -> Dict[str, Any]:
        """训练分类模型"""
        # 设置PyCaret环境
        try:
            # 尝试新版本PyCaret参数
            clf_setup(data=data, target=target_variable, session_id=123,
                     normalize=True, train_size=train_size, verbose=False)
        except TypeError:
            # 如果失败，尝试旧版本参数
            clf_setup(data=data, target=target_variable, session_id=123,
                     normalize=True, train_size=train_size, silent=True)

        # 比较模型
        best_model = clf_compare_models()

        # 获取模型比较结果
        model_comparison = clf_pull()
        best_model_name = str(best_model)

        # 获取性能指标
        metrics = {}
        if 'Accuracy' in model_comparison.index:
            metrics['Accuracy'] = model_comparison.loc['Accuracy', best_model_name]
        if 'AUC' in model_comparison.index:
            metrics['AUC'] = model_comparison.loc['AUC', best_model_name]
        if 'F1 Score' in model_comparison.index:
            metrics['F1 Score'] = model_comparison.loc['F1 Score', best_model_name]
        if 'Precision' in model_comparison.index:
            metrics['Precision'] = model_comparison.loc['Precision', best_model_name]
        if 'Recall' in model_comparison.index:
            metrics['Recall'] = model_comparison.loc['Recall', best_model_name]

        # 生成模型名称
        model_name = f"classification_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 生成可视化
        try:
            visualizations = self._create_classification_visualizations(model_comparison, metrics,
                                                                    best_model, data, target_variable)
            print(f"分类可视化生成成功，包含: {list(visualizations.keys())}")
            print(f"分类可视化详情: {visualizations}")
        except Exception as e:
            print(f"分类可视化生成失败: {e}")
            import traceback
            traceback.print_exc()
            visualizations = {}

        return {
            'model': best_model,
            'model_name': model_name,
            'task_type': 'classification',
            'target_variable': target_variable,
            'metrics': metrics,
            'comparison': model_comparison,
            'visualizations': visualizations
        }

    def _train_regression_model(self, data: pd.DataFrame, target_variable: str,
                              train_size: float) -> Dict[str, Any]:
        """训练回归模型"""
        # 设置PyCaret环境
        try:
            # 尝试新版本PyCaret参数
            reg_setup(data=data, target=target_variable, session_id=123,
                     train_size=train_size, verbose=False)
        except TypeError:
            # 如果失败，尝试旧版本参数
            reg_setup(data=data, target=target_variable, session_id=123,
                     train_size=train_size, silent=True)

        # 比较模型
        best_model = reg_compare_models()

        # 获取模型比较结果
        model_comparison = reg_pull()
        best_model_name = str(best_model)

        # 获取性能指标
        metrics = {}
        if 'R2' in model_comparison.index:
            metrics['R2'] = model_comparison.loc['R2', best_model_name]
        if 'RMSE' in model_comparison.index:
            metrics['RMSE'] = model_comparison.loc['RMSE', best_model_name]
        if 'MAE' in model_comparison.index:
            metrics['MAE'] = model_comparison.loc['MAE', best_model_name]
        if 'MAPE' in model_comparison.index:
            metrics['MAPE'] = model_comparison.loc['MAPE', best_model_name]

        # 生成模型名称
        model_name = f"regression_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 生成可视化
        try:
            visualizations = self._create_regression_visualizations(model_comparison, metrics,
                                                                  best_model, data, target_variable)
            print(f"回归可视化生成成功，包含: {list(visualizations.keys())}")
            print(f"回归可视化详情: {visualizations}")
        except Exception as e:
            print(f"回归可视化生成失败: {e}")
            import traceback
            traceback.print_exc()
            visualizations = {}

        return {
            'model': best_model,
            'model_name': model_name,
            'task_type': 'regression',
            'target_variable': target_variable,
            'metrics': metrics,
            'comparison': model_comparison,
            'visualizations': visualizations
        }

    def _train_clustering_model(self, data: pd.DataFrame, n_clusters: int) -> Dict[str, Any]:
        """训练聚类模型"""
        # 只使用数值型特征
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            raise Exception("聚类任务需要数值型特征")

        # 设置PyCaret环境
        try:
            # 尝试新版本PyCaret参数
            cluster_setup(data=numeric_data, session_id=123, normalize=True, verbose=False)
        except TypeError:
            # 如果失败，尝试旧版本参数
            cluster_setup(data=numeric_data, session_id=123, normalize=True, silent=True)

        # 创建K-means模型
        kmeans_model = create_model('kmeans', num_clusters=n_clusters)

        # 分配聚类标签
        clustered_data = assign_model(kmeans_model)

        # 计算聚类统计信息
        cluster_stats = self._calculate_cluster_stats(clustered_data, numeric_data.columns)

        # 生成模型名称
        model_name = f"clustering_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 生成可视化
        try:
            visualizations = self._create_clustering_visualizations(numeric_data, clustered_data['Cluster'], n_clusters)
            print(f"聚类可视化生成成功，包含: {list(visualizations.keys())}")
        except Exception as e:
            print(f"聚类可视化生成失败: {e}")
            visualizations = {}

        return {
            'model': kmeans_model,
            'model_name': model_name,
            'task_type': 'clustering',
            'n_clusters': n_clusters,
            'clustered_data': clustered_data,
            'cluster_stats': cluster_stats,
            'visualizations': visualizations,
            'metrics': {'n_clusters': n_clusters}
        }

    def _calculate_cluster_stats(self, clustered_data: pd.DataFrame,
                               numeric_columns: List[str]) -> pd.DataFrame:
        """计算聚类统计信息"""
        stats = clustered_data.groupby('Cluster')[numeric_columns].agg(['mean', 'std', 'count'])
        return stats.round(3)

    def _create_clustering_visualizations(self, data: pd.DataFrame, cluster_labels: pd.Series, n_clusters: int) -> Dict[str, Any]:
        """创建聚类可视化数据"""
        visualizations = {}

        print(f"开始创建聚类可视化，数据形状: {data.shape}, 聚类标签形状: {cluster_labels.shape}")

        # 获取数值型列
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"数值型列: {numeric_columns}")

        if len(numeric_columns) >= 2:
            # 散点图数据
            scatter_data = pd.DataFrame({
                'x': data.iloc[:, 0],
                'y': data.iloc[:, 1],
                'color': cluster_labels
            })
            visualizations['scatter'] = scatter_data

        if len(numeric_columns) >= 3:
            # 3D散点图数据
            scatter_3d_data = pd.DataFrame({
                'x': data.iloc[:, 0],
                'y': data.iloc[:, 1],
                'z': data.iloc[:, 2],
                'color': cluster_labels
            })
            visualizations['scatter_3d'] = scatter_3d_data

        # 聚类分布饼图数据
        cluster_counts = cluster_labels.value_counts().sort_index()
        visualizations['pie'] = cluster_counts

        # 聚类中心热力图数据
        if len(numeric_columns) >= 2:
            data_with_clusters = data.copy()
            data_with_clusters['Cluster'] = cluster_labels
            cluster_centers = data_with_clusters.groupby('Cluster')[numeric_columns].mean()
            visualizations['heatmap'] = cluster_centers

        return visualizations

    def _create_classification_visualizations(self, model_comparison: pd.DataFrame, metrics: Dict[str, float],
                                            best_model: Any, data: pd.DataFrame, target_variable: str) -> Dict[str, Any]:
        """创建分类任务的可视化数据"""
        visualizations = {}

        try:
            # 1. 模型性能比较图
            if not model_comparison.empty:
                visualizations['model_comparison'] = model_comparison

            # 2. 性能指标雷达图数据
            if metrics:
                # 标准化指标到0-1范围用于可视化
                normalized_metrics = {}
                for metric, value in metrics.items():
                    if metric == 'Accuracy':
                        normalized_metrics[metric] = value  # 已经是0-1范围
                    elif metric == 'AUC':
                        normalized_metrics[metric] = value  # 已经是0-1范围
                    elif metric == 'F1 Score':
                        normalized_metrics[metric] = value  # 已经是0-1范围
                    elif metric == 'Precision':
                        normalized_metrics[metric] = value  # 已经是0-1范围
                    elif metric == 'Recall':
                        normalized_metrics[metric] = value  # 已经是0-1范围

                if normalized_metrics:
                    metrics_df = pd.DataFrame(list(normalized_metrics.items()),
                                            columns=['Metric', 'Value'])
                    visualizations['metrics_radar'] = metrics_df

            # 3. 生成真实混淆矩阵
            confusion_data = self._generate_confusion_matrix(best_model, data, target_variable)
            if confusion_data:
                visualizations['confusion_matrix'] = confusion_data

            # 4. 提取真实特征重要性
            feature_data = self._extract_feature_importance(best_model, data, target_variable)
            if feature_data:
                visualizations['feature_importance'] = feature_data

            # 5. 分类性能指标对比
            if metrics:
                # 创建更全面的分类指标
                classification_metrics = {}
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        classification_metrics[metric] = value

                if classification_metrics:
                    metrics_comparison_df = pd.DataFrame(list(classification_metrics.items()),
                                                       columns=['Metric', 'Value'])
                    visualizations['metrics_comparison'] = metrics_comparison_df

        except Exception as e:
            print(f"分类可视化创建过程中出错: {e}")
            import traceback
            traceback.print_exc()

        return visualizations

    def _create_regression_visualizations(self, model_comparison: pd.DataFrame, metrics: Dict[str, float],
                                       best_model: Any, data: pd.DataFrame, target_variable: str) -> Dict[str, Any]:
        """创建回归任务的可视化数据"""
        visualizations = {}

        try:
            # 1. 模型性能比较图
            if not model_comparison.empty:
                visualizations['model_comparison'] = model_comparison

            # 2. 回归指标对比图数据
            if metrics:
                # 创建指标对比数据
                metrics_df = pd.DataFrame(list(metrics.items()),
                                        columns=['Metric', 'Value'])
                visualizations['metrics_comparison'] = metrics_df

            # 3. 生成真实残差分析
            residual_data = self._generate_residual_analysis(best_model, data, target_variable)
            if residual_data:
                visualizations['residuals'] = residual_data

            # 4. 生成真实预测vs实际值散点图
            prediction_data = self._generate_prediction_scatter(best_model, data, target_variable)
            if prediction_data:
                visualizations['prediction_scatter'] = prediction_data

            # 5. 提取真实特征重要性
            feature_data = self._extract_feature_importance(best_model, data, target_variable)
            if feature_data:
                visualizations['feature_importance'] = feature_data

            # 6. 生成真实残差直方图
            if residual_data and 'residuals' in residual_data:
                residual_hist_data = {
                    'residuals': residual_data['residuals'],
                    'bins': 30
                }
                visualizations['residual_histogram'] = residual_hist_data

            # 7. 生成真实Q-Q图数据
            qq_data = self._generate_qq_plot(residual_data['residuals'] if residual_data else None)
            if qq_data:
                visualizations['qq_plot'] = qq_data

        except Exception as e:
            print(f"回归可视化创建过程中出错: {e}")
            import traceback
            traceback.print_exc()

        return visualizations

    def _generate_confusion_matrix(self, model: Any, data: pd.DataFrame, target_variable: str) -> Dict[str, Any]:
        """生成真实混淆矩阵"""
        try:
            from pycaret.classification import predict_model

            # 获取预测结果
            predictions = predict_model(model, data=data)

            # 提取真实标签和预测标签
            y_true = data[target_variable].values
            y_pred = predictions['prediction_label'].values

            # 获取类别标签
            labels = sorted(list(set(y_true)))

            # 计算混淆矩阵
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            return {
                'matrix': cm.tolist(),
                'labels': labels
            }

        except Exception as e:
            print(f"生成混淆矩阵失败: {e}")
            return None

    def _generate_residual_analysis(self, model: Any, data: pd.DataFrame, target_variable: str) -> Dict[str, Any]:
        """生成真实残差分析数据"""
        try:
            from pycaret.regression import predict_model

            # 获取预测结果
            predictions = predict_model(model, data=data)

            # 提取真实值和预测值
            y_true = data[target_variable].values
            y_pred = predictions['prediction_label'].values

            # 计算残差
            residuals = y_true - y_pred

            return {
                'residuals': residuals.tolist(),
                'fitted': y_pred.tolist()
            }

        except Exception as e:
            print(f"生成残差分析失败: {e}")
            return None

    def _generate_prediction_scatter(self, model: Any, data: pd.DataFrame, target_variable: str) -> pd.DataFrame:
        """生成真实预测vs实际值散点图数据"""
        try:
            from pycaret.regression import predict_model

            # 获取预测结果
            predictions = predict_model(model, data=data)

            # 提取真实值和预测值
            y_true = data[target_variable].values
            y_pred = predictions['prediction_label'].values

            return pd.DataFrame({
                'actual': y_true,
                'predicted': y_pred
            })

        except Exception as e:
            print(f"生成预测散点图数据失败: {e}")
            return None

    def _generate_qq_plot(self, residuals: List[float]) -> Dict[str, Any]:
        """生成Q-Q图数据"""
        try:
            if residuals is None or len(residuals) == 0:
                return None

            from scipy import stats
            import numpy as np

            # 计算理论分位数和样本分位数
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
            sorted_residuals = np.sort(residuals)

            return {
                'theoretical': theoretical_quantiles.tolist(),
                'sample': sorted_residuals.tolist()
            }

        except Exception as e:
            print(f"生成Q-Q图数据失败: {e}")
            return None

    def _extract_feature_importance(self, model: Any, data: pd.DataFrame, target_variable: str) -> Dict[str, Any]:
        """提取真实特征重要性"""
        try:
            import numpy as np

            # 获取特征名（排除目标变量）
            feature_names = [col for col in data.columns if col != target_variable]

            # 尝试不同方法获取特征重要性
            importance_scores = None

            # 方法1: 如果模型有feature_importances_属性
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_

            # 方法2: 如果模型是Linear模型，使用系数的绝对值
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if coef.ndim == 1:
                    importance_scores = np.abs(coef)
                else:
                    importance_scores = np.mean(np.abs(coef), axis=0)

            # 方法3: 如果模型是pipeline，尝试获取内部模型
            elif hasattr(model, 'steps'):
                for step_name, step_model in model.steps:
                    if hasattr(step_model, 'feature_importances_'):
                        importance_scores = step_model.feature_importances_
                        break
                    elif hasattr(step_model, 'coef_'):
                        coef = step_model.coef_
                        if coef.ndim == 1:
                            importance_scores = np.abs(coef)
                        else:
                            importance_scores = np.mean(np.abs(coef), axis=0)
                        break

            # 如果成功获取特征重要性
            if importance_scores is not None and len(importance_scores) > 0:
                # 确保特征数量匹配
                if len(importance_scores) == len(feature_names):
                    # 标准化特征重要性
                    importance_scores = np.array(importance_scores)
                    if importance_scores.sum() > 0:
                        importance_scores = importance_scores / importance_scores.sum()

                    # 按重要性排序
                    indices = np.argsort(importance_scores)[::-1]
                    sorted_features = [feature_names[i] for i in indices]
                    sorted_importance = importance_scores[indices]

                    return {
                        'features': sorted_features,
                        'importance': sorted_importance.tolist(),
                        'title': '特征重要性'
                    }

            # 如果无法获取特征重要性，返回None
            print("无法获取模型特征重要性")
            return None

        except Exception as e:
            print(f"提取特征重要性失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict(self, model: Any, data: pd.DataFrame) -> Any:
        """
        使用训练好的模型进行预测

        Args:
            model: 训练好的模型
            data: 待预测数据

        Returns:
            预测结果
        """
        try:
            if not PYCARET_AVAILABLE:
                raise Exception("PyCaret未安装，无法进行预测")

            # 根据模型类型进行预测
            if hasattr(model, 'predict'):
                # 聚类模型
                if hasattr(model, 'cluster_centers'):
                    from pycaret.clustering import assign_model
                    return assign_model(model, data=data)
                # 分类或回归模型
                else:
                    from pycaret.classification import predict_model as clf_predict_model
                    from pycaret.regression import predict_model as reg_predict_model

                    try:
                        # 尝试分类预测
                        return clf_predict_model(model, data=data)
                    except:
                        # 如果失败，尝试回归预测
                        return reg_predict_model(model, data=data)
            else:
                raise Exception("无效的模型类型")

        except Exception as e:
            raise Exception(f"预测失败: {str(e)}")

    def get_model_summary(self, model: Any) -> Dict[str, Any]:
        """获取模型摘要信息"""
        if model is None:
            return {}

        summary = {
            'model_type': type(model).__name__,
            'has_predict': hasattr(model, 'predict'),
            'has_fit': hasattr(model, 'fit'),
            'has_classes': hasattr(model, 'classes_') if hasattr(model, 'classes_') else False
        }

        # 如果是聚类模型，添加聚类信息
        if hasattr(model, 'cluster_centers'):
            summary.update({
                'n_clusters': len(model.cluster_centers),
                'cluster_centers_shape': model.cluster_centers.shape
            })

        return summary