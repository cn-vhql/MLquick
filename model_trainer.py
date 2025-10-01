#!/usr/bin/env python3
"""
模型训练模块 - 负责模型训练和评估
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Tuple, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from data_processor import create_features_targets, calculate_technical_indicators
from model_config import ModelConfig, ModelEvaluator, ModelOptimizer, get_default_config


def regression_prediction(X: pd.DataFrame, y: pd.Series, train_size: float = 0.7) -> Tuple[Dict[str, Dict], Any, pd.DataFrame, pd.Series]:
    """
    回归预测训练

    Args:
        X: 特征矩阵
        y: 目标变量
        train_size: 训练集比例

    Returns:
        模型结果字典, 最佳模型, 测试集特征, 测试集目标
    """
    config = get_default_config()
    models = config.get_regression_models()
    evaluator = ModelEvaluator()

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=42, shuffle=False)

    results = {}

    with st.spinner("正在训练回归模型..."):
        progress_bar = st.progress(0)
        total_models = len(models)

        for i, (name, model) in enumerate(models.items()):
            try:
                # 训练模型
                model.fit(X_train, y_train)

                # 评估模型
                metrics = evaluator.evaluate_regression(model, X_test, y_test)
                results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': model.predict(X_test)
                }

            except Exception as e:
                st.error(f"训练{name}模型时出错: {str(e)}")
                results[name] = None

            progress_bar.progress((i + 1) / total_models)

    # 获取最佳模型
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_model_name, best_result = evaluator.get_best_model(valid_results, 'regression')
        best_model = best_result['model']
        st.success(f"最佳回归模型: {best_model_name} (R² = {best_result['metrics']['r2']:.4f})")
    else:
        best_model = None
        st.error("所有回归模型训练失败")

    return results, best_model, X_test, y_test


def classification_prediction(X: pd.DataFrame, y: pd.Series, train_size: float = 0.7) -> Tuple[Dict[str, Dict], Any, pd.DataFrame, pd.Series]:
    """
    分类预测训练

    Args:
        X: 特征矩阵
        y: 目标变量
        train_size: 训练集比例

    Returns:
        模型结果字典, 最佳模型, 测试集特征, 测试集目标
    """
    config = get_default_config()
    models = config.get_classification_models()
    evaluator = ModelEvaluator()

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=42, shuffle=False)

    results = {}

    with st.spinner("正在训练分类模型..."):
        progress_bar = st.progress(0)
        total_models = len(models)

        for i, (name, model) in enumerate(models.items()):
            try:
                # 训练模型
                model.fit(X_train, y_train)

                # 评估模型
                metrics = evaluator.evaluate_classification(model, X_test, y_test)
                results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': model.predict(X_test)
                }

            except Exception as e:
                st.error(f"训练{name}模型时出错: {str(e)}")
                results[name] = None

            progress_bar.progress((i + 1) / total_models)

    # 获取最佳模型
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_model_name, best_result = evaluator.get_best_model(valid_results, 'classification')
        best_model = best_result['model']
        st.success(f"最佳分类模型: {best_model_name} (准确率 = {best_result['metrics']['accuracy']:.4f})")
    else:
        best_model = None
        st.error("所有分类模型训练失败")

    return results, best_model, X_test, y_test


def train_complete_workflow(df: pd.DataFrame, historical_days: int = 7, prediction_days: int = 3,
                           task_type: str = 'regression', train_size: float = 0.7) -> Dict[str, Any]:
    """
    完整的模型训练工作流

    Args:
        df: 原始数据DataFrame
        historical_days: 历史数据天数
        prediction_days: 预测天数
        task_type: 任务类型
        train_size: 训练集比例

    Returns:
        包含所有训练结果的字典
    """
    # 计算技术指标
    df_processed = calculate_technical_indicators(df)

    # 创建特征和目标变量
    X, y = create_features_targets(
        df_processed,
        historical_days=historical_days,
        prediction_days=prediction_days,
        task_type=task_type
    )

    if len(X) == 0:
        st.error("特征工程失败，请检查数据质量和参数设置")
        return {}

    # 训练模型
    if task_type == 'regression':
        results, best_model, X_test, y_test = regression_prediction(X, y, train_size)
    else:
        results, best_model, X_test, y_test = classification_prediction(X, y, train_size)

    return {
        'df_processed': df_processed,
        'X': X,
        'y': y,
        'X_test': X_test,
        'y_test': y_test,
        'results': results,
        'best_model': best_model,
        'task_type': task_type,
        'historical_days': historical_days,
        'prediction_days': prediction_days
    }


def plot_model_comparison(results: Dict[str, Dict], task_type: str) -> plt.Figure:
    """
    绘制模型性能对比图

    Args:
        results: 模型结果字典
        task_type: 任务类型

    Returns:
        matplotlib图形对象
    """
    valid_results = {k: v for k, v in results.items() if v is not None}

    if not valid_results:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    if task_type == 'regression':
        # 回归模型对比
        models = list(valid_results.keys())
        r2_scores = [valid_results[model]['metrics']['r2'] for model in models]

        bars = ax.bar(models, r2_scores, alpha=0.7)
        ax.set_ylabel('R² Score')
        ax.set_title('Regression Model Performance Comparison (R² Score)')
        ax.set_ylim(0, 1)

        # 添加数值标签
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')

    else:
        # 分类模型对比
        models = list(valid_results.keys())
        accuracies = [valid_results[model]['metrics']['accuracy'] for model in models]

        bars = ax.bar(models, accuracies, alpha=0.7)
        ax.set_ylabel('Accuracy')
        ax.set_title('Classification Model Performance Comparison (Accuracy)')
        ax.set_ylim(0, 1)

        # 添加数值标签
        for bar, accuracy in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{accuracy:.3f}', ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> plt.Figure:
    """
    绘制预测散点图（回归模型）

    Args:
        y_true: 真实值
        y_pred: 预测值
        model_name: 模型名称

    Returns:
        matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.6, s=30)

    # 添加理想预测线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')

    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'{model_name} - Prediction Comparison')

    # 计算R²
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: List[str], model_name: str) -> plt.Figure:
    """
    绘制混淆矩阵

    Args:
        confusion_matrix: 混淆矩阵
        class_names: 类别名称
        model_name: 模型名称

    Returns:
        matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)

    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title(f'{model_name} - Confusion Matrix')

    plt.tight_layout()
    return fig


def evaluate_model_performance(results: Dict[str, Dict], task_type: str) -> pd.DataFrame:
    """
    生成模型性能评估表

    Args:
        results: 模型结果字典
        task_type: 任务类型

    Returns:
        包含性能指标的DataFrame
    """
    valid_results = {k: v for k, v in results.items() if v is not None}

    if not valid_results:
        return pd.DataFrame()

    performance_data = []

    if task_type == 'regression':
        for model_name, result in valid_results.items():
            metrics = result['metrics']
            performance_data.append({
                '模型': model_name,
                'R²分数': f"{metrics['r2']:.4f}",
                '均方误差': f"{metrics['mse']:.4f}",
                '均方根误差': f"{metrics['rmse']:.4f}",
                '平均绝对误差': f"{metrics['mae']:.4f}"
            })
    else:
        for model_name, result in valid_results.items():
            metrics = result['metrics']
            performance_data.append({
                '模型': model_name,
                '准确率': f"{metrics['accuracy']:.4f}",
                '精确率(宏平均)': f"{metrics['classification_report']['macro avg']['precision']:.4f}",
                '召回率(宏平均)': f"{metrics['classification_report']['macro avg']['recall']:.4f}",
                'F1分数(宏平均)': f"{metrics['classification_report']['macro avg']['f1-score']:.4f}"
            })

    return pd.DataFrame(performance_data)


def get_feature_importance_from_model(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    从模型中提取特征重要性

    Args:
        model: 训练好的模型
        feature_names: 特征名称列表

    Returns:
        包含特征重要性的DataFrame
    """
    try:
        # 处理Pipeline模型
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            inner_model = model.named_steps['model']
        else:
            inner_model = model

        # 获取特征重要性
        if hasattr(inner_model, 'feature_importances_'):
            importance = inner_model.feature_importances_
        elif hasattr(inner_model, 'coef_'):
            importance = np.abs(inner_model.coef_)
            if len(importance.shape) > 1:
                importance = np.mean(importance, axis=0)
        else:
            return pd.DataFrame({'Feature': feature_names, 'Importance': [0] * len(feature_names)})

        # 创建DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })

        # 按重要性排序
        importance_df = importance_df.sort_values('Importance', ascending=False)

        return importance_df

    except Exception as e:
        st.error(f"提取特征重要性时出错: {str(e)}")
        return pd.DataFrame({'Feature': feature_names, 'Importance': [0] * len(feature_names)})