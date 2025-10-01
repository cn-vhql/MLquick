#!/usr/bin/env python3
"""
模型配置模块 - 负责定义和配置机器学习模型
"""
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any, Tuple
import numpy as np


class ModelConfig:
    """模型配置类"""

    def __init__(self):
        self.regression_models = self._init_regression_models()
        self.classification_models = self._init_classification_models()

    def _init_regression_models(self) -> Dict[str, Any]:
        """初始化回归模型"""
        return {
            'Linear Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ]),
            'Ridge Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(alpha=1.0))
            ]),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        }

    def _init_classification_models(self) -> Dict[str, Any]:
        """初始化分类模型"""
        return {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Hist Gradient Boosting': HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(
                    multi_class='multinomial',
                    max_iter=1000,
                    random_state=42
                ))
            ])
        }

    def get_regression_models(self) -> Dict[str, Any]:
        """获取回归模型"""
        return self.regression_models

    def get_classification_models(self) -> Dict[str, Any]:
        """获取分类模型"""
        return self.classification_models

    def get_model_params(self, model_name: str, task_type: str) -> Dict[str, Any]:
        """获取模型参数"""
        if task_type == 'regression':
            models = self.regression_models
        else:
            models = self.classification_models

        if model_name not in models:
            raise ValueError(f"模型 {model_name} 不在支持列表中")

        model = models[model_name]

        # 如果是Pipeline，获取内部模型的参数
        if hasattr(model, 'named_steps'):
            if 'model' in model.named_steps:
                return model.named_steps['model'].get_params()
        else:
            return model.get_params()

        return {}

    def update_model_params(self, model_name: str, task_type: str, params: Dict[str, Any]) -> None:
        """更新模型参数"""
        if task_type == 'regression':
            models = self.regression_models
        else:
            models = self.classification_models

        if model_name not in models:
            raise ValueError(f"模型 {model_name} 不在支持列表中")

        model = models[model_name]

        # 如果是Pipeline，更新内部模型的参数
        if hasattr(model, 'named_steps'):
            if 'model' in model.named_steps:
                model.named_steps['model'].set_params(**params)
        else:
            model.set_params(**params)


class ModelEvaluator:
    """模型评估类"""

    @staticmethod
    def evaluate_regression(model, X_test, y_test) -> Dict[str, float]:
        """评估回归模型"""
        y_pred = model.predict(X_test)

        return {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mae': np.mean(np.abs(y_test - y_pred))
        }

    @staticmethod
    def evaluate_classification(model, X_test, y_test) -> Dict[str, Any]:
        """评估分类模型"""
        y_pred = model.predict(X_test)

        # 获取类别名称
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        class_names = ['Down', 'Sideways', 'Up'] if len(unique_labels) == 3 else ['Class_0', 'Class_1']

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, target_names=class_names, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'class_names': class_names
        }

    @staticmethod
    def get_best_model(results: Dict[str, Dict], task_type: str) -> Tuple[str, Any]:
        """获取最佳模型"""
        if task_type == 'regression':
            # 回归任务选择R²最高的模型
            best_name = max(results.keys(), key=lambda x: results[x]['metrics']['r2'])
        else:
            # 分类任务选择准确率最高的模型
            best_name = max(results.keys(), key=lambda x: results[x]['metrics']['accuracy'])

        return best_name, results[best_name]


class ModelOptimizer:
    """模型优化类"""

    def __init__(self):
        self.param_grids = self._init_param_grids()

    def _init_param_grids(self) -> Dict[str, Dict]:
        """初始化参数网格"""
        return {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'Ridge Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Logistic Regression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'max_iter': [500, 1000, 2000]
            }
        }

    def get_param_grid(self, model_name: str) -> Dict[str, list]:
        """获取模型的参数网格"""
        return self.param_grids.get(model_name, {})

    def suggest_params(self, model_name: str, task_type: str) -> Dict[str, Any]:
        """为模型建议参数"""
        param_grid = self.get_param_grid(model_name)

        if not param_grid:
            return {}

        # 返回每个参数的中等值作为建议
        suggested_params = {}
        for param_name, param_values in param_grid.items():
            if len(param_values) > 0:
                middle_index = len(param_values) // 2
                suggested_params[param_name] = param_values[middle_index]

        return suggested_params


class ModelRegistry:
    """模型注册表"""

    def __init__(self):
        self.registered_models = {}
        self.model_metadata = {}

    def register_model(self, name: str, model: Any, metadata: Dict[str, Any] = None) -> None:
        """注册模型"""
        self.registered_models[name] = model
        self.model_metadata[name] = metadata or {}

    def get_model(self, name: str) -> Any:
        """获取注册的模型"""
        if name not in self.registered_models:
            raise ValueError(f"模型 {name} 未注册")
        return self.registered_models[name]

    def get_model_metadata(self, name: str) -> Dict[str, Any]:
        """获取模型元数据"""
        return self.model_metadata.get(name, {})

    def list_models(self) -> list:
        """列出所有注册的模型"""
        return list(self.registered_models.keys())

    def remove_model(self, name: str) -> None:
        """移除注册的模型"""
        if name in self.registered_models:
            del self.registered_models[name]
        if name in self.model_metadata:
            del self.model_metadata[name]


def get_default_config() -> ModelConfig:
    """获取默认模型配置"""
    return ModelConfig()


def get_model_info(task_type: str) -> Dict[str, Dict[str, str]]:
    """获取模型信息"""
    config = get_default_config()

    if task_type == 'regression':
        models = config.get_regression_models()
        return {
            'Linear Regression': {
                'description': '线性回归模型，适用于线性关系的数据',
                'advantages': '简单快速，可解释性强',
                'disadvantages': '无法处理非线性关系'
            },
            'Ridge Regression': {
                'description': '岭回归模型，带有L2正则化的线性回归',
                'advantages': '防止过拟合，处理多重共线性',
                'disadvantages': '需要调节正则化参数'
            },
            'Random Forest': {
                'description': '随机森林回归模型，集成多个决策树',
                'advantages': '处理非线性关系，抗过拟合能力强',
                'disadvantages': '模型复杂，可解释性较差'
            },
            'Gradient Boosting': {
                'description': '梯度提升回归模型，逐步优化残差',
                'advantages': '预测精度高，能处理复杂关系',
                'disadvantages': '训练时间较长，对异常值敏感'
            }
        }
    else:
        models = config.get_classification_models()
        return {
            'Random Forest': {
                'description': '随机森林分类模型，集成多个决策树',
                'advantages': '处理非线性关系，抗过拟合能力强',
                'disadvantages': '模型复杂，可解释性较差'
            },
            'Hist Gradient Boosting': {
                'description': '直方图梯度提升分类模型，优化了训练速度',
                'advantages': '训练速度快，内存效率高，预测精度高',
                'disadvantages': '参数调节较复杂'
            },
            'Logistic Regression': {
                'description': '逻辑回归分类模型，线性分类器',
                'advantages': '简单快速，可解释性强，输出概率',
                'disadvantages': '无法处理非线性关系'
            }
        }