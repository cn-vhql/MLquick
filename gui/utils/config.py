#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件 - 应用程序配置常量
"""

# 应用信息
APP_NAME = "MLquick"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "机器学习零代码桌面应用"
APP_AUTHOR = "MLquick Team"

# 窗口设置
DEFAULT_WINDOW_WIDTH = 1200
DEFAULT_WINDOW_HEIGHT = 800
MIN_WINDOW_WIDTH = 800
MIN_WINDOW_HEIGHT = 600

# 文件设置
SUPPORTED_DATA_FORMATS = [
    ("CSV files", "*.csv"),
    ("Excel files", "*.xlsx *.xls"),
    ("All files", "*.*")
]

SUPPORTED_MODEL_FORMATS = [
    ("Model files", "*.pkl"),
    ("Zip files", "*.zip"),
    ("All files", "*.*")
]

# 机器学习设置
DEFAULT_TRAIN_SIZE = 0.7
MIN_TRAIN_SIZE = 0.1
MAX_TRAIN_SIZE = 0.9
DEFAULT_N_CLUSTERS = 3
MIN_N_CLUSTERS = 2
MAX_N_CLUSTERS = 20

# 文本处理设置
DEFAULT_MAX_FEATURES = 1000
DEFAULT_MIN_WORD_LENGTH = 2
DEFAULT_REMOVE_STOPWORDS = True

# 可视化设置
DEFAULT_FIGURE_SIZE = (10, 6)
DEFAULT_DPI = 100
DEFAULT_COLOR_PALETTE = "Set3"

# 性能设置
MAX_DISPLAY_ROWS = 1000
MAX_PREVIEW_ROWS = 100
MAX_FEATURES_FOR_DISPLAY = 50
TASK_CHECK_INTERVAL = 100  # 毫秒

# 文件路径设置
DEFAULT_MODELS_DIR = "models"
DEFAULT_EXPORT_DIR = "exports"
DEFAULT_TEMP_DIR = "temp"

# 错误消息
ERROR_MESSAGES = {
    "no_data": "请先上传数据文件",
    "no_model": "请先训练或加载模型",
    "invalid_target": "请选择有效的目标变量",
    "invalid_clusters": "聚类数量必须大于1",
    "file_not_found": "文件不存在",
    "invalid_format": "不支持的文件格式",
    "loading_failed": "数据加载失败",
    "training_failed": "模型训练失败",
    "prediction_failed": "预测失败",
    "model_save_failed": "模型保存失败",
    "model_load_failed": "模型加载失败"
}

# 成功消息
SUCCESS_MESSAGES = {
    "data_loaded": "数据加载成功",
    "model_trained": "模型训练完成",
    "model_saved": "模型保存成功",
    "model_loaded": "模型加载成功",
    "prediction_complete": "预测完成",
    "export_complete": "导出完成"
}

# 支持的任务类型
TASK_TYPES = {
    "分类": "classification",
    "回归": "regression",
    "聚类": "clustering"
}

# 任务类型配置
TASK_CONFIG = {
    "classification": {
        "requires_target": True,
        "supported_metrics": ["Accuracy", "AUC", "F1 Score", "Precision", "Recall"],
        "default_algorithms": ["lr", "rf", "svm", "knn", "nb"]
    },
    "regression": {
        "requires_target": True,
        "supported_metrics": ["R2", "RMSE", "MAE", "MAPE"],
        "default_algorithms": ["lr", "rf", "svm", "knn", "lasso"]
    },
    "clustering": {
        "requires_target": False,
        "supported_metrics": ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"],
        "default_algorithms": ["kmeans", "hierarchical", "dbscan"]
    }
}

# PyCaret设置
PYCARET_CONFIG = {
    "session_id": 123,
    "normalize": True,
    "silent": True,
    "verbose": False
}

# 字体设置
FONT_CONFIG = {
    "default_family": "Arial",
    "default_size": 10,
    "title_size": 16,
    "subtitle_size": 12,
    "button_size": 10,
    "label_size": 9
}

# 颜色主题
COLOR_THEME = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "success": "#F18F01",
    "warning": "#C73E1D",
    "error": "#C73E1D",
    "background": "#FFFFFF",
    "surface": "#F5F5F5",
    "text": "#333333",
    "text_secondary": "#666666"
}

# 快捷键
SHORTCUTS = {
    "new": "Ctrl+N",
    "open": "Ctrl+O",
    "save": "Ctrl+S",
    "train": "F5",
    "predict": "F6",
    "exit": "Ctrl+Q"
}

# 日志设置
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_name": "mlquick_gui.log",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# 调试设置
DEBUG_MODE = False
SHOW_PERFORMANCE_INFO = False
ENABLE_LOGGING = True

# 国际化设置
DEFAULT_LANGUAGE = "zh_CN"
SUPPORTED_LANGUAGES = ["zh_CN", "en_US"]

# 数据验证规则
VALIDATION_RULES = {
    "max_file_size": 100 * 1024 * 1024,  # 100MB
    "max_columns": 1000,
    "max_rows": 1000000,
    "allowed_extensions": [".csv", ".xlsx", ".xls"],
    "min_samples_per_class": 2,
    "max_missing_ratio": 0.5
}