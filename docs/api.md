# API 文档

## 概述

MLquick 提供了简单易用的API接口，支持分类、回归和预测任务。

## 主要模块

### models.classification

#### `classification_task(data, target_variable, train_size=0.7)`

执行分类任务建模。

**参数:**
- `data` (pd.DataFrame): 输入数据集
- `target_variable` (str): 目标变量列名
- `train_size` (float): 训练集比例，默认0.7

**返回值:**
- `best_model`: 训练好的最佳模型
- `model_comparison` (pd.DataFrame): 模型性能对比结果

**示例:**
```python
from src.models.classification import classification_task
import pandas as pd

# 加载数据
data = pd.read_csv('data.csv')

# 执行分类任务
model, comparison = classification_task(data, 'target', train_size=0.8)
```

### models.regression

#### `regression_task(data, target_variable, train_size=0.7)`

执行回归任务建模。

**参数:**
- `data` (pd.DataFrame): 输入数据集
- `target_variable` (str): 目标变量列名
- `train_size` (float): 训练集比例，默认0.7

**返回值:**
- `best_model`: 训练好的最佳模型
- `model_comparison` (pd.DataFrame): 模型性能对比结果

### models.prediction

#### `prediction(model_path, prediction_file)`

使用已训练模型进行预测。

**参数:**
- `model_path` (str): 模型文件路径（不含.pkl扩展名）
- `prediction_file`: 预测数据文件

**返回值:**
- `predictions` (pd.DataFrame): 预测结果

## 错误处理

所有函数都包含适当的错误处理机制：

- 数据格式验证
- 模型文件存在性检查
- 文件编码处理
- 异常情况提示

## 性能优化建议

1. **数据预处理**: 建议在使用API前进行基本的数据清洗
2. **内存管理**: 大数据集建议分批处理
3. **模型缓存**: 训练好的模型会自动保存，可重复使用