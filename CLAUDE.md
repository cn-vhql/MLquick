# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

MLquick是一个基于Streamlit和PyCaret的零代码机器学习建模平台，支持分类、回归和聚类任务的自动化建模。

## 常用开发命令

### 启动应用
```bash
streamlit run src/MLquick.py
```

### 安装依赖
```bash
pip install -r requirements.txt
```

### 虚拟环境设置
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

## 项目架构

### 核心应用结构
- `src/MLquick.py` - 主应用文件，包含完整的Streamlit界面和机器学习逻辑
- 单文件架构：所有功能集中在主文件中，包括模型训练、预测、可视化等

### 主要功能模块
1. **数据处理**: CSV/Excel文件读取和预处理
2. **模型训练**: 基于PyCaret的自动化机器学习
   - 分类任务 (`classification_task`)
   - 回归任务 (`regression_task`)
   - 聚类任务 (`clustering_task`)
3. **模型管理**: 模型保存、加载、导入导出
4. **预测功能**: 使用训练好的模型进行预测
5. **可视化**: 聚类结果的多维度可视化展示

### 数据存储结构
- `models/` - 训练好的模型文件存储
- `data/samples/` - 示例数据集
  - `classification_sample.csv` - 分类任务示例
  - `regression_sample.csv` - 回归任务示例
  - `clustering_sample.csv` - 聚类任务示例

### 依赖关系
- **Streamlit 1.49.1**: Web应用框架
- **PyCaret 3.3.2**: 低代码机器学习库
- **Pandas**: 数据处理
- **Plotly/Matplotlib**: 数据可视化
- **Scikit-learn**: 机器学习算法

## 重要实现细节

### 模型管理机制
- 模型文件按时间戳命名，避免覆盖
- 支持模型的导入导出功能
- 自动生成模型信息文件

### 会话状态管理
使用`st.session_state`管理以下状态：
- `best_model` - 当前训练的模型
- `model_comparison` - 模型对比结果
- `current_model_name` - 当前模型名称
- `clustered_data` - 聚类结果数据
- `visualizations` - 可视化图表

### 聚类可视化功能
自动生成四种可视化图表：
- 2D散点图
- 3D散点图
- 聚类分布饼图
- 聚类中心热力图

## 开发注意事项

### PyCaret集成
- 分类任务使用`pycaret.classification`
- 回归任务使用`pycaret.regression`
- 聚类任务使用`pycaret.clustering`

### 文件路径处理
- 模型文件存储在`../models/`目录
- 相对路径处理需注意当前工作目录

### 数据预处理
- 自动检测数值型特征
- 聚类任务只使用数值型特征
- 自动处理文件编码(UTF-8-BOM)

### 错误处理
- 完整的异常捕获机制
- 用户友好的错误提示
- 数据验证和文件格式检查