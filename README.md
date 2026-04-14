# MLquick - 机器学习零代码应用平台

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.49.1-red.svg)](https://streamlit.io)
[![PyCaret](https://img.shields.io/badge/pycaret-3.3.2-orange.svg)](https://pycaret.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

一个基于Streamlit和PyCaret的快速机器学习建模平台，支持分类和回归任务的自动化模型训练、比较和预测。

## ✨ 功能特性

- 🚀 **快速建模**: 基于PyCaret实现自动化机器学习流程
- 📊 **多模型对比**: 自动训练和比较多种机器学习算法
- 🎯 **分类任务**: 支持各种分类问题的建模和预测
- 📈 **回归任务**: 支持回归问题的建模和预测
- 🎲 **聚类分析**: K-means无监督聚类，自动发现数据群组
- 💾 **模型保存**: 自动保存最佳模型供后续使用
- 📁 **文件支持**: 支持CSV和Excel格式数据文件
- 🎨 **交互界面**: 基于Streamlit的直观Web界面
- 🔧 **自动预处理**: 数据标准化和特征工程
- 📋 **详细报告**: 模型性能对比和评估指标
- 📊 **可视化分析**: 聚类结果多维度可视化展示

## ✨ 应用截图

![alt text](/data/png/image-1.png)
![alt text](/data/png/image-2.png)
![alt text](/data/png/image-3.png)
![alt text](/data/png/image.png)
![alt text](/data/png/image-4.png)
![alt text](/data/png/image-5.png)
![alt text](/data/png/image-6.png)

## 🛠️ 技术栈

- **Streamlit 1.49.1** - Web应用框架
- **PyCaret 3.3.2** - 低代码机器学习库
- **Pandas** - 数据处理和分析
- **Matplotlib** - 数据可视化
- **Scikit-learn** - 机器学习算法库
- **Python 3.7+** - 编程语言

## 📋 目录

- [快速开始](#-快速开始)
- [安装说明](#-安装说明)
- [使用指南](#-使用指南)
- [项目结构](#-项目结构)
- [API文档](#-api文档)
- [示例数据](#-示例数据)
- [常见问题](#-常见问题)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)

## 🚀 快速开始

### 前置要求

- Python 3.7+
- pip 包管理器

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/MLquick.git
cd MLquick
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 运行应用

```bash
streamlit run src/MLquick.py
```

### 5. 访问应用

打开浏览器访问 `http://localhost:8501`

## 📦 安装说明

### 创建 requirements.txt

```bash
# 核心依赖
streamlit==1.49.1
pycaret==3.3.2
pandas>=1.5.0
matplotlib>=3.5.0
numpy>=1.21.0
openpyxl>=3.0.0

# 机器学习算法
scikit-learn>=1.1.0
xgboost>=1.6.0
lightgbm>=3.3.0

# 可选依赖
seaborn>=0.11.0  # 可视化
plotly>=5.0.0    # 交互式图表
```

### Docker 部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "src/MLquick.py"]
```

## 📖 使用指南

### 1. 数据准备

- 支持格式: CSV, Excel (.xlsx, .xls)
- 数据要求: 无缺失值，特征列清晰
- 文件大小: 建议 < 100MB

### 2. 模型训练流程

```
上传数据 → 选择任务类型 → 设置目标变量 → 配置训练参数 → 训练模型 → 查看结果
```

### 3. 详细步骤

#### 步骤1: 数据上传

- 点击"上传数据集"按钮
- 选择CSV或Excel文件
- 系统自动显示数据预览

#### 步骤2: 任务配置

- **任务类型**: 选择"分类"或"回归"
- **目标变量**: 从下拉列表选择预测目标
- **训练集比例**: 设置0.6-0.8之间，推荐0.7

#### 步骤3: 模型训练

- 点击"训练模型"按钮
- 系统自动执行以下步骤：
  - 数据预处理
  - 特征工程
  - 多算法训练
  - 性能评估
  - 最佳模型选择

#### 步骤4: 结果分析

- 查看模型对比表格
- 最佳模型信息
- 性能指标详情

#### 步骤5: 模型预测

- 勾选"载入最佳模型进行预测"
- 上传预测数据文件
- 获取预测结果

## 📁 项目结构

```
MLquick/
├── src/                        # 源代码目录
│   ├── __init__.py            # 包初始化文件
│   ├── MLquick.py             # 主应用程序
│   ├── models/                # 模型相关模块
│   │   └── __init__.py        # 模型模块初始化
│   └── utils/                 # 工具模块
│       └── __init__.py        # 工具模块初始化
├── docs/                      # 文档目录
│   ├── api.md                 # API文档
│   └── examples.md            # 示例数据说明
├── data/                      # 数据目录
│   ├── samples/               # 示例数据
│   │   ├── classification_sample.csv  # 分类任务样例数据
│   │   ├── regression_sample.csv      # 回归任务样例数据
│   │   └── clustering_sample.csv     # 聚类任务样例数据
│   └── png/                   # 图片资源
├── models/                    # 训练好的模型文件
├── notebooks/                 # Jupyter笔记本（预留）
├── tests/                     # 测试文件（预留）
├── .venv/                     # Python虚拟环境
├── .git/                      # Git版本控制
├── .claude/                   # Claude配置
├── requirements.txt           # 依赖包列表
├── .gitignore                 # Git忽略文件配置
├── LICENSE                    # MIT许可证
├── README.md                  # 项目说明文档
└── logs.log                   # 日志文件
```

## 🔧 API文档

### 主要函数

#### `classification_task(data, target_variable, train_size)`

**功能**: 执行分类任务建模
**参数**:

- `data` (pd.DataFrame): 输入数据
- `target_variable` (str): 目标变量名
- `train_size` (float): 训练集比例

**返回值**:

- `best_model`: 最佳训练模型
- `model_comparison`: 模型对比结果

#### `regression_task(data, target_variable, train_size)`

**功能**: 执行回归任务建模
**参数**: 同分类任务

#### `prediction(model_path, prediction_file)`

**功能**: 使用已训练模型进行预测
**参数**:

- `model_path` (str): 模型文件路径
- `prediction_file`: 预测数据文件

## 📊 示例数据

项目提供了两个样例数据文件，位于 `data/samples/` 目录：

### 1. 分类任务数据 (`classification_sample.csv`)

**场景**: 客户购买行为预测

- **样本数量**: 500条
- **特征数量**: 6个特征
- **目标变量**: `purchase_category` (购买类别)
  - Electronics (电子产品)
  - Books (图书)
  - Clothing (服装)

**特征说明**:

- `age`: 年龄
- `income`: 收入
- `education_level`: 教育水平
- `years_experience`: 工作经验年限
- `has_credit_card`: 是否有信用卡
- `marital_status`: 婚姻状况

**使用方法**:

1. 上传 `data/samples/classification_sample.csv` 文件
2. 选择任务类型为"分类"
3. 选择目标变量为 `purchase_category`
4. 设置训练集比例（推荐0.7）
5. 点击"训练模型"

### 2. 回归任务数据 (`regression_sample.csv`)

**场景**: 房价预测

- **样本数量**: 400条
- **特征数量**: 11个特征
- **目标变量**: `price_in_thousands` (房价，单位：千美元)

**特征说明**:

- `house_age`: 房屋年龄
- `square_feet`: 面积（平方英尺）
- `num_bedrooms`: 卧室数量
- `num_bathrooms`: 浴室数量
- `garage_size`: 车库大小
- `neighborhood_quality`: 社区质量评分 (1-10)
- `school_rating`: 学校评分 (1-10)
- `distance_to_downtown`: 距离市中心距离（英里）
- `has_pool`: 是否有游泳池
- `has_garden`: 是否有花园
- `year_built`: 建造年份

**使用方法**:

1. 上传 `data/samples/regression_sample.csv` 文件
2. 选择任务类型为"回归"
3. 选择目标变量为 `price_in_thousands`
4. 设置训练集比例（推荐0.7）
5. 点击"训练模型"

### 3. 聚类任务数据 (`clustering_sample.csv`)

**场景**: 客户细分分析

- **样本数量**: 200条
- **特征数量**: 10个数值特征
- **任务类型**: 无监督聚类分析

**特征说明**:

- `age`: 年龄
- `income`: 年收入
- `spending_score`: 消费评分 (0-100)
- `savings_score`: 储蓄评分 (0-100)
- `years_as_customer`: 客户年限
- `avg_monthly_purchases`: 月均购买次数
- `online_frequency`: 线上购物频率
- `in_store_frequency`: 线下购物频率
- `discount_usage`: 优惠使用率 (%)
- `loyalty_points`: 会员积分

**使用方法**:

1. 上传 `data/samples/clustering_sample.csv` 文件
2. 选择任务类型为"聚类"
3. 设置聚类数量（推荐3-5个）
4. 选择用于聚类的特征（可多选，默认使用前5个数值特征）
5. 点击"训练模型"

**预期聚类结果**:

- 聚类0: 高收入高消费的优质客户
- 聚类1: 中等收入的稳定客户
- 聚类2: 年轻的低消费客户

### 快速开始示例

```python
# 加载分类样例数据
import pandas as pd
data = pd.read_csv('data/samples/classification_sample.csv')

# 或者加载回归样例数据
data = pd.read_csv('data/samples/regression_sample.csv')

# 或者加载聚类样例数据
data = pd.read_csv('data/samples/clustering_sample.csv')
```

## ❓ 常见问题

### Q1: 支持哪些数据格式？

A: 目前支持CSV和Excel格式，确保文件编码为UTF-8。

### Q2: 训练时间很长怎么办？

A: 可以通过减少训练集比例或选择特定算法来缩短训练时间。

### Q3: 如何提高模型性能？

A:

- 确保数据质量，处理缺失值和异常值
- 尝试不同的训练集比例
- 进行特征工程和特征选择

### Q4: 模型文件保存在哪里？

A: 模型文件自动保存在项目根目录，文件名为`best_classification_model.pkl`或`best_regression_model.pkl`。

### Q5: 可以保存预测结果吗？

A: 可以在界面上选择下载预测结果为CSV文件。

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 开发流程

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 代码规范

- 遵循PEP 8编码规范
- 添加适当的注释和文档字符串
- 编写单元测试
- 确保代码通过所有测试

### 问题报告

使用GitHub Issues报告问题，请包含：

- 详细的错误描述
- 重现步骤
- 环境信息
- 相关日志

## 📄 许可证

本项目采用 Apache License 2.0 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Streamlit](https://streamlit.io/) - 强大的Web应用框架
- [PyCaret](https://pycaret.org/) - 简化的机器学习库
- [Scikit-learn](https://scikit-learn.org/) - 机器学习算法库

## 📞 联系方式

- 项目主页: [GitHub Repository](https://github.com/cn-vhql/MLquick)
- 问题反馈: [GitHub Issues](https://github.com/cn-vhql/MLquick/issues)
- 邮箱: yl_zhangqiang@foxmail.com

---

⭐ 如果这个项目对你有帮助，请给我们一个星标！
