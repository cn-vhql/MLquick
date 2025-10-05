# 示例数据文档

## 概述

MLquick项目提供了两个完整的样例数据集，帮助用户快速了解和测试平台功能。

## 数据文件位置

```
data/samples/
├── classification_sample.csv  # 分类任务数据
└── regression_sample.csv      # 回归任务数据
```

## 分类任务数据集

### 文件信息
- **文件名**: `classification_sample.csv`
- **文件大小**: ~45KB
- **数据行数**: 500行
- **特征列数**: 6列
- **目标变量**: 1列

### 数据描述
这个数据集模拟了电商平台的客户购买行为数据，用于预测客户的购买类别。

#### 特征详细说明

| 字段名 | 数据类型 | 取值范围 | 说明 |
|--------|----------|----------|------|
| age | 整数 | 21-48 | 客户年龄 |
| income | 整数 | 25000-104000 | 年收入（美元） |
| education_level | 字符串 | High School, Bachelor, Master, PhD | 教育水平 |
| years_experience | 整数 | 0-21 | 工作经验年限 |
| has_credit_card | 字符串 | Yes, No | 是否持有信用卡 |
| marital_status | 字符串 | Single, Married | 婚姻状况 |
| purchase_category | 字符串 | Electronics, Books, Clothing | 购买类别（目标变量） |

### 数据分布
- **年龄分布**: 21-48岁，主要分布在25-40岁
- **收入分布**: 25k-104k美元，呈正态分布
- **教育水平**: Bachelor最多，其次为Master和High School
- **购买类别**: Electronics (约40%), Books (约35%), Clothing (约25%)

### 使用建议
- **训练集比例**: 推荐0.7
- **预期准确率**: 75-85%
- **最佳算法**: 通常Random Forest或XGBoost表现较好

## 回归任务数据集

### 文件信息
- **文件名**: `regression_sample.csv`
- **文件大小**: ~52KB
- **数据行数**: 400行
- **特征列数**: 11列
- **目标变量**: 1列

### 数据描述
这个数据集模拟了房地产市场的房价数据，用于预测房屋价格。

#### 特征详细说明

| 字段名 | 数据类型 | 取值范围 | 说明 |
|--------|----------|----------|------|
| house_age | 整数 | 3-26 | 房屋年龄（年） |
| square_feet | 整数 | 1300-3950 | 建筑面积（平方英尺） |
| num_bedrooms | 整数 | 2-6 | 卧室数量 |
| num_bathrooms | 浮点数 | 1.0-3.5 | 浴室数量 |
| garage_size | 整数 | 0-3 | 车库大小（车位数） |
| neighborhood_quality | 整数 | 5-10 | 社区质量评分 |
| school_rating | 整数 | 5-10 | 学区评分 |
| distance_to_downtown | 浮点数 | 1.4-19.6 | 距离市中心距离（英里） |
| has_pool | 字符串 | Yes, No | 是否有游泳池 |
| has_garden | 字符串 | Yes, No | 是否有花园 |
| year_built | 整数 | 1997-2025 | 建造年份 |
| price_in_thousands | 浮点数 | 145.2-700.5 | 房价（千美元） |

### 数据分布
- **房价范围**: 145.2k - 700.5k美元
- **面积范围**: 1300 - 3950平方英尺
- **房龄分布**: 主要集中在5-25年
- **位置特征**: 距离市中心1.4-19.6英里

### 相关性分析
- **强相关特征**: square_feet, num_bedrooms, num_bathrooms
- **中等相关**: neighborhood_quality, school_rating
- **弱相关**: has_pool, has_garden, distance_to_downtown

### 使用建议
- **训练集比例**: 推荐0.7
- **预期R²分数**: 0.85-0.95
- **最佳算法**: 通常Gradient Boosting或Random Forest表现较好
- **特征工程**: 可以尝试创建房间密度、房龄等衍生特征

## 快速测试流程

### 1. 分类任务测试
```python
# 1. 启动应用
streamlit run src/MLquick.py

# 2. 在Web界面中：
# - 上传 data/samples/classification_sample.csv
# - 选择任务类型：分类
# - 选择目标变量：purchase_category
# - 训练集比例：0.7
# - 点击训练模型
```

### 2. 回归任务测试
```python
# 1. 启动应用
streamlit run src/MLquick.py

# 2. 在Web界面中：
# - 上传 data/samples/regression_sample.csv
# - 选择任务类型：回归
# - 选择目标变量：price_in_thousands
# - 训练集比例：0.7
# - 点击训练模型
```

## 预期结果

### 分类任务
- **模型对比**: 通常显示5-10个算法的性能对比
- **最佳准确率**: 约80%
- **训练时间**: 10-30秒

### 回归任务
- **模型对比**: 通常显示5-10个算法的R²、MAE、RMSE对比
- **最佳R²**: 约0.90
- **训练时间**: 15-45秒

## 扩展实验

### 1. 特征重要性分析
训练完成后，可以分析各特征对预测结果的重要性：
- 分类任务：age, income, education_level通常最重要
- 回归任务：square_feet, neighborhood_quality通常最重要

### 2. 参数调优
可以尝试调整以下参数：
- 训练集比例：0.6-0.8
- 数据预处理选项
- 特征选择方法

### 3. 新数据预测
使用训练好的模型对新数据进行预测：
- 准备相同格式的测试数据
- 使用平台的预测功能
- 分析预测结果的准确性

## 常见问题

### Q: 数据文件无法上传？
A: 检查文件格式是否为CSV，文件大小是否超过限制。

### Q: 模型训练失败？
A: 检查数据是否包含缺失值，目标变量是否正确选择。

### Q: 预测结果不准确？
A: 尝试调整训练集比例，或进行数据预处理。

## 数据来源

这些样例数据是为演示目的而生成的合成数据，模拟真实世界的特征分布和关系。数据不包含任何真实的个人信息。