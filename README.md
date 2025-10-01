# 🤖 AI期货预测系统

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

一个基于机器学习的期货价格预测系统，使用Streamlit构建交互式Web界面，支持多种技术指标分析和智能预测。

## ✨ 主要功能

### 📊 数据管理
- **多品种支持**: 支持多种期货品种（原油、黄金、白银、铜等）
- **实时数据**: 获取最新的期货市场数据
- **数据预览**: 完整的数据质量检查和统计信息
- **趋势分析**: 自动分析价格趋势分布

### 🎯 特征工程
- **33个技术指标**: 包括移动平均线、RSI、MACD、布林带、KDJ等
- **智能特征选择**: 可视化特征配置界面，支持自定义特征组合
- **预设配置**: 提供5种预设特征配置（基础、技术分析、全特征等）
- **特征重要性**: 自动计算和显示特征重要性排名

### 🤖 机器学习
- **多模型支持**: 随机森林、XGBoost、LightGBM、线性回归等
- **双重任务**: 支持回归（价格预测）和分类（趋势预测）
- **自动调优**: 智能超参数优化
- **性能评估**: 全面的模型评估指标和可视化

### 🔮 智能预测
- **未来趋势**: 预测未来1-10天的价格走势
- **置信度分析**: 提供预测结果的置信度评估
- **详细报告**: 自动生成专业的预测分析报告
- **可视化图表**: 直观的预测结果展示

## 🏗️ 系统架构

```
📁 ai_quick/                    # 项目根目录
│
├── 🚀 应用入口
│   ├── app.py                  # 🎯 主启动文件
│   └── setup.py                # 📦 包安装配置
│
├── 📁 src/                     # 源代码目录
│   ├── streamlit_app.py        # 🎨 主应用界面
│   ├── data_fetcher.py         # 📊 数据获取模块
│   ├── data_processor.py       # 🔧 数据处理和特征工程
│   ├── model_trainer.py        # 🤖 模型训练和评估
│   ├── model_predictor.py      # 🔮 预测和报告生成
│   ├── model_config.py         # ⚙️ 模型配置管理
│   └── feature_library.py      # 📖 特征库管理系统
│
├── 📁 config/                  # 配置文件目录
│   └── feature_configs.json    # 💾 特征配置文件（自动生成）
│
├── 📁 docs/                    # 文档目录
│   ├── FEATURE_CONFIG_GUIDE.md # 📋 特征配置使用指南
│   └── PROJECT_OVERVIEW.md     # 📋 项目结构概览
│
├── 📄 项目文档
│   ├── README.md               # 📖 项目主要说明文档
│   ├── CHANGELOG.md            # 📈 版本更新日志
│   ├── LICENSE                 # ⚖️ GPL v3 开源许可证
│   ├── requirements.txt        # 📦 Python依赖包列表
│   └── .gitignore              # 🚫 Git忽略文件
│
└── 📁 tests/                   # 测试目录 (待开发)
```

### 核心模块说明

| 模块 | 功能 | 主要类/函数 |
|------|------|-------------|
| **streamlit_app.py** | Web界面和用户交互 | `main()`, `render_*_tab()` |
| **data_fetcher.py** | 期货数据获取 | `get_futures_data()` |
| **data_processor.py** | 数据处理和特征工程 | `calculate_technical_indicators()`, `create_features_targets()` |
| **model_trainer.py** | 模型训练和评估 | `train_complete_workflow()`, `evaluate_model_performance()` |
| **model_predictor.py** | 预测和报告生成 | `predict_future_trend()`, `generate_prediction_report()` |
| **model_config.py** | 模型配置管理 | `ModelConfig`, `ModelEvaluator` |
| **feature_library.py** | 特征库管理 | `FeatureLibrary`, `FeatureConfig` |

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 2GB+ RAM
- 网络连接（获取数据）

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/yourusername/ai_quick.git
cd ai_quick
```

2. **安装依赖**
```bash
# 方式1: 使用requirements.txt
pip install -r requirements.txt

# 方式2: 使用setup.py（推荐）
pip install -e .
```

3. **启动应用**
```bash
# 方式1: 使用streamlit直接运行
streamlit run app.py

# 方式2: 使用命令行工具（安装后）
ai-futures
```

4. **访问应用**
打开浏览器访问 `http://localhost:8501`

## 📖 使用指南

### 基础使用流程

1. **📊 数据获取**
   - 选择期货品种
   - 设置日期范围
   - 点击"获取数据"

2. **⚙️ 特征配置**
   - 选择预设配置或自定义特征
   - 保存特征配置（可选）

3. **🤖 模型训练**
   - 配置训练参数
   - 选择任务类型（回归/分类）
   - 点击"开始训练"

4. **🔮 查看预测**
   - 查看训练结果和模型性能
   - 生成未来预测报告
   - 分析特征重要性

### 高级功能

#### 特征配置管理
- 访问"⚙️ 特征配置"标签页
- 使用预设配置快速开始
- 自定义选择特定技术指标
- 保存和加载特征配置

详细使用说明请参考：[docs/FEATURE_CONFIG_GUIDE.md](./docs/FEATURE_CONFIG_GUIDE.md)

## 🔧 技术指标说明

### 基础指标
- **OHLCV**: 开盘价、最高价、最低价、收盘价、成交量
- **移动平均线**: MA5、MA10、MA20
- **价格变化**: 1日、3日、5日价格变化率

### 技术指标
- **RSI**: 相对强弱指数（超买超卖指标）
- **MACD**: 指数平滑异同移动平均线
- **布林带**: 价格通道指标
- **KDJ**: 随机指标
- **威廉指标**: 超买超卖指标

### 高级指标
- **ATR**: 平均真实波幅（波动率）
- **VWAP**: 成交量加权平均价
- **OBV**: 能量潮指标
- **CCI**: 商品通道指数

## 📊 模型性能

### 支持的算法
- **随机森林** (Random Forest)
- **XGBoost** (Extreme Gradient Boosting)
- **LightGBM** (Light Gradient Boosting)
- **线性回归** (Linear Regression)
- **支持向量机** (Support Vector Machine)
- **决策树** (Decision Tree)

### 评估指标
- **回归任务**: R²、MAE、MSE、RMSE
- **分类任务**: 准确率、精确率、召回率、F1分数
- **混淆矩阵**: 详细分类结果分析

## 🛠️ 开发指南

### 添加新特征

1. 在 `data_processor.py` 的 `calculate_technical_indicators()` 函数中添加计算逻辑
2. 在 `feature_library.py` 中定义新特征的配置
3. 更新特征描述和类别

### 添加新模型

1. 在 `model_config.py` 的 `get_default_config()` 中添加模型配置
2. 在相应的训练函数中添加模型训练逻辑
3. 更新模型评估和可视化代码

### 自定义配置

所有主要参数都可以通过界面或配置文件调整：

- **训练参数**: 历史天数、预测天数、训练集比例
- **特征配置**: 启用的技术指标列表
- **模型参数**: 各算法的超参数设置

## ⚠️ 重要声明

### 投资风险提示
⚠️ **本系统仅供学习和研究使用，不构成投资建议**

- 期货市场存在重大风险，可能导致资金损失
- 预测结果基于历史数据，不保证未来准确性
- 投资决策应结合多种分析方法
- 请根据自身风险承受能力谨慎投资

### 技术限制
- 数据质量依赖于数据源
- 模型性能受市场环境影响
- 极端市场条件下预测可能失效
- 建议定期重新训练模型

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

### 贡献流程
1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范
- 遵循 PEP 8 代码风格
- 添加适当的注释和文档字符串
- 确保所有测试通过
- 更新相关文档

## 📄 许可证

本项目采用 **GPL v3** 开源许可证。详情请参考 [LICENSE](LICENSE) 文件。

### GPL v3 主要条款
- ✅ 自由使用、修改、分发
- ✅ 商业使用
- ⚠️ 修改后的代码必须开源
- ⚠️ 需要包含许可证和版权声明
- ⚠️ 提供源代码访问途径

## 📞 联系方式

- **项目主页**: https://github.com/yourusername/ai_quick
- **问题反馈**: https://github.com/yourusername/ai_quick/issues
- **邮箱**: yl_zhangqiang@foxmail.com

## 🙏 致谢

感谢以下开源项目的支持：
- [Streamlit](https://streamlit.io/) - Web应用框架
- [pandas](https://pandas.pydata.org/) - 数据处理
- [scikit-learn](https://scikit-learn.org/) - 机器学习
- [yfinance](https://pypi.org/project/yfinance/) - 金融数据
- [matplotlib](https://matplotlib.org/) - 数据可视化
- [seaborn](https://seaborn.pydata.org/) - 统计可视化

---

**⭐ 如果这个项目对你有帮助，请给个Star支持一下！**

## 📈 更新日志

### v1.0.0 (最新)
- ✨ 完整的特征配置管理系统
- 🤖 多模型机器学习支持
- 📊 交互式Web界面
- 📈 期货价格预测功能
- 🔮 智能报告生成
- 💾 配置保存和加载

### 未来计划
- 🌐 多市场数据源支持
- 📱 移动端适配
- 🔄 实时数据更新
- 🎯 策略回测功能
- 📊 更多技术指标
- 🤖 深度学习模型支持

---

**免责声明**: 本项目仅用于教育和研究目的。使用本系统进行实际交易的任何损失由用户自行承担。