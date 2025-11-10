# MLquick GUI - 机器学习零代码桌面应用

MLquick GUI版本是基于Tkinter开发的桌面应用程序，提供完整的机器学习零代码解决方案，支持分类、回归和聚类任务。

## 🚀 快速启动

### Windows用户
1. **双击运行**: `启动MLquick.bat` ⭐
2. **命令行运行**: `python start_gui.py`

### Linux/macOS用户
1. **终端运行**: `bash start_mlquick.sh`
2. **命令行运行**: `python3 start_gui.py`

### 启动流程
1. 自动检查依赖安装情况
2. 根据环境选择启动模式
3. 智能错误处理和备用方案

## 🎯 功能特性

### 核心功能
- **分类任务**: 支持二分类和多分类问题
- **回归任务**: 支持数值预测问题
- **聚类任务**: 支持无监督聚类分析
- **文本处理**: 中英文文本预处理和特征提取
- **模型管理**: 模型保存、加载、导入导出
- **数据可视化**: 基于Matplotlib的丰富图表

### 数据处理
- 支持CSV、Excel文件格式
- 自动数据类型检测
- 缺失值处理
- 数据预览和统计分析

### 文本处理
- 中英文分词（jieba + NLTK）
- 停用词过滤
- TF-IDF特征提取
- 词云图生成

### 可视化功能
- 散点图和3D散点图
- 饼图和热力图
- 词云图
- 模型性能对比图
- 特征重要性图

### 模型管理
- 模型自动保存
- 模型导入导出（ZIP格式）
- 模型版本管理
- 模型性能统计

## 🔧 环境要求

### 依赖库
- **必需**: pandas, numpy, matplotlib, tkinter, openpyxl, Pillow
- **机器学习**: scikit-learn, seaborn
- **高级功能**: PyCaret, jieba, wordcloud, nltk

### 安装步骤

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **启动应用**
```bash
# Windows
python start_gui.py
# 或双击 启动MLquick.bat

# Linux/macOS
python3 start_gui.py
# 或运行 bash start_mlquick.sh
```

## 🔍 故障排除

### 常见问题

1. **启动时提示依赖缺失**
```bash
# 安装基础依赖
pip install pandas numpy matplotlib openpyxl Pillow

# 安装机器学习依赖
pip install scikit-learn seaborn

# 安装完整功能依赖
pip install pycaret==3.3.2 jieba wordcloud nltk
```

2. **PyCaret不可用**
- 现象: 提示"机器学习功能受限"
- 解决: `pip install pycaret==3.3.2`

3. **中文显示异常**
- Windows: 系统已内置中文字体
- Linux: `sudo apt install fonts-wqy-microhei`
- macOS: 系统已内置中文字体

4. **启动后立即退出**
- 检查Python版本是否≥3.8
- 确保在正确的目录中运行
- 查看终端输出的错误信息

## 📁 项目结构

```
gui/
├── main.py                 # 主应用程序
├── start_gui.py           # 启动脚本 ⭐
├── 启动MLquick.bat         # Windows一键启动
├── start_mlquick.sh       # Linux/macOS启动脚本
├── requirements.txt        # 依赖列表
├── README.md              # 说明文档
├── components/            # 核心组件
│   ├── data_manager.py    # 数据管理
│   ├── ml_engine.py       # 机器学习引擎
│   ├── text_processor.py  # 文本处理
│   ├── visualizer.py      # 可视化组件
│   └── model_manager.py   # 模型管理
├── utils/                 # 工具函数
│   ├── ui_utils.py        # UI工具
│   └── config.py          # 配置文件
└── assets/               # 资源文件
```

## 🛠️ 技术栈

- **GUI框架**: Tkinter (Python内置)
- **数据处理**: Pandas, NumPy
- **机器学习**: PyCaret, Scikit-learn
- **可视化**: Matplotlib, Seaborn
- **文本处理**: jieba, NLTK, wordcloud
- **图像处理**: Pillow

## 📖 使用指南

### 基本流程
1. **上传数据** → CSV/Excel文件
2. **选择任务** → 分类/回归/聚类
3. **配置参数** → 目标变量、训练比例等
4. **训练模型** → 自动算法选择和优化
5. **查看结果** → 性能指标、可视化图表
6. **模型管理** → 保存、加载、预测

### 功能特色
- ✅ **智能启动**: 自动检测依赖，选择最佳启动模式
- ✅ **多格式支持**: CSV、Excel，自动编码检测
- ✅ **中文友好**: 完整中文字体和文本处理支持
- ✅ **一键操作**: 批处理文件直接启动
- ✅ **错误处理**: 完善的异常捕获和用户提示

## 🎯 许可证

本项目采用 MIT 许可证，可自由使用和分发。

---

**MLquick GUI** - 让机器学习变得简单高效！ 🚀