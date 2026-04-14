# MLquick

MLquick 是一个面向普通业务人员的零代码机器学习桌面应用，当前默认形态为 `PySide6 + PyCaret` 的 Windows 桌面端。

它支持分类、回归、聚类三类任务，提供从数据导入、训练配置、模型对比、测试集预览、批量预测到结果导出的完整流程。

## 当前版本

- 当前版本：`0.0.3`
- 桌面入口：`src/mlquick_desktop.py`
- 打包配置：`MLquickDesktop.spec`
- 单文件 exe：
  - `dist/MLquickDesktop.exe`
  - `dist/MLquickDesktop-0.0.3.exe`

## 核心能力

- 分类、回归、聚类三类任务统一在桌面端完成
- 支持自动对比候选模型，也支持手动指定单模型训练
- 训练模式和预测模式完全分离，避免参数和结果混杂
- 训练结果支持摘要卡片、模型对比、测试集预览、训练图表分区展示
- 测试集预测预览保留原始字段并追加预测结果
- 支持模型导出、结果导出、批量预测、导出记录追踪
- 支持 CSV、Excel 数据集导入

## 0.0.3 更新点

- 训练结果区域改为“摘要 + 子 Tab”结构
- 子 Tab 拆分为“模型对比 / 测试集预览 / 训练图表”
- 聚类任务自动隐藏模型对比子 Tab
- 无训练图表时自动隐藏训练图表子 Tab
- 测试集预览区域默认高度调大，并支持手动调节
- 左侧配置按钮和右侧结果按钮统一缩小、统一样式
- 单文件 exe 重新打包为 `0.0.3`

## 界面说明

### 训练模式

- 左侧为训练配置区
- 右侧包含数据预览、训练结果、模型详情、运行日志
- 训练结果页分为两层：
  - 上方固定训练摘要
  - 下方子 Tab 展示详细结果

### 预测模式

- 左侧为模型管理与预测入口
- 右侧独立分成三块：
  - 预测输入预览
  - 预测结果表
  - 导出记录

## 环境准备

项目当前使用 `uv` 管理 Python 环境，推荐在 Windows 下使用现有 `.venv`。

### 1. 创建虚拟环境

```powershell
uv venv .venv
```

### 2. 安装依赖

```powershell
uv pip install -r requirements.txt
```

### 3. 启动桌面端

```powershell
.\.venv\Scripts\python.exe src\mlquick_desktop.py
```

## 打包

当前打包方式为 PyInstaller 单文件、无终端窗口模式。

```powershell
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean MLquickDesktop.spec
```

打包完成后，产物位于：

- `dist/MLquickDesktop.exe`
- `dist/MLquickDesktop-0.0.3.exe`

本次 `0.0.3` 实际单文件体积约为 `398 MB`。

## 测试

### 冒烟测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_smoke.py" -v
```

### 桌面初始化测试

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_smoke.SmokeTests.test_desktop_initialization -v
```

## 示例数据

示例数据位于 `data/samples`：

- 分类示例
- 回归示例
- 聚类示例
- 文本分类示例
- 文本回归示例

可直接用于本地训练和回归验证。

## 项目结构

```text
MLquick/
├─ assets/                  # 图标、logo 等资源
├─ data/samples/            # 示例数据
├─ dist/                    # 打包产物
├─ src/
│  ├─ __init__.py
│  ├─ mlquick_desktop.py    # PySide6 桌面端入口
│  └─ mlquick_core/         # 训练、预测、模型注册等核心逻辑
├─ tests/                   # 冒烟测试
├─ MLquickDesktop.spec      # PyInstaller 打包配置
├─ version_info.txt         # Windows 文件版本信息
└─ README.md
```

## 技术栈

- PySide6
- PyCaret 3.3.2
- pandas
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- openpyxl
- uv
- PyInstaller

## 说明

- 当前主产品形态是桌面端，不再以 Streamlit Web 页面作为默认入口
- 训练结果页优先服务“看结果”和“导结果”，所以布局会持续向信息密度和业务可读性优化
- 如需重新打包，请优先使用仓库内 `.venv`，避免依赖版本漂移
