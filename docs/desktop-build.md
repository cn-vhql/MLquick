# MLquick Desktop Build

## 运行桌面版

```powershell
python src/mlquick_desktop.py
```

## 打包成 exe

先安装依赖：

```powershell
pip install -r requirements.txt
```

然后执行：

```powershell
pyinstaller --noconfirm --clean --windowed --name MLquickDesktop src/mlquick_desktop.py
```

打包完成后可执行文件位于：

```text
dist/MLquickDesktop/MLquickDesktop.exe
```

## 说明

- 模型、预测结果和元数据默认保存在 `%USERPROFILE%\\MLquickWorkspace`
- 当前 MVP 支持分类、回归、聚类训练与批量预测
- 聚类任务当前仅支持数值型特征
