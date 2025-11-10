@echo off
chcp 65001 > nul
echo.
echo ====================================
echo    MLquick GUI 一键启动器
echo ====================================
echo.

REM 检查Python是否可用
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未找到Python，请先安装Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python已安装

REM 切换到脚本目录
cd /d "%~dp0"

echo 当前目录: %CD%

REM 检查start_gui.py是否存在
if not exist "start_gui.py" (
    echo ❌ 未找到启动脚本 start_gui.py
    echo 请确保在正确的目录中运行此脚本
    pause
    exit /b 1
)

echo.
echo 启动MLquick GUI...
echo.

REM 启动应用
python start_gui.py

REM 如果应用异常退出，显示提示
if %errorlevel% neq 0 (
    echo.
    echo ❌ 应用异常退出
    echo.
    echo 可能的解决方案:
    echo 1. 运行 python check_env.py 检查环境
    echo 2. 运行 python fix_dependencies.py 修复依赖
    echo 3. 运行 pip install -r requirements.txt 安装依赖
    echo.
    pause
) else (
    echo.
    echo 应用已正常退出
)

echo.
echo 感谢使用 MLquick GUI！
pause