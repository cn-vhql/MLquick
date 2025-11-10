#!/bin/bash

# MLquick GUI 启动脚本 (Linux/macOS)

echo "===================================="
echo "   MLquick GUI 一键启动器"
echo "===================================="
echo

# 检查Python是否可用
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ 未找到Python，请先安装Python 3.8+"
        echo "Ubuntu/Debian: sudo apt install python3 python3-pip"
        echo "CentOS/RHEL: sudo yum install python3 python3-pip"
        echo "macOS: brew install python3"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "✅ Python已安装: $("$PYTHON_CMD" --version 2>&1)"

# 切换到脚本目录
cd "$(dirname "$0")"
echo "当前目录: $(pwd)"

# 检查启动脚本是否存在
if [ ! -f "start_gui.py" ]; then
    echo "❌ 未找到启动脚本 start_gui.py"
    echo "请确保在正确的目录中运行此脚本"
    exit 1
fi

echo
echo "启动MLquick GUI..."
echo

# 启动应用
"$PYTHON_CMD" start_gui.py
exit_code=$?

# 处理异常退出
if [ $exit_code -ne 0 ]; then
    echo
    echo "❌ 应用异常退出 (退出码: $exit_code)"
    echo
    echo "可能的解决方案:"
    echo "1. 运行 $PYTHON_CMD check_env.py 检查环境"
    echo "2. 运行 $PYTHON_CMD fix_dependencies.py 修复依赖"
    echo "3. 运行 $PYTHON_CMD -m pip install -r requirements.txt 安装依赖"
    echo
    read -p "按Enter键退出..."
else
    echo
    echo "应用已正常退出"
fi

echo
echo "感谢使用 MLquick GUI！"