#!/bin/bash
# AI期货预测系统启动脚本
# AI Futures Prediction System Launch Script

echo "🚀 正在启动AI期货预测系统..."
echo "🚀 Starting AI Futures Prediction System..."

# 检查Python版本
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✅ Python版本: $python_version"
else
    echo "❌ 错误：未找到Python3"
    echo "❌ Error: Python3 not found"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖包..."
echo "📦 Checking dependencies..."
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "❌ 错误：未安装Streamlit"
    echo "❌ Error: Streamlit not installed"
    echo "请运行: pip install -r requirements.txt"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

echo "✅ 依赖检查完成"
echo "✅ Dependencies check completed"

# 启动应用
echo "🎯 启动Streamlit应用..."
echo "🎯 Launching Streamlit application..."
echo "📱 访问地址: http://localhost:8501"
echo "📱 URL: http://localhost:8501"
echo ""

# 启动命令
cd "$(dirname "$0")"
python3 -m streamlit run app.py --server.headless=false --server.port=8501

echo ""
echo "👋 感谢使用AI期货预测系统！"
echo "👋 Thank you for using AI Futures Prediction System!"