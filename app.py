#!/usr/bin/env python3
"""
AI期货预测系统 - 主启动文件
AI Futures Prediction System - Main Launch File

这是项目的主要入口点，用于启动Streamlit应用。
This is the main entry point for launching the Streamlit application.

使用方法 Usage:
    streamlit run app.py

作者 Author: AI Quick Team
许可证 License: GPL v3
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    # 导入主应用模块
    import streamlit_app as main_app

    # 启动应用
    if __name__ == "__main__":
        main_app.main()

except ImportError as e:
    print(f"❌ 导入错误 Import Error: {e}")
    print("请确保已安装所有依赖包 / Please ensure all dependencies are installed")
    print("运行: pip install -r requirements.txt")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

except Exception as e:
    print(f"❌ 启动错误 Startup Error: {e}")
    sys.exit(1)