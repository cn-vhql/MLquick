#!/usr/bin/env python3
"""
期货行情预测平台启动脚本 - 模块化版本
"""
import os
import sys
import subprocess

def main():
    """启动期货行情预测平台"""
    print("正在启动期货行情预测平台...")
    print("平台架构: 模块化设计")
    print("模块包括: 数据获取、数据处理、模型配置、模型训练、模型预测、主界面")

    # 检查是否在正确的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    platform_file = os.path.join(script_dir, 'streamlit_app.py')

    if not os.path.exists(platform_file):
        print(f"错误: 找不到主界面文件 {platform_file}")
        print("请确保以下模块文件都存在:")
        required_files = [
            'streamlit_app.py',
            'data_fetcher.py',
            'data_processor.py',
            'model_config.py',
            'model_trainer.py',
            'model_predictor.py'
        ]
        for file in required_files:
            file_path = os.path.join(script_dir, file)
            status = "✓" if os.path.exists(file_path) else "✗"
            print(f"  {status} {file}")
        sys.exit(1)

    # 检查依赖模块
    print("\n检查模块依赖...")
    required_modules = [
        'streamlit',
        'pandas',
        'numpy',
        'akshare',
        'matplotlib',
        'seaborn',
        'scikit-learn'
    ]

    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"  ✗ {module} (未安装)")

    if missing_modules:
        print(f"\n警告: 缺少以下依赖模块: {', '.join(missing_modules)}")
        print("请运行: pip install -r requirements_futures.txt")

    try:
        print(f"\n启动Streamlit应用: {platform_file}")
        print("访问地址: http://localhost:8501")
        print("按 Ctrl+C 停止平台\n")

        # 启动Streamlit应用
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            platform_file,
            '--server.port', '8501',
            '--server.address', '0.0.0.0',
            '--server.headless', 'false'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"启动失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查端口8501是否被占用")
        print("2. 确保所有依赖模块已正确安装")
        print("3. 检查Python环境是否正确")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n平台已停止")
        print("感谢使用期货行情预测平台！")

if __name__ == "__main__":
    main()