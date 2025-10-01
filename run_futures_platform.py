#!/usr/bin/env python3
"""
期货行情预测平台启动脚本
"""
import os
import sys
import subprocess

def main():
    """启动期货行情预测平台"""
    print("正在启动期货行情预测平台...")

    # 检查是否在正确的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    platform_file = os.path.join(script_dir, 'futures_prediction_platform.py')

    if not os.path.exists(platform_file):
        print(f"错误: 找不到平台文件 {platform_file}")
        sys.exit(1)

    try:
        # 启动Streamlit应用
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            platform_file,
            '--server.port', '8501',
            '--server.address', '0.0.0.0'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"启动失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n平台已停止")

if __name__ == "__main__":
    main()