#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLquick GUI 快速启动脚本
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """检查关键依赖"""
    print("检查依赖包...")

    # 基础依赖（必需）
    basic_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib.pyplot'),
        ('tkinter', 'tkinter'),  # 内置
        ('openpyxl', 'openpyxl'),
        ('Pillow', 'PIL'),  # Pillow是包名，PIL是导入名
    ]

    # ML依赖（可选）
    ml_packages = [
        ('scikit-learn', 'sklearn'),
        ('seaborn', 'seaborn'),
    ]

    # 高级ML依赖（可选）
    advanced_ml_packages = [
        ('pycaret', 'pycaret'),
        ('jieba', 'jieba'),
        ('wordcloud', 'wordcloud'),
        ('nltk', 'nltk'),
    ]

    missing_basic = []
    missing_ml = []
    missing_advanced = []

    # 检查基础依赖
    print("\n基础依赖:")
    for name, import_name in basic_packages:
        try:
            __import__(import_name)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name}")
            missing_basic.append(name)

    # 检查ML依赖
    print("\n机器学习依赖:")
    for name, import_name in ml_packages:
        try:
            __import__(import_name)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name}")
            missing_ml.append(name)

    # 检查高级ML依赖
    print("\n高级功能依赖:")
    for name, import_name in advanced_ml_packages:
        try:
            __import__(import_name)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name}")
            missing_advanced.append(name)

    # 如果基础依赖缺失，无法启动
    if missing_basic:
        print(f"\n❌ 缺少必需的基础依赖: {', '.join(missing_basic)}")
        print("请安装:")
        print("pip install pandas numpy matplotlib openpyxl Pillow")
        return False, "基础依赖缺失"

    # 如果部分ML依赖缺失，可以使用简化版
    if missing_ml or missing_advanced:
        print(f"\n⚠️ 部分功能依赖缺失:")
        if missing_ml:
            print(f"  机器学习: {', '.join(missing_ml)}")
        if missing_advanced:
            print(f"  高级功能: {', '.join(missing_advanced)}")
        print("应用将以简化模式启动")
        return True, "部分功能缺失"

    print("\n✅ 所有依赖都已安装!")
    return True, "完整功能"

def main():
    """主函数"""
    print("MLquick GUI 启动器")
    print("=" * 40)

    # 检查依赖
    deps_ok, status = check_dependencies()

    if not deps_ok:
        print("\n请安装缺失的依赖后重试")
        input("按Enter键退出...")
        return

    try:
        print(f"\n启动MLquick GUI... ({status})")
        print("-" * 40)

        # 尝试导入主应用
        from main import MLquickGUI

        # 检查PyCaret是否可用
        try:
            from pycaret.classification import setup
            print("✅ PyCaret可用 - 机器学习功能正常")
            ml_available = True
        except ImportError:
            print("⚠️ PyCaret不可用 - 机器学习功能受限")
            ml_available = False

        app = MLquickGUI()

        # 如果ML功能不可用，显示提示
        if not ml_available:
            tk.messagebox.showinfo("功能提示",
                "机器学习功能不可用，但您仍可以使用:\n"
                "• 数据加载和预览\n"
                "• 数据可视化\n"
                "• 基础统计分析\n\n"
                "如需完整功能，请安装PyCaret:\n"
                "pip install pycaret==3.3.2")

        print("✅ 应用启动成功!")
        app.run()

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请检查Python环境和依赖安装")
        messagebox.showerror("启动错误", f"应用导入失败:\n{e}\n\n请检查依赖是否正确安装")

        # 提供备用选项
        choice = messagebox.askyesno("备用选项",
            "完整版启动失败。\n"
            "是否启动数据分析版(仅支持数据可视化和基础分析)?")

        if choice:
            try:
                print("启动备用版本...")
                import fallback_app
                backup_app = fallback_app.MLquickFallback()
                backup_app.run()
            except ImportError:
                messagebox.showerror("错误", "备用版本也无法启动，请检查基础依赖")

    except Exception as e:
        print(f"❌ 应用错误: {e}")
        messagebox.showerror("运行错误", f"应用运行时出错:\n{e}")

    finally:
        print("\n应用已退出")

if __name__ == "__main__":
    main()