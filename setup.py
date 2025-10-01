#!/usr/bin/env python3
"""
AI期货预测系统安装配置
AI Futures Prediction System Setup Configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-futures-prediction",
    version="1.0.0",
    author="AI Quick Team",
    author_email="your.email@example.com",
    description="基于机器学习的期货价格预测系统 / Machine Learning-based Futures Price Prediction System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai_quick",
    license="GPL v3",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ai-futures=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md", "*.txt"],
    },
    keywords="futures prediction machine learning technical indicators streamlit",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ai_quick/issues",
        "Source": "https://github.com/yourusername/ai_quick",
        "Documentation": "https://github.com/yourusername/ai_quick/blob/main/README.md",
    },
)