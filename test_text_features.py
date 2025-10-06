#!/usr/bin/env python3
"""
测试文本功能的简单脚本
"""

import pandas as pd
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入文本处理函数
try:
    from MLquick import (
        detect_language,
        preprocess_text_column,
        extract_text_features,
        create_text_visualizations
    )
    print("✅ 文本处理函数导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def test_text_classification_data():
    """测试文本分类样例数据"""
    print("\n=== 测试文本分类数据 ===")

    file_path = "data/samples/text_classification_sample.csv"
    df = pd.read_csv(file_path)

    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print("前3行数据:")
    print(df.head(3))

    # 测试语言检测
    sample_text = df['text'].iloc[0]
    detected_lang = detect_language(sample_text)
    print(f"\n语言检测: '{sample_text}' -> {detected_lang}")

    # 测试文本预处理
    print("\n测试文本预处理...")
    processed_text = preprocess_text_column(df['text'][:5])
    print("预处理结果:")
    for i, (original, processed) in enumerate(zip(df['text'][:5], processed_text)):
        print(f"{i+1}. {original[:30]}... -> {processed[:50]}...")

    return df

def test_text_regression_data():
    """测试文本回归样例数据"""
    print("\n=== 测试文本回归数据 ===")

    file_path = "data/samples/text_regression_sample.csv"
    df = pd.read_csv(file_path)

    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print("前3行数据:")
    print(df.head(3))

    return df

def test_text_clustering_data():
    """测试文本聚类样例数据"""
    print("\n=== 测试文本聚类数据 ===")

    file_path = "data/samples/text_clustering_sample.csv"
    df = pd.read_csv(file_path)

    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print("前3行数据:")
    print(df.head(3))

    # 测试特征提取
    print("\n测试文本特征提取...")
    processed_text = preprocess_text_column(df['text_content'][:10])

    if processed_text.str.strip().any():
        features, feature_names, vectorizer = extract_text_features(processed_text, max_features=100)
        if features is not None:
            print(f"特征矩阵形状: {features.shape}")
            print(f"特征名称示例: {feature_names[:10]}")
        else:
            print("特征提取失败")
    else:
        print("预处理后的文本为空")

    return df

def test_mixed_language_detection():
    """测试混合语言检测"""
    print("\n=== 测试语言检测 ===")

    test_texts = [
        "这是一个中文句子",
        "This is an English sentence",
        "这是一个中文 and English mixed sentence",
        "",
        "12345",
        "中文字符数量多过english"
    ]

    for text in test_texts:
        lang = detect_language(text)
        print(f"'{text}' -> {lang}")

def main():
    """主测试函数"""
    print("开始测试MLquick文本功能...")

    try:
        # 测试各个数据集
        test_text_classification_data()
        test_text_regression_data()
        test_text_clustering_data()
        test_mixed_language_detection()

        print("\n✅ 所有测试完成！文本功能基本正常。")

    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)