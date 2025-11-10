#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试真实数据可视化系统
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from components.ml_engine import MLEngine

def test_classification_task():
    """测试分类任务"""
    print("=" * 60)
    print("测试分类任务")
    print("=" * 60)

    try:
        # 加载分类数据
        data_path = "../data/samples/classification_sample.csv"
        data = pd.read_csv(data_path)
        print(f"数据加载成功: {data.shape}")
        print(f"特征列: {[col for col in data.columns if col != 'target']}")
        print(f"目标变量: target")
        print(f"类别分布: {data['target'].value_counts().to_dict()}")

        # 初始化ML引擎
        ml_engine = MLEngine()

        # 训练分类模型
        print("\n开始训练分类模型...")
        result = ml_engine.train_model(
            data=data,
            task_type="classification",
            target_variable="target",
            train_size=0.7
        )

        print("[成功] 分类模型训练成功!")
        print(f"模型名称: {result['model_name']}")
        print(f"性能指标: {result['metrics']}")

        # 检查可视化
        if 'visualizations' in result and result['visualizations']:
            print(f"\n[成功] 生成的可视化图表: {list(result['visualizations'].keys())}")

            # 验证每个可视化数据
            for viz_name, viz_data in result['visualizations'].items():
                print(f"  - {viz_name}: {type(viz_data)}")
                if viz_name == 'confusion_matrix':
                    print(f"    混淆矩阵形状: {np.array(viz_data['matrix']).shape}")
                    print(f"    类别标签: {viz_data['labels']}")
                elif viz_name == 'feature_importance':
                    print(f"    特征数量: {len(viz_data['features'])}")
                    print(f"    重要性总和: {sum(viz_data['importance']):.3f}")
        else:
            print("[错误] 未生成可视化数据")

        return True

    except Exception as e:
        print(f"[失败] 分类任务测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_regression_task():
    """测试回归任务"""
    print("\n" + "=" * 60)
    print("测试回归任务")
    print("=" * 60)

    try:
        # 加载回归数据
        data_path = "../data/samples/regression_sample.csv"
        data = pd.read_csv(data_path)
        print(f"数据加载成功: {data.shape}")
        print(f"特征列: {[col for col in data.columns if col != 'price_in_thousands']}")
        print(f"目标变量: price_in_thousands")
        print(f"目标变量范围: {data['price_in_thousands'].min():.1f} - {data['price_in_thousands'].max():.1f}")

        # 处理分类特征
        for col in data.columns:
            if data[col].dtype == 'object':
                print(f"转换分类特征: {col}")
                data[col] = pd.Categorical(data[col]).codes

        # 初始化ML引擎
        ml_engine = MLEngine()

        # 训练回归模型
        print("\n开始训练回归模型...")
        result = ml_engine.train_model(
            data=data,
            task_type="regression",
            target_variable="price_in_thousands",
            train_size=0.7
        )

        print("[成功] 回归模型训练成功!")
        print(f"模型名称: {result['model_name']}")
        print(f"性能指标: {result['metrics']}")

        # 检查可视化
        if 'visualizations' in result and result['visualizations']:
            print(f"\n[成功] 生成的可视化图表: {list(result['visualizations'].keys())}")

            # 验证每个可视化数据
            for viz_name, viz_data in result['visualizations'].items():
                print(f"  - {viz_name}: {type(viz_data)}")
                if viz_name == 'residuals':
                    print(f"    残差数量: {len(viz_data['residuals'])}")
                    print(f"    残差范围: {min(viz_data['residuals']):.3f} - {max(viz_data['residuals']):.3f}")
                elif viz_name == 'prediction_scatter':
                    print(f"    预测数据形状: {viz_data.shape}")
                elif viz_name == 'feature_importance':
                    print(f"    特征数量: {len(viz_data['features'])}")
                    print(f"    重要性总和: {sum(viz_data['importance']):.3f}")
        else:
            print("[失败] 未生成可视化数据")

        return True

    except Exception as e:
        print(f"[失败] 回归任务测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_clustering_task():
    """测试聚类任务"""
    print("\n" + "=" * 60)
    print("测试聚类任务")
    print("=" * 60)

    try:
        # 加载聚类数据
        data_path = "../data/samples/clustering_sample.csv"
        data = pd.read_csv(data_path)
        print(f"数据加载成功: {data.shape}")
        print(f"特征列: {list(data.columns)}")

        # 初始化ML引擎
        ml_engine = MLEngine()

        # 训练聚类模型
        print("\n开始训练聚类模型...")
        result = ml_engine.train_model(
            data=data,
            task_type="clustering",
            n_clusters=4
        )

        print("[成功] 聚类模型训练成功!")
        print(f"模型名称: {result['model_name']}")
        print(f"聚类数量: {result['n_clusters']}")

        # 检查聚类结果
        if 'clustered_data' in result:
            cluster_counts = result['clustered_data']['Cluster'].value_counts().sort_index()
            print(f"聚类分布: {cluster_counts.to_dict()}")

        # 检查可视化
        if 'visualizations' in result and result['visualizations']:
            print(f"\n[成功] 生成的可视化图表: {list(result['visualizations'].keys())}")

            # 验证每个可视化数据
            for viz_name, viz_data in result['visualizations'].items():
                print(f"  - {viz_name}: {type(viz_data)}")
                if viz_name == 'scatter':
                    print(f"    散点图数据形状: {viz_data.shape}")
                elif viz_name == 'pie':
                    print(f"    饼图数据: {dict(viz_data)}")
                elif viz_name == 'heatmap':
                    print(f"    热力图数据形状: {viz_data.shape}")
        else:
            print("[失败] 未生成可视化数据")

        return True

    except Exception as e:
        print(f"[失败] 聚类任务测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("MLquick 真实数据可视化系统测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 测试结果
    results = {
        'classification': False,
        'regression': False,
        'clustering': False
    }

    # 执行测试
    results['classification'] = test_classification_task()
    results['regression'] = test_regression_task()
    results['clustering'] = test_clustering_task()

    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)

    for task, success in results.items():
        status = "[通过]" if success else "[失败]"
        print(f"{task.capitalize()} 任务: {status}")

    total_passed = sum(results.values())
    print(f"\n总通过率: {total_passed}/3 ({total_passed/3*100:.1f}%)")

    if total_passed == 3:
        print("\n[成功] 所有测试通过！真实数据可视化系统工作正常。")
    else:
        print(f"\n[警告] {3-total_passed} 个测试失败，需要进一步调试。")

if __name__ == "__main__":
    main()