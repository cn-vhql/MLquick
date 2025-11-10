#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化组件 - 处理图表生成和显示
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List, Optional
import warnings

# 尝试导入词云库
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    warnings.warn("wordcloud未安装，词云图功能将不可用")

class Visualizer:
    """可视化器"""

    def __init__(self, parent):
        self.parent = parent
        self.current_figures = {}
        self.canvas_widgets = {}

        # 设置matplotlib中文字体支持
        self._setup_chinese_font()

        # 设置seaborn样式
        try:
            sns.set_style("whitegrid")
            plt.style.use('seaborn-v0_8')
        except:
            # 如果新版本不可用，使用旧版本
            try:
                plt.style.use('seaborn')
            except:
                # 如果都不可用，使用默认样式
                try:
                    plt.style.use('default')
                except:
                    pass

    def _setup_chinese_font(self):
        """设置matplotlib中文字体支持"""
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            warnings.warn("无法设置中文字体，图表中的中文可能显示为方框")

    def get_chinese_font_path(self):
        """获取可用的中文字体路径"""
        import matplotlib.font_manager as fm

        # 常见的中文字体路径（Windows）
        chinese_font_paths = [
            'C:/Windows/Fonts/msyh.ttc',      # Microsoft YaHei
            'C:/Windows/Fonts/msyhbd.ttc',    # Microsoft YaHei Bold
            'C:/Windows/Fonts/simhei.ttf',    # SimHei
            'C:/Windows/Fonts/simsun.ttc',    # SimSun
            'C:/Windows/Fonts/simkai.ttf',    # KaiTi
        ]

        # 检查字体文件是否存在
        for font_path in chinese_font_paths:
            if os.path.exists(font_path):
                return font_path

        return None

    def clear_visualizations(self):
        """清除所有可视化内容"""
        for canvas in self.canvas_widgets.values():
            canvas.get_tk_widget().destroy()

        self.current_figures.clear()
        self.canvas_widgets.clear()

    def display_visualizations(self, visualizations: Dict[str, Any]):
        """显示可视化图表"""
        self.clear_visualizations()

        # 查找可视化框架
        viz_frame = self._find_visualization_frame()
        if viz_frame is None:
            return

        # 清空可视化框架
        for widget in viz_frame.winfo_children():
            widget.destroy()

        # 创建滚动区域
        canvas = tk.Canvas(viz_frame)
        scrollbar = ttk.Scrollbar(viz_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 生成图表
        if not visualizations:
            # 如果没有可视化数据，显示提示信息
            label = ttk.Label(scrollable_frame, text="暂无可视化数据",
                             font=("Arial", 12), foreground="gray")
            label.pack(pady=50)
        else:
            row = 0
            for viz_name, viz_data in visualizations.items():
                try:
                    print(f"开始创建图表: {viz_name}, 数据类型: {type(viz_data)}, 数据形状: {getattr(viz_data, 'shape', 'N/A')}")

                    if viz_name.startswith('wordcloud'):
                        fig = self._create_wordcloud_visualization(viz_data, viz_name)
                    elif viz_name == 'scatter':
                        fig = self._create_scatter_plot(viz_data)
                    elif viz_name == 'scatter_3d':
                        fig = self._create_3d_scatter_plot(viz_data)
                    elif viz_name == 'pie':
                        print(f"饼图数据详情: {viz_data}")
                        fig = self._create_pie_chart(viz_data)
                    elif viz_name == 'heatmap':
                        print(f"热力图数据详情: {viz_data}")
                        fig = self._create_heatmap(viz_data)
                    elif viz_name == 'feature_importance':
                        fig = self._create_feature_importance_plot(viz_data)
                    elif viz_name == 'model_comparison':
                        fig = self._create_model_comparison_plot(viz_data)
                    elif viz_name == 'metrics_radar':
                        fig = self._create_metrics_radar_plot(viz_data)
                    elif viz_name == 'metrics_comparison':
                        fig = self._create_metrics_comparison_plot(viz_data)
                    elif viz_name == 'confusion_matrix':
                        fig = self._create_confusion_matrix_plot(viz_data)
                    elif viz_name == 'residuals':
                        fig = self._create_residuals_plot(viz_data)
                    elif viz_name == 'residual_histogram':
                        fig = self._create_residual_histogram_plot(viz_data)
                    elif viz_name == 'qq_plot':
                        fig = self._create_qq_plot(viz_data)
                    elif viz_name == 'prediction_scatter':
                        fig = self._create_prediction_scatter_plot(viz_data)
                    else:
                        continue

                    print(f"图表 {viz_name} 创建成功: {fig is not None}")

                    if fig:
                        # 创建matplotlib画布
                        canvas_widget = FigureCanvasTkAgg(fig, scrollable_frame)
                        canvas_widget.draw()
                        canvas_widget.get_tk_widget().grid(row=row, column=0, padx=10, pady=10, sticky="ew")

                        # 暂时跳过导航工具栏以避免几何管理器冲突
                        # 如果需要工具栏，可以改为在图表下方添加简单的缩放按钮
                        print("跳过工具栏创建以避免几何管理器冲突")

                        self.canvas_widgets[viz_name] = canvas_widget
                        row += 2

                except Exception as e:
                    print(f"图表 {viz_name} 生成失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # 显示错误信息
                    error_label = ttk.Label(scrollable_frame,
                                         text=f"图表 {viz_name} 生成失败: {str(e)}",
                                         font=("Arial", 10), foreground="red")
                    error_label.grid(row=row, column=0, padx=10, pady=5)
                    row += 1

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _find_visualization_frame(self):
        """查找可视化框架"""
        # 从主窗口开始查找
        def search_frame(widget):
            # 检查是否包含可视化相关的标签页
            for child in widget.winfo_children():
                if isinstance(child, ttk.Notebook):
                    for i in range(child.index('end')):
                        try:
                            tab_text = child.tab(i, 'text')
                            if '可视' in tab_text or 'Visualization' in tab_text:
                                return child.winfo_children()[i]  # 返回对应的框架
                        except:
                            continue
                # 递归搜索子框架
                if hasattr(child, 'winfo_children'):
                    result = search_frame(child)
                    if result:
                        return result
            return None

        frame = search_frame(self.parent)

        # 调试信息
        if frame is None:
            print("警告: 未找到可视化框架")
            # 添加更详细的调试信息
            print(f"搜索起点: {self.parent}")
            print(f"窗口类型: {type(self.parent)}")
            self._debug_widget_tree(self.parent)
        else:
            print(f"找到可视化框架: {frame}")

        return frame

    def _debug_widget_tree(self, widget, level=0):
        """调试widget树结构"""
        indent = "  " * level
        try:
            widget_info = f"{type(widget).__name__}"
            if hasattr(widget, 'winfo_children'):
                children = widget.winfo_children()
                print(f"{indent}{widget_info} (子组件数: {len(children)})")
                for child in children:
                    if isinstance(child, ttk.Notebook):
                        print(f"{indent}  Notebook:")
                        for i in range(child.index('end')):
                            try:
                                tab_text = child.tab(i, 'text')
                                print(f"{indent}    Tab {i}: '{tab_text}'")
                            except:
                                print(f"{indent}    Tab {i}: 无法获取文本")
                    elif hasattr(child, 'winfo_children'):
                        self._debug_widget_tree(child, level + 1)
        except Exception as e:
            print(f"{indent}调试错误: {e}")

    def _create_wordcloud_visualization(self, text_data, title="词云图"):
        """创建词云图"""
        if not WORDCLOUD_AVAILABLE:
            return None

        try:
            # 检测是否包含中文
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in str(text_data))

            font_path = None
            if has_chinese:
                font_path = self.get_chinese_font_path()

            # 创建词云图
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                font_path=font_path,
                colormap='viridis',
                relative_scaling=0.5,
                min_font_size=10
            ).generate(str(text_data))

            # 创建图像
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(title, fontsize=16, pad=20)

            plt.tight_layout()
            return fig

        except Exception as e:
            warnings.warn(f"词云图生成失败: {str(e)}")
            return None

    def _create_scatter_plot(self, data):
        """创建散点图"""
        try:
            print(f"开始创建散点图，数据: {data}")
            fig, ax = plt.subplots(figsize=(10, 6))

            if isinstance(data, pd.DataFrame):
                print(f"DataFrame列数: {len(data.columns)}, 列名: {data.columns}")
                # 假设数据包含x, y, color列
                if len(data.columns) >= 3:
                    print("创建带颜色的散点图")
                    # 处理字符串标签，转换为数字
                    color_data = data.iloc[:, 2]
                    if color_data.dtype == 'object':
                        # 如果是字符串标签，转换为数字
                        unique_labels = color_data.unique()
                        label_to_num = {label: i for i, label in enumerate(unique_labels)}
                        numeric_colors = color_data.map(label_to_num)
                        print(f"将标签转换为数字: {dict(zip(unique_labels, range(len(unique_labels))))}")

                        scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1],
                                           c=numeric_colors, cmap='viridis', alpha=0.6)
                        # 创建自定义图例
                        for label, num in label_to_num.items():
                            ax.scatter([], [], c=[plt.cm.viridis(num / len(unique_labels))],
                                     label=label, alpha=0.6)
                        ax.legend(title=data.columns[2])
                    else:
                        scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1],
                                           c=color_data, cmap='viridis', alpha=0.6)

                    plt.colorbar(scatter, ax=ax)
                    ax.set_xlabel(data.columns[0])
                    ax.set_ylabel(data.columns[1])
                    ax.set_title(f"{data.columns[0]} vs {data.columns[1]}")
                else:
                    print("创建简单散点图")
                    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], alpha=0.6)
                    ax.set_xlabel(data.columns[0])
                    ax.set_ylabel(data.columns[1])
                    ax.set_title(f"{data.columns[0]} vs {data.columns[1]}")
            else:
                # 如果数据是字典
                print("处理字典格式数据")
                x = data.get('x', [])
                y = data.get('y', [])
                colors = data.get('colors', None)
                labels = data.get('labels', {})

                if colors is not None:
                    scatter = ax.scatter(x, y, c=colors, cmap='viridis', alpha=0.6)
                    plt.colorbar(scatter, ax=ax)
                else:
                    ax.scatter(x, y, alpha=0.6)

                ax.set_xlabel(labels.get('x', 'X'))
                ax.set_ylabel(labels.get('y', 'Y'))
                ax.set_title(labels.get('title', '散点图'))

            plt.tight_layout()
            print("散点图创建成功")
            return fig

        except Exception as e:
            print(f"散点图生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _create_3d_scatter_plot(self, data):
        """创建3D散点图"""
        try:
            print(f"开始创建3D散点图，数据: {data}")
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            if isinstance(data, pd.DataFrame):
                print(f"3D DataFrame列数: {len(data.columns)}, 列名: {data.columns}")
                if len(data.columns) >= 4:
                    print("创建带颜色的3D散点图")
                    # 处理字符串标签，转换为数字
                    color_data = data.iloc[:, 3]
                    if color_data.dtype == 'object':
                        # 如果是字符串标签，转换为数字
                        unique_labels = color_data.unique()
                        label_to_num = {label: i for i, label in enumerate(unique_labels)}
                        numeric_colors = color_data.map(label_to_num)
                        print(f"3D将标签转换为数字: {dict(zip(unique_labels, range(len(unique_labels))))}")

                        scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2],
                                           c=numeric_colors, cmap='viridis', alpha=0.6)
                        # 创建自定义图例
                        for label, num in label_to_num.items():
                            ax.scatter([], [], [], c=[plt.cm.viridis(num / len(unique_labels))],
                                     label=label, alpha=0.6)
                        ax.legend(title=data.columns[3])
                    else:
                        scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2],
                                           c=color_data, cmap='viridis', alpha=0.6)

                    plt.colorbar(scatter, ax=ax, shrink=0.5)
                    ax.set_xlabel(data.columns[0])
                    ax.set_ylabel(data.columns[1])
                    ax.set_zlabel(data.columns[2])
                    ax.set_title(f"3D: {data.columns[0]} vs {data.columns[1]} vs {data.columns[2]}")
                else:
                    print("创建简单3D散点图")
                    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], alpha=0.6)
                    ax.set_xlabel(data.columns[0])
                    ax.set_ylabel(data.columns[1])
                    ax.set_zlabel(data.columns[2])
                    ax.set_title(f"3D: {data.columns[0]} vs {data.columns[1]} vs {data.columns[2]}")
            else:
                print("处理3D字典格式数据")
                x = data.get('x', [])
                y = data.get('y', [])
                z = data.get('z', [])
                colors = data.get('colors', None)
                labels = data.get('labels', {})

                if colors is not None:
                    scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', alpha=0.6)
                    plt.colorbar(scatter, ax=ax, shrink=0.5)
                else:
                    ax.scatter(x, y, z, alpha=0.6)

                ax.set_xlabel(labels.get('x', 'X'))
                ax.set_ylabel(labels.get('y', 'Y'))
                ax.set_zlabel(labels.get('z', 'Z'))
                ax.set_title(labels.get('title', '3D散点图'))

            plt.tight_layout()
            print("3D散点图创建成功")
            return fig

        except Exception as e:
            print(f"3D散点图生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _create_pie_chart(self, data):
        """创建饼图"""
        try:
            fig, ax = plt.subplots(figsize=(8, 8))

            if isinstance(data, pd.Series):
                values = data.values
                labels = data.index
            elif isinstance(data, dict):
                values = list(data.values())
                labels = list(data.keys())
            else:
                # 假设是数值列表
                values = data
                labels = [f"类别 {i+1}" for i in range(len(data))]

            # 使用更通用的颜色映射
            try:
                colors = plt.cm.Set3(np.linspace(0, 1, len(values)))
            except:
                colors = plt.cm.tab10(np.linspace(0, 1, len(values)))
            wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors,
                                             autopct='%1.1f%%', startangle=90)

            # 美化文本
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            ax.set_title('分布图', fontsize=16, pad=20)
            ax.axis('equal')

            plt.tight_layout()
            return fig

        except Exception as e:
            warnings.warn(f"饼图生成失败: {str(e)}")
            return None

    def _create_heatmap(self, data):
        """创建热力图"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            if isinstance(data, pd.DataFrame):
                # 尝试使用不同的seaborn热力图参数以适应不同版本
                try:
                    sns.heatmap(data, annot=True, cmap='RdYlBu_r', center=0,
                               square=True, linewidths=0.5, ax=ax, fmt='.2f')
                except TypeError:
                    # 如果某些参数不支持，使用基本版本
                    sns.heatmap(data, annot=True, cmap='RdYlBu_r', ax=ax, fmt='.2f')
                ax.set_title('热力图', fontsize=16, pad=20)
            else:
                # 创建示例热力图数据
                if isinstance(data, dict) and 'matrix' in data:
                    matrix = np.array(data['matrix'])
                    x_labels = data.get('x_labels', range(matrix.shape[1]))
                    y_labels = data.get('y_labels', range(matrix.shape[0]))

                    sns.heatmap(matrix, xticklabels=x_labels, yticklabels=y_labels,
                               annot=True, cmap='RdYlBu_r', center=0,
                               square=True, linewidths=0.5, ax=ax, fmt='.2f')
                else:
                    # 如果数据不合适，创建一个示例热力图
                    sample_data = np.random.rand(10, 10)
                    sns.heatmap(sample_data, annot=True, cmap='RdYlBu_r',
                               square=True, linewidths=0.5, ax=ax, fmt='.2f')

                ax.set_title('热力图', fontsize=16, pad=20)

            plt.tight_layout()
            return fig

        except Exception as e:
            warnings.warn(f"热力图生成失败: {str(e)}")
            return None

    def _create_feature_importance_plot(self, data):
        """创建特征重要性图"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            if isinstance(data, dict):
                features = data.get('features', [])
                importance = data.get('importance', [])
                title = data.get('title', '特征重要性')
            else:
                # 假设是DataFrame，第一列是特征名，第二列是重要性
                if isinstance(data, pd.DataFrame):
                    features = data.iloc[:, 0].values
                    importance = data.iloc[:, 1].values
                    title = '特征重要性'
                else:
                    return None

            # 排序
            indices = np.argsort(importance)[::-1]
            features = [features[i] for i in indices]
            importance = [importance[i] for i in indices]

            # 只显示前20个重要特征
            if len(features) > 20:
                features = features[:20]
                importance = importance[:20]

            # 创建水平条形图
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, importance, color='skyblue', edgecolor='navy', alpha=0.7)

            # 添加数值标签
            for i, (bar, imp) in enumerate(zip(bars, importance)):
                ax.text(bar.get_width() + max(importance) * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{imp:.3f}', ha='left', va='center', fontsize=10)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('重要性')
            ax.set_title(title, fontsize=16, pad=20)

            # 反转y轴，使重要性最高的在顶部
            ax.invert_yaxis()

            plt.tight_layout()
            return fig

        except Exception as e:
            warnings.warn(f"特征重要性图生成失败: {str(e)}")
            return None

    def _create_model_comparison_plot(self, data):
        """创建模型比较图"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            if isinstance(data, pd.DataFrame):
                # 提取准确率等指标
                if 'Accuracy' in data.index:
                    metrics_data = data.loc['Accuracy'].sort_values(ascending=True)
                elif 'R2' in data.index:
                    metrics_data = data.loc['R2'].sort_values(ascending=True)
                elif 'RMSE' in data.index:
                    metrics_data = data.loc['RMSE'].sort_values(ascending=False)  # RMSE越小越好
                else:
                    # 使用第一行数据
                    metrics_data = data.iloc[0].sort_values(ascending=True)

                models = metrics_data.index.tolist()
                values = metrics_data.values.tolist()

                # 创建条形图
                bars = ax.barh(models, values, color='lightcoral', edgecolor='darkred', alpha=0.7)

                # 添加数值标签
                for bar, value in zip(bars, values):
                    ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                           f'{value:.4f}', ha='left', va='center', fontsize=10)

                ax.set_xlabel('分数')
                ax.set_title('模型性能比较', fontsize=16, pad=20)

            else:
                # 创建示例比较图
                models = ['Model A', 'Model B', 'Model C', 'Model D', 'Model E']
                scores = [0.85, 0.92, 0.78, 0.89, 0.95]

                bars = ax.barh(models, scores, color='lightcoral', edgecolor='darkred', alpha=0.7)

                for bar, score in zip(bars, scores):
                    ax.text(bar.get_width() + max(scores) * 0.01, bar.get_y() + bar.get_height()/2,
                           f'{score:.3f}', ha='left', va='center', fontsize=10)

                ax.set_xlabel('分数')
                ax.set_title('模型性能比较', fontsize=16, pad=20)

            plt.tight_layout()
            return fig

        except Exception as e:
            warnings.warn(f"模型比较图生成失败: {str(e)}")
            return None

    def create_clustering_visualizations(self, data, cluster_labels, n_clusters):
        """创建聚类可视化图表"""
        visualizations = {}

        # 获取数值型列
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_columns) >= 2:
            # 散点图
            scatter_data = pd.DataFrame({
                'x': data.iloc[:, 0],
                'y': data.iloc[:, 1],
                'color': cluster_labels
            })
            visualizations['scatter'] = scatter_data

        if len(numeric_columns) >= 3:
            # 3D散点图
            scatter_3d_data = pd.DataFrame({
                'x': data.iloc[:, 0],
                'y': data.iloc[:, 1],
                'z': data.iloc[:, 2],
                'color': cluster_labels
            })
            visualizations['scatter_3d'] = scatter_3d_data

        # 聚类分布饼图
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        visualizations['pie'] = cluster_counts

        # 聚类中心热力图
        if len(numeric_columns) >= 2:
            data_with_clusters = data.copy()
            data_with_clusters['Cluster'] = cluster_labels
            cluster_centers = data_with_clusters.groupby('Cluster')[numeric_columns].mean()
            visualizations['heatmap'] = cluster_centers

        return visualizations

    def _create_metrics_radar_plot(self, data):
        """创建分类指标雷达图"""
        try:
            print(f"开始创建雷达图，数据: {data}")
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

            if isinstance(data, pd.DataFrame):
                metrics = data['Metric'].values
                values = data['Value'].values

                # 计算角度
                angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
                values = np.concatenate((values, [values[0]]))  # 闭合图形
                angles += angles[:1]  # 闭合图形

                # 绘制雷达图
                ax.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
                ax.fill(angles, values, alpha=0.25, color='blue')

                # 设置标签
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metrics)
                ax.set_ylim(0, 1)
                ax.set_title('分类性能指标雷达图', fontsize=16, pad=20)
                ax.grid(True)

            plt.tight_layout()
            print("雷达图创建成功")
            return fig

        except Exception as e:
            print(f"雷达图生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _create_metrics_comparison_plot(self, data):
        """创建回归指标对比图"""
        try:
            print(f"开始创建指标对比图，数据: {data}")
            fig, ax = plt.subplots(figsize=(10, 6))

            if isinstance(data, pd.DataFrame):
                metrics = data['Metric'].values
                values = data['Value'].values

                # 创建条形图
                bars = ax.bar(metrics, values, color='lightblue', edgecolor='navy', alpha=0.7)

                # 添加数值标签
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=10)

                ax.set_ylabel('指标值')
                ax.set_title('回归性能指标对比', fontsize=16, pad=20)
                plt.xticks(rotation=45)

            plt.tight_layout()
            print("指标对比图创建成功")
            return fig

        except Exception as e:
            print(f"指标对比图生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _create_confusion_matrix_plot(self, data):
        """创建混淆矩阵热力图"""
        try:
            print(f"开始创建混淆矩阵，数据: {data}")
            fig, ax = plt.subplots(figsize=(8, 6))

            if isinstance(data, dict) and 'matrix' in data:
                matrix = np.array(data['matrix'])
                labels = data.get('labels', ['Class 0', 'Class 1'])

                # 创建热力图
                sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                           xticklabels=labels, yticklabels=labels,
                           ax=ax, cbar=True)

                ax.set_xlabel('预测标签')
                ax.set_ylabel('真实标签')
                ax.set_title('混淆矩阵', fontsize=16, pad=20)

            plt.tight_layout()
            print("混淆矩阵创建成功")
            return fig

        except Exception as e:
            print(f"混淆矩阵生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _create_residuals_plot(self, data):
        """创建残差图"""
        try:
            print(f"开始创建残差图，数据类型: {type(data)}")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            if isinstance(data, dict):
                residuals = data.get('residuals', [])
                fitted = data.get('fitted', [])

                if len(residuals) > 0 and len(fitted) > 0:
                    # 残差vs拟合值图
                    ax1.scatter(fitted, residuals, alpha=0.6, color='blue')
                    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
                    ax1.set_xlabel('拟合值')
                    ax1.set_ylabel('残差')
                    ax1.set_title('残差 vs 拟合值')
                    ax1.grid(True, alpha=0.3)

                    # 残差直方图
                    ax2.hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
                    ax2.set_xlabel('残差')
                    ax2.set_ylabel('频数')
                    ax2.set_title('残差分布')
                    ax2.grid(True, alpha=0.3)

            fig.suptitle('残差分析', fontsize=16)
            plt.tight_layout()
            print("残差图创建成功")
            return fig

        except Exception as e:
            print(f"残差图生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _create_prediction_scatter_plot(self, data):
        """创建预测vs实际值散点图"""
        try:
            print(f"开始创建预测散点图，数据: {data}")
            fig, ax = plt.subplots(figsize=(10, 8))

            if isinstance(data, pd.DataFrame):
                if 'actual' in data.columns and 'predicted' in data.columns:
                    actual = data['actual'].values
                    predicted = data['predicted'].values

                    # 创建散点图
                    ax.scatter(actual, predicted, alpha=0.6, color='blue')

                    # 添加完美预测线
                    min_val = min(min(actual), min(predicted))
                    max_val = max(max(actual), max(predicted))
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

                    # 计算R²
                    correlation = np.corrcoef(actual, predicted)[0, 1]
                    r_squared = correlation ** 2

                    ax.set_xlabel('实际值')
                    ax.set_ylabel('预测值')
                    ax.set_title(f'预测 vs 实际值 (R² = {r_squared:.3f})', fontsize=16, pad=20)
                    ax.grid(True, alpha=0.3)
                    ax.axis('equal')

            plt.tight_layout()
            print("预测散点图创建成功")
            return fig

        except Exception as e:
            print(f"预测散点图生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _create_residual_histogram_plot(self, data):
        """创建残差直方图"""
        try:
            print(f"开始创建残差直方图，数据: {data}")
            fig, ax = plt.subplots(figsize=(10, 6))

            if isinstance(data, dict):
                residuals = data.get('residuals', [])
                bins = data.get('bins', 30)

                if len(residuals) > 0:
                    ax.hist(residuals, bins=bins, alpha=0.7, color='skyblue',
                           edgecolor='black', density=True)

                    # 添加正态分布曲线
                    import numpy as np
                    mu, sigma = np.mean(residuals), np.std(residuals)
                    x = np.linspace(min(residuals), max(residuals), 100)
                    normal_curve = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                    ax.plot(x, normal_curve, 'r-', linewidth=2, label='正态分布')

                    ax.set_xlabel('残差')
                    ax.set_ylabel('密度')
                    ax.set_title('残差分布直方图', fontsize=16, pad=20)
                    ax.legend()
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()
            print("残差直方图创建成功")
            return fig

        except Exception as e:
            print(f"残差直方图生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _create_qq_plot(self, data):
        """创建Q-Q图"""
        try:
            print(f"开始创建Q-Q图，数据: {data}")
            fig, ax = plt.subplots(figsize=(8, 8))

            if isinstance(data, dict):
                theoretical = data.get('theoretical', [])
                sample = data.get('sample', [])

                if len(theoretical) > 0 and len(sample) > 0:
                    ax.scatter(theoretical, sample, alpha=0.6, color='blue')

                    # 添加参考线
                    min_val = min(min(theoretical), min(sample))
                    max_val = max(max(theoretical), max(sample))
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

                    ax.set_xlabel('理论分位数')
                    ax.set_ylabel('样本分位数')
                    ax.set_title('Q-Q图 (残差正态性检验)', fontsize=16, pad=20)
                    ax.grid(True, alpha=0.3)
                    ax.axis('equal')

            plt.tight_layout()
            print("Q-Q图创建成功")
            return fig

        except Exception as e:
            print(f"Q-Q图生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def create_text_visualizations(self, text_data, labels=None, title="文本分析"):
        """创建文本分析可视化"""
        visualizations = {}

        try:
            # 生成词云图
            all_text = ' '.join(text_data.dropna().astype(str))

            if all_text.strip():
                visualizations['wordcloud'] = all_text

                # 如果有标签，创建不同类别的词云
                if labels is not None and len(labels) == len(text_data):
                    unique_labels = pd.Series(labels).unique()
                    for label in unique_labels[:3]:  # 最多显示3个类别
                        label_text = ' '.join(text_data[labels == label].dropna().astype(str))
                        if label_text.strip():
                            visualizations[f'wordcloud_{label}'] = label_text

        except Exception as e:
            warnings.warn(f"文本可视化生成失败: {str(e)}")

        return visualizations

    def _create_metrics_radar_plot(self, data):
        """创建分类指标雷达图"""
        try:
            print(f"开始创建雷达图，数据: {data}")
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

            if isinstance(data, pd.DataFrame):
                metrics = data['Metric'].values
                values = data['Value'].values

                # 计算角度
                angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
                values = np.concatenate((values, [values[0]]))  # 闭合图形
                angles += angles[:1]  # 闭合图形

                # 绘制雷达图
                ax.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
                ax.fill(angles, values, alpha=0.25, color='blue')

                # 设置标签
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metrics)
                ax.set_ylim(0, 1)
                ax.set_title('分类性能指标雷达图', fontsize=16, pad=20)
                ax.grid(True)

            plt.tight_layout()
            print("雷达图创建成功")
            return fig

        except Exception as e:
            print(f"雷达图生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _create_metrics_comparison_plot(self, data):
        """创建回归指标对比图"""
        try:
            print(f"开始创建指标对比图，数据: {data}")
            fig, ax = plt.subplots(figsize=(10, 6))

            if isinstance(data, pd.DataFrame):
                metrics = data['Metric'].values
                values = data['Value'].values

                # 创建条形图
                bars = ax.bar(metrics, values, color='lightblue', edgecolor='navy', alpha=0.7)

                # 添加数值标签
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=10)

                ax.set_ylabel('指标值')
                ax.set_title('回归性能指标对比', fontsize=16, pad=20)
                plt.xticks(rotation=45)

            plt.tight_layout()
            print("指标对比图创建成功")
            return fig

        except Exception as e:
            print(f"指标对比图生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _create_confusion_matrix_plot(self, data):
        """创建混淆矩阵热力图"""
        try:
            print(f"开始创建混淆矩阵，数据: {data}")
            fig, ax = plt.subplots(figsize=(8, 6))

            if isinstance(data, dict) and 'matrix' in data:
                matrix = np.array(data['matrix'])
                labels = data.get('labels', ['Class 0', 'Class 1'])

                # 创建热力图
                sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                           xticklabels=labels, yticklabels=labels,
                           ax=ax, cbar=True)

                ax.set_xlabel('预测标签')
                ax.set_ylabel('真实标签')
                ax.set_title('混淆矩阵', fontsize=16, pad=20)

            plt.tight_layout()
            print("混淆矩阵创建成功")
            return fig

        except Exception as e:
            print(f"混淆矩阵生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _create_residuals_plot(self, data):
        """创建残差图"""
        try:
            print(f"开始创建残差图，数据类型: {type(data)}")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            if isinstance(data, dict):
                residuals = data.get('residuals', [])
                fitted = data.get('fitted', [])

                if len(residuals) > 0 and len(fitted) > 0:
                    # 残差vs拟合值图
                    ax1.scatter(fitted, residuals, alpha=0.6, color='blue')
                    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
                    ax1.set_xlabel('拟合值')
                    ax1.set_ylabel('残差')
                    ax1.set_title('残差 vs 拟合值')
                    ax1.grid(True, alpha=0.3)

                    # 残差直方图
                    ax2.hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
                    ax2.set_xlabel('残差')
                    ax2.set_ylabel('频数')
                    ax2.set_title('残差分布')
                    ax2.grid(True, alpha=0.3)

            fig.suptitle('残差分析', fontsize=16)
            plt.tight_layout()
            print("残差图创建成功")
            return fig

        except Exception as e:
            print(f"残差图生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _create_prediction_scatter_plot(self, data):
        """创建预测vs实际值散点图"""
        try:
            print(f"开始创建预测散点图，数据: {data}")
            fig, ax = plt.subplots(figsize=(10, 8))

            if isinstance(data, pd.DataFrame):
                if 'actual' in data.columns and 'predicted' in data.columns:
                    actual = data['actual'].values
                    predicted = data['predicted'].values

                    # 创建散点图
                    ax.scatter(actual, predicted, alpha=0.6, color='blue')

                    # 添加完美预测线
                    min_val = min(min(actual), min(predicted))
                    max_val = max(max(actual), max(predicted))
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

                    # 计算R²
                    correlation = np.corrcoef(actual, predicted)[0, 1]
                    r_squared = correlation ** 2

                    ax.set_xlabel('实际值')
                    ax.set_ylabel('预测值')
                    ax.set_title(f'预测 vs 实际值 (R² = {r_squared:.3f})', fontsize=16, pad=20)
                    ax.grid(True, alpha=0.3)
                    ax.axis('equal')

            plt.tight_layout()
            print("预测散点图创建成功")
            return fig

        except Exception as e:
            print(f"预测散点图生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _create_residual_histogram_plot(self, data):
        """创建残差直方图"""
        try:
            print(f"开始创建残差直方图，数据: {data}")
            fig, ax = plt.subplots(figsize=(10, 6))

            if isinstance(data, dict):
                residuals = data.get('residuals', [])
                bins = data.get('bins', 30)

                if len(residuals) > 0:
                    ax.hist(residuals, bins=bins, alpha=0.7, color='skyblue',
                           edgecolor='black', density=True)

                    # 添加正态分布曲线
                    import numpy as np
                    mu, sigma = np.mean(residuals), np.std(residuals)
                    x = np.linspace(min(residuals), max(residuals), 100)
                    normal_curve = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                    ax.plot(x, normal_curve, 'r-', linewidth=2, label='正态分布')

                    ax.set_xlabel('残差')
                    ax.set_ylabel('密度')
                    ax.set_title('残差分布直方图', fontsize=16, pad=20)
                    ax.legend()
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()
            print("残差直方图创建成功")
            return fig

        except Exception as e:
            print(f"残差直方图生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _create_qq_plot(self, data):
        """创建Q-Q图"""
        try:
            print(f"开始创建Q-Q图，数据: {data}")
            fig, ax = plt.subplots(figsize=(8, 8))

            if isinstance(data, dict):
                theoretical = data.get('theoretical', [])
                sample = data.get('sample', [])

                if len(theoretical) > 0 and len(sample) > 0:
                    ax.scatter(theoretical, sample, alpha=0.6, color='blue')

                    # 添加参考线
                    min_val = min(min(theoretical), min(sample))
                    max_val = max(max(theoretical), max(sample))
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

                    ax.set_xlabel('理论分位数')
                    ax.set_ylabel('样本分位数')
                    ax.set_title('Q-Q图 (残差正态性检验)', fontsize=16, pad=20)
                    ax.grid(True, alpha=0.3)
                    ax.axis('equal')

            plt.tight_layout()
            print("Q-Q图创建成功")
            return fig

        except Exception as e:
            print(f"Q-Q图生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None