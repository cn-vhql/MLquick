#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLquick GUI版本 - 基于Tkinter的机器学习零代码桌面应用
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.scrolledtext as scrolledtext
import sys
import os
import pandas as pd
import threading
import queue
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.data_manager import DataManager
from components.ml_engine import MLEngine
from components.visualizer import Visualizer
from components.model_manager import ModelManager
from utils.ui_utils import create_centered_window, show_loading, hide_loading
from utils.config import APP_NAME, APP_VERSION

class MLquickGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()
        self.setup_components()
        self.setup_ui()

        # 数据和模型状态
        self.current_data = None
        self.current_model = None
        self.model_name = None

        # 线程间通信队列
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # 启动结果处理循环
        self.process_results()

    def setup_window(self):
        """设置主窗口"""
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("1200x800")

        # 设置窗口图标和样式
        try:
            self.root.iconbitmap("assets/icon.ico")
        except:
            pass

        # 配置样式
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # 自定义样式
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        self.style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'))
        self.style.configure('Success.TLabel', foreground='green')
        self.style.configure('Error.TLabel', foreground='red')

    def setup_components(self):
        """初始化组件"""
        self.data_manager = DataManager()
        self.ml_engine = MLEngine()
        self.visualizer = Visualizer(self.root)
        self.model_manager = ModelManager()

    def setup_ui(self):
        """设置用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # 标题
        title_label = ttk.Label(main_frame, text=APP_NAME, style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # 左侧控制面板
        self.create_control_panel(main_frame)

        # 右侧结果面板
        self.create_result_panel(main_frame)

        # 底部状态栏
        self.create_status_bar(main_frame)

    def create_control_panel(self, parent):
        """创建左侧控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # 数据上传区域
        data_frame = ttk.LabelFrame(control_frame, text="数据管理", padding="5")
        data_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(data_frame, text="上传数据集",
                  command=self.upload_data).pack(fill=tk.X, pady=2)

        self.data_info_label = ttk.Label(data_frame, text="未加载数据")
        self.data_info_label.pack(fill=tk.X, pady=2)

        # 模型配置区域
        config_frame = ttk.LabelFrame(control_frame, text="模型配置", padding="5")
        config_frame.pack(fill=tk.X, pady=(0, 10))

        # 任务类型选择
        ttk.Label(config_frame, text="任务类型:").pack(anchor=tk.W)
        self.task_type = tk.StringVar(value="分类")
        task_combo = ttk.Combobox(config_frame, textvariable=self.task_type,
                                  values=["分类", "回归", "聚类"], state="readonly")
        task_combo.pack(fill=tk.X, pady=(0, 10))
        task_combo.bind("<<ComboboxSelected>>", self.on_task_type_change)

        # 目标变量选择（分类和回归任务）
        self.target_frame = ttk.Frame(config_frame)
        ttk.Label(self.target_frame, text="目标变量:").pack(anchor=tk.W)
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(self.target_frame, textvariable=self.target_var,
                                        state="readonly")
        self.target_combo.pack(fill=tk.X, pady=(0, 10))

        # 聚类数量配置（聚类任务）
        self.cluster_frame = ttk.Frame(config_frame)
        ttk.Label(self.cluster_frame, text="聚类数量:").pack(anchor=tk.W)
        self.n_clusters = tk.IntVar(value=3)
        ttk.Spinbox(self.cluster_frame, from_=2, to=20,
                   textvariable=self.n_clusters).pack(fill=tk.X, pady=(0, 10))

        # 训练集比例（分类和回归任务）
        self.train_size_frame = ttk.Frame(config_frame)
        ttk.Label(self.train_size_frame, text="训练集比例:").pack(anchor=tk.W)
        self.train_size = tk.DoubleVar(value=0.7)
        ttk.Scale(self.train_size_frame, from_=0.1, to=0.9,
                 variable=self.train_size, orient=tk.HORIZONTAL).pack(fill=tk.X)
        self.train_size_label = ttk.Label(self.train_size_frame, text="70%")
        self.train_size_label.pack(anchor=tk.W)
        self.train_size.trace('w', self.update_train_size_label)

        # 文本处理选项
        self.text_frame = ttk.LabelFrame(config_frame, text="文本处理选项", padding="5")
        self.enable_text_processing = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.text_frame, text="启用文本预处理",
                       variable=self.enable_text_processing).pack(anchor=tk.W)

        # 训练按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)

        self.train_button = ttk.Button(button_frame, text="开始训练模型",
                                      command=self.train_model, style="Accent.TButton")
        self.train_button.pack(fill=tk.X)

        # 模型管理
        model_frame = ttk.LabelFrame(control_frame, text="模型管理", padding="5")
        model_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(model_frame, text="保存模型",
                  command=self.save_model).pack(fill=tk.X, pady=2)
        ttk.Button(model_frame, text="加载模型",
                  command=self.load_model).pack(fill=tk.X, pady=2)
        ttk.Button(model_frame, text="预测",
                  command=self.predict).pack(fill=tk.X, pady=2)

        # 初始状态设置
        self.on_task_type_change()

    def create_result_panel(self, parent):
        """创建右侧结果面板"""
        result_frame = ttk.LabelFrame(parent, text="结果展示", padding="10")
        result_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 创建Notebook用于多标签页
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 数据预览标签页
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="数据预览")

        # 模型结果标签页
        self.model_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.model_frame, text="模型结果")

        # 可视化标签页
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="可视化")

        # 预测结果标签页
        self.prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_frame, text="预测结果")

        # 初始化各标签页内容
        self.setup_data_tab()
        self.setup_model_tab()
        self.setup_visualization_tab()
        self.setup_prediction_tab()

        # 调试信息：确认可视化框架已设置
        print(f"UI初始化完成，viz_frame: {getattr(self, 'viz_frame', 'Not set')}")

    def setup_data_tab(self):
        """设置数据预览标签页"""
        # 数据表格
        self.data_tree = ttk.Treeview(self.data_frame)

        # 滚动条
        data_scroll_y = ttk.Scrollbar(self.data_frame, orient=tk.VERTICAL,
                                     command=self.data_tree.yview)
        data_scroll_x = ttk.Scrollbar(self.data_frame, orient=tk.HORIZONTAL,
                                     command=self.data_tree.xview)

        self.data_tree.configure(yscrollcommand=data_scroll_y.set,
                                xscrollcommand=data_scroll_x.set)

        # 布局
        self.data_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        data_scroll_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        data_scroll_x.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.data_frame.columnconfigure(0, weight=1)
        self.data_frame.rowconfigure(0, weight=1)

    def setup_model_tab(self):
        """设置模型结果标签页"""
        # 使用ScrolledText显示模型结果
        self.model_text = scrolledtext.ScrolledText(self.model_frame,
                                                    wrap=tk.WORD, height=20)
        self.model_text.pack(fill=tk.BOTH, expand=True)

    def setup_visualization_tab(self):
        """设置可视化标签页"""
        # 配置可视化框架的网格权重
        self.viz_frame.columnconfigure(0, weight=1)
        self.viz_frame.columnconfigure(1, weight=1)
        self.viz_frame.rowconfigure(0, weight=1)
        self.viz_frame.rowconfigure(1, weight=1)

        # 添加一个提示标签
        placeholder_label = ttk.Label(self.viz_frame, text="可视化图表将在训练完成后显示在这里",
                                     font=('Arial', 12))
        placeholder_label.grid(row=0, column=0, columnspan=2, pady=20)

    def setup_prediction_tab(self):
        """设置预测结果标签页"""
        # 预测结果表格
        self.prediction_tree = ttk.Treeview(self.prediction_frame)

        # 滚动条
        pred_scroll_y = ttk.Scrollbar(self.prediction_frame, orient=tk.VERTICAL,
                                     command=self.prediction_tree.yview)
        pred_scroll_x = ttk.Scrollbar(self.prediction_frame, orient=tk.HORIZONTAL,
                                     command=self.prediction_tree.xview)

        self.prediction_tree.configure(yscrollcommand=pred_scroll_y.set,
                                     xscrollcommand=pred_scroll_x.set)

        # 布局
        self.prediction_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        pred_scroll_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        pred_scroll_x.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.prediction_frame.columnconfigure(0, weight=1)
        self.prediction_frame.rowconfigure(0, weight=1)

    def create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        self.status_label = ttk.Label(status_frame, text="就绪")
        self.status_label.pack(side=tk.LEFT)

        # 进度条
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, padx=(10, 0))

    def update_status(self, message):
        """更新状态栏"""
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def update_train_size_label(self, *args):
        """更新训练集比例标签"""
        percentage = int(self.train_size.get() * 100)
        self.train_size_label.config(text=f"{percentage}%")

    def on_task_type_change(self, event=None):
        """任务类型改变时的处理"""
        task_type = self.task_type.get()

        if task_type == "聚类":
            # 聚类任务显示聚类数量，隐藏目标变量和训练集比例
            self.target_frame.pack_forget()
            self.train_size_frame.pack_forget()
            self.cluster_frame.pack(fill=tk.X, pady=(0, 10))
        else:
            # 分类和回归任务显示目标变量和训练集比例，隐藏聚类数量
            self.cluster_frame.pack_forget()
            self.target_frame.pack(fill=tk.X, pady=(0, 10))
            self.train_size_frame.pack(fill=tk.X, pady=(0, 10))

    def upload_data(self):
        """上传数据文件"""
        file_path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"),
                      ("All files", "*.*")]
        )

        if file_path:
            try:
                self.update_status("正在加载数据...")
                self.current_data = self.data_manager.load_data(file_path)

                # 更新数据信息
                shape = self.current_data.shape
                self.data_info_label.config(
                    text=f"已加载: {shape[0]} 行 × {shape[1]} 列"
                )

                # 更新目标变量选项
                columns = list(self.current_data.columns)
                self.target_combo['values'] = columns
                if columns:
                    self.target_combo.set(columns[-1])  # 默认选择最后一列

                # 显示数据预览
                self.display_data_preview()

                # 检查文本列
                text_columns = self.data_manager.get_text_columns(self.current_data)
                if text_columns:
                    self.text_frame.pack(fill=tk.X, pady=(0, 10))
                    self.update_status(f"数据加载完成，发现 {len(text_columns)} 个文本列")
                else:
                    self.text_frame.pack_forget()
                    self.update_status("数据加载完成")

            except Exception as e:
                messagebox.showerror("错误", f"数据加载失败: {str(e)}")
                self.update_status("数据加载失败")

    def display_data_preview(self):
        """显示数据预览"""
        if self.current_data is not None:
            # 清空现有数据
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)

            # 设置列
            columns = list(self.current_data.columns)
            self.data_tree['columns'] = columns
            self.data_tree['show'] = 'headings'

            # 设置列标题
            for col in columns:
                self.data_tree.heading(col, text=col)
                self.data_tree.column(col, width=100)

            # 添加数据（只显示前100行）
            sample_data = self.current_data.head(100)
            for _, row in sample_data.iterrows():
                values = [str(val) for val in row]
                self.data_tree.insert('', tk.END, values=values)

    def train_model(self):
        """训练模型"""
        if self.current_data is None:
            messagebox.showwarning("警告", "请先上传数据")
            return

        task_type = self.task_type.get()

        # 验证配置
        if task_type != "聚类":
            if not self.target_var.get():
                messagebox.showwarning("警告", "请选择目标变量")
                return
        else:
            if self.n_clusters.get() < 2:
                messagebox.showwarning("警告", "聚类数量必须大于1")
                return

        # 显示进度
        self.progress.start()
        self.train_button.config(state='disabled')
        self.update_status("正在训练模型...")

        # 在后台线程中训练模型
        threading.Thread(target=self._train_model_thread, daemon=True).start()

    def _train_model_thread(self):
        """后台训练模型的线程"""
        try:
            task_type = self.task_type.get()

            # 转换任务类型为英文
            task_type_map = {
                "分类": "classification",
                "回归": "regression",
                "聚类": "clustering"
            }
            english_task_type = task_type_map.get(task_type, task_type)

            # 准备参数
            params = {
                'data': self.current_data,
                'task_type': english_task_type,
                'enable_text_processing': self.enable_text_processing.get()
            }

            if task_type == "聚类":
                params['n_clusters'] = self.n_clusters.get()
            else:
                params['target_variable'] = self.target_var.get()
                params['train_size'] = self.train_size.get()

            # 训练模型
            result = self.ml_engine.train_model(**params)

            # 将结果放入队列
            self.result_queue.put(('success', result))

        except Exception as e:
            self.result_queue.put(('error', str(e)))

    def process_results(self):
        """处理后台任务结果"""
        try:
            while not self.result_queue.empty():
                status, result = self.result_queue.get_nowait()

                if status == 'success':
                    self.on_training_complete(result)
                else:
                    messagebox.showerror("错误", f"模型训练失败: {result}")
                    self.update_status("模型训练失败")

                # 停止进度条，恢复按钮
                self.progress.stop()
                self.train_button.config(state='normal')

        except queue.Empty:
            pass

        # 继续检查结果
        self.root.after(100, self.process_results)

    def on_training_complete(self, result):
        """训练完成后的处理"""
        self.current_model = result['model']
        self.model_name = result['model_name']

        # 显示模型结果
        self.display_model_results(result)

        # 显示可视化
        if 'visualizations' in result:
            print(f"可视化数据: {result['visualizations'].keys()}")
            try:
                self.visualizer.display_visualizations(result['visualizations'])
            except Exception as e:
                print(f"可视化显示失败: {e}")
                # 简单可视化显示
                self._simple_visualization_display(result['visualizations'])
        else:
            print("无可视化数据")

        self.update_status(f"模型训练完成: {self.model_name}")
        messagebox.showinfo("成功", f"模型训练完成!\n模型名称: {self.model_name}")

    def display_model_results(self, result):
        """显示模型结果"""
        self.model_text.delete(1.0, tk.END)

        # 显示基本信息
        info_text = f"模型名称: {result['model_name']}\n"
        info_text += f"任务类型: {result['task_type']}\n"
        info_text += f"训练时间: {result['training_time']}\n"
        info_text += f"数据集大小: {result['data_shape']}\n\n"

        # 聚类任务特殊显示
        if result['task_type'] == 'clustering':
            info_text += "聚类分析结果:\n"
            if 'n_clusters' in result:
                info_text += f"  聚类数量: {result['n_clusters']}\n"
            if 'cluster_stats' in result:
                info_text += "\n聚类统计信息:\n"
                info_text += str(result['cluster_stats'])

        # 显示性能指标
        if 'metrics' in result:
            if result['task_type'] != 'clustering':
                info_text += "\n模型性能指标:\n"
            else:
                info_text += "\n聚类指标:\n"
            for metric, value in result['metrics'].items():
                if isinstance(value, (int, float)):
                    info_text += f"  {metric}: {value:.4f}\n"
                else:
                    info_text += f"  {metric}: {value}\n"

        # 显示模型比较结果（如果有）
        if 'comparison' in result:
            info_text += "\n模型比较结果:\n"
            info_text += str(result['comparison'])

        # 显示文本处理信息（如果有）
        if 'text_processing_info' in result and result['text_processing_info']:
            info_text += "\n文本处理信息:\n"
            text_info = result['text_processing_info']
            info_text += f"  处理的列数: {text_info.get('processed_columns_count', 0)}\n"
            if text_info.get('processed_columns'):
                info_text += f"  处理的列: {', '.join(text_info['processed_columns'])}\n"

        self.model_text.insert(tk.END, info_text)

    def save_model(self):
        """保存模型"""
        if self.current_model is None:
            messagebox.showwarning("警告", "没有可保存的模型")
            return

        try:
            self.model_manager.save_model(self.current_model, self.model_name)
            messagebox.showinfo("成功", f"模型已保存: {self.model_name}")
            self.update_status(f"模型已保存: {self.model_name}")
        except Exception as e:
            messagebox.showerror("错误", f"模型保存失败: {str(e)}")

    def load_model(self):
        """加载模型"""
        try:
            models = self.model_manager.get_available_models()
            if not models:
                messagebox.showinfo("信息", "没有可用的模型")
                return

            # 创建模型选择对话框
            dialog = tk.Toplevel(self.root)
            dialog.title("选择模型")
            dialog.geometry("400x300")
            dialog.transient(self.root)
            dialog.grab_set()

            ttk.Label(dialog, text="选择要加载的模型:").pack(pady=10)

            model_var = tk.StringVar()
            model_listbox = tk.Listbox(dialog, listvariable=model_var)
            model_listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

            for model in models:
                model_listbox.insert(tk.END, model)

            def on_select():
                selection = model_listbox.curselection()
                if selection:
                    selected_model = models[selection[0]]
                    try:
                        self.current_model = self.model_manager.load_model(selected_model)
                        self.model_name = selected_model
                        messagebox.showinfo("成功", f"模型已加载: {selected_model}")
                        self.update_status(f"模型已加载: {selected_model}")
                        dialog.destroy()
                    except Exception as e:
                        messagebox.showerror("错误", f"模型加载失败: {str(e)}")

            ttk.Button(dialog, text="加载", command=on_select).pack(pady=10)

        except Exception as e:
            messagebox.showerror("错误", f"获取模型列表失败: {str(e)}")

    def predict(self):
        """进行预测"""
        if self.current_model is None:
            messagebox.showwarning("警告", "请先训练或加载模型")
            return

        # 选择预测数据文件
        file_path = filedialog.askopenfilename(
            title="选择预测数据文件",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        )

        if file_path:
            try:
                self.update_status("正在加载预测数据...")
                prediction_data = self.data_manager.load_data(file_path)

                self.update_status("正在进行预测...")
                predictions = self.ml_engine.predict(self.current_model, prediction_data)

                # 显示预测结果
                self.display_predictions(predictions, prediction_data)

                self.update_status("预测完成")

            except Exception as e:
                messagebox.showerror("错误", f"预测失败: {str(e)}")
                self.update_status("预测失败")

    def display_predictions(self, predictions, original_data):
        """显示预测结果"""
        # 清空现有数据
        for item in self.prediction_tree.get_children():
            self.prediction_tree.delete(item)

        # 准备显示数据
        if isinstance(predictions, pd.DataFrame):
            display_data = predictions
        else:
            # 如果预测结果是Series，添加到原始数据
            display_data = original_data.copy()
            display_data['Prediction'] = predictions

        # 设置列
        columns = list(display_data.columns)
        self.prediction_tree['columns'] = columns
        self.prediction_tree['show'] = 'headings'

        # 设置列标题
        for col in columns:
            self.prediction_tree.heading(col, text=col)
            self.prediction_tree.column(col, width=100)

        # 添加数据
        for _, row in display_data.iterrows():
            values = [str(val) for val in row]
            self.prediction_tree.insert('', tk.END, values=values)

        # 切换到预测结果标签页
        self.notebook.select(3)

    def _simple_visualization_display(self, visualizations):
        """简单的可视化显示"""
        try:
            print(f"开始简单可视化显示，可视化数据: {visualizations}")
            print(f"self.viz_frame 属性存在: {hasattr(self, 'viz_frame')}")

            # 直接使用预定义的可视化框架
            viz_frame = getattr(self, 'viz_frame', None)
            print(f"可视化框架: {viz_frame}")
            print(f"可视化框架类型: {type(viz_frame)}")

            # 确保可视化框架存在
            if viz_frame is None:
                print("可视化框架为None，尝试通过notebook查找")
                # 备用方法：通过notebook查找可视化框架
                for i in range(self.notebook.index('end')):
                    tab_text = self.notebook.tab(i, 'text')
                    print(f"标签页 {i}: {tab_text}")
                    if '可视' in tab_text:
                        viz_frame = self.notebook.winfo_children()[i]
                        print(f"通过notebook找到可视化框架: {viz_frame}")
                        break

                if viz_frame is None:
                    print("未找到可视化框架")
                    return

            # 检查可视化框架是否有效
            if not hasattr(viz_frame, 'winfo_children'):
                print("可视化框架不是有效的Tkinter组件")
                return

            print(f"可视化框架的子组件数量: {len(viz_frame.winfo_children())}")

            # 清空现有内容
            for widget in viz_frame.winfo_children():
                widget.destroy()

            # 创建简单的图表显示
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False

            charts_for_display = {}

            for viz_name, viz_data in visualizations.items():
                print(f"处理可视化: {viz_name}, 数据类型: {type(viz_data)}")

                try:
                    if viz_name == 'pie':
                        # 创建饼图
                        fig, ax = plt.subplots(figsize=(6, 4))
                        if isinstance(viz_data, (pd.Series, dict)):
                            if isinstance(viz_data, pd.Series):
                                values = viz_data.values
                                labels = viz_data.index
                            else:
                                values = list(viz_data.values())
                                labels = list(viz_data.keys())

                            ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                            ax.set_title('聚类分布')
                        plt.tight_layout()

                        # 创建画布
                        canvas = FigureCanvasTkAgg(fig, viz_frame)
                        canvas.draw()
                        charts_for_display['pie'] = canvas

                    elif viz_name in ['scatter', 'scatter_3d']:
                        # 创建散点图
                        fig, ax = plt.subplots(figsize=(6, 4))
                        if isinstance(viz_data, pd.DataFrame):
                            if len(viz_data.columns) >= 3:
                                scatter = ax.scatter(viz_data.iloc[:, 0], viz_data.iloc[:, 1],
                                                  c=viz_data.iloc[:, 2], cmap='viridis', alpha=0.6)
                                plt.colorbar(scatter, ax=ax)
                            else:
                                ax.scatter(viz_data.iloc[:, 0], viz_data.iloc[:, 1], alpha=0.6)

                            ax.set_xlabel(viz_data.columns[0])
                            ax.set_ylabel(viz_data.columns[1])
                            ax.set_title('聚类散点图')
                        plt.tight_layout()

                        # 创建画布
                        canvas = FigureCanvasTkAgg(fig, viz_frame)
                        canvas.draw()
                        charts_for_display['scatter'] = canvas

                    elif viz_name == 'heatmap':
                        # 创建热力图
                        fig, ax = plt.subplots(figsize=(8, 6))
                        if isinstance(viz_data, pd.DataFrame):
                            import seaborn as sns
                            sns.heatmap(viz_data, annot=True, cmap='coolwarm', ax=ax)
                            ax.set_title('聚类中心热力图')
                        plt.tight_layout()

                        # 创建画布
                        canvas = FigureCanvasTkAgg(fig, viz_frame)
                        canvas.draw()
                        charts_for_display['heatmap'] = canvas

                except Exception as e:
                    print(f"创建 {viz_name} 图表失败: {e}")
                    continue

            # 使用grid布局显示图表
            for i, (chart_type, canvas) in enumerate(charts_for_display.items()):
                canvas.get_tk_widget().grid(row=i//2, column=i%2, padx=10, pady=10)

            plt.close('all')  # 关闭所有图形
            print(f"成功显示 {len(charts_for_display)} 个图表")

            # 切换到可视化标签页
            for i in range(self.notebook.index('end')):
                if '可视' in self.notebook.tab(i, 'text'):
                    self.notebook.select(i)
                    print(f"已切换到可视化标签页: {i}")
                    break

        except Exception as e:
            print(f"简单可视化显示失败: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """运行应用"""
        self.root.mainloop()

if __name__ == "__main__":
    app = MLquickGUI()
    app.run()