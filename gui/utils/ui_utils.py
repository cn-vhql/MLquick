#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI工具函数 - 提供界面相关的辅助功能
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
from typing import Optional, Callable, Any


def create_centered_window(title: str, width: int = 600, height: int = 400) -> tk.Toplevel:
    """
    创建居中的子窗口

    Args:
        title: 窗口标题
        width: 窗口宽度
        height: 窗口高度

    Returns:
        创建的Toplevel窗口
    """
    window = tk.Toplevel()
    window.title(title)
    window.geometry(f"{width}x{height}")
    window.resizable(False, False)

    # 居中显示
    window.update_idletasks()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")

    return window


def show_loading(parent: tk.Widget, message: str = "正在处理...") -> tuple:
    """
    显示加载指示器

    Args:
        parent: 父组件
        message: 加载消息

    Returns:
        (加载窗口, 进度条) 的元组
    """
    # 创建加载窗口
    loading_window = tk.Toplevel(parent)
    loading_window.title("处理中")
    loading_window.geometry("300x100")
    loading_window.resizable(False, False)

    # 居中显示
    loading_window.update_idletasks()
    x = (loading_window.winfo_screenwidth() // 2) - 150
    y = (loading_window.winfo_screenheight() // 2) - 50
    loading_window.geometry(f"300x100+{x}+{y}")

    # 设置为模态窗口
    loading_window.transient(parent)
    loading_window.grab_set()

    # 创建内容
    frame = ttk.Frame(loading_window, padding="20")
    frame.pack(fill=tk.BOTH, expand=True)

    # 消息标签
    label = ttk.Label(frame, text=message)
    label.pack(pady=(0, 10))

    # 进度条
    progress = ttk.Progressbar(frame, mode='indeterminate', length=200)
    progress.pack()

    # 启动进度条
    progress.start(10)

    # 禁用关闭按钮
    loading_window.protocol("WM_DELETE_WINDOW", lambda: None)

    return loading_window, progress


def hide_loading(loading_window: tk.Toplevel, progress: ttk.Progressbar):
    """
    隐藏加载指示器

    Args:
        loading_window: 加载窗口
        progress: 进度条
    """
    try:
        progress.stop()
        loading_window.grab_release()
        loading_window.destroy()
    except:
        pass


def show_message(parent: tk.Widget, title: str, message: str,
                message_type: str = "info") -> None:
    """
    显示消息对话框

    Args:
        parent: 父组件
        title: 对话框标题
        message: 消息内容
        message_type: 消息类型 ("info", "warning", "error")
    """
    if message_type == "info":
        messagebox.showinfo(title, message, parent=parent)
    elif message_type == "warning":
        messagebox.showwarning(title, message, parent=parent)
    elif message_type == "error":
        messagebox.showerror(title, message, parent=parent)
    else:
        messagebox.showinfo(title, message, parent=parent)


def ask_confirmation(parent: tk.Widget, title: str, message: str) -> bool:
    """
    显示确认对话框

    Args:
        parent: 父组件
        title: 对话框标题
        message: 消息内容

    Returns:
        用户是否确认
    """
    return messagebox.askyesno(title, message, parent=parent)


def create_scrollable_frame(parent: tk.Widget, width: int = 400, height: int = 300) -> tuple:
    """
    创建可滚动的框架

    Args:
        parent: 父组件
        width: 宽度
        height: 高度

    Returns:
        (canvas, scrollbar, scrollable_frame) 的元组
    """
    # 创建主框架
    main_frame = ttk.Frame(parent)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 创建Canvas和Scrollbar
    canvas = tk.Canvas(main_frame, width=width, height=height)
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    # 配置滚动
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # 布局
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # 绑定鼠标滚轮事件
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    canvas.bind("<MouseWheel>", _on_mousewheel)

    return canvas, scrollbar, scrollable_frame


def run_with_loading(parent: tk.Widget, func: Callable, message: str = "正在处理...",
                    success_message: str = "操作完成",
                    error_message: str = "操作失败") -> Any:
    """
    在显示加载指示器的情况下运行函数

    Args:
        parent: 父组件
        func: 要执行的函数
        message: 加载消息
        success_message: 成功消息
        error_message: 失败消息

    Returns:
        函数执行结果
    """
    loading_window, progress = show_loading(parent, message)

    def run_function():
        try:
            result = func()
            # 使用after方法在主线程中隐藏加载窗口
            parent.after(0, lambda: hide_loading(loading_window, progress))
            parent.after(0, lambda: show_message(parent, "成功", success_message, "info"))
            return result
        except Exception as e:
            parent.after(0, lambda: hide_loading(loading_window, progress))
            parent.after(0, lambda: show_message(parent, "错误", f"{error_message}: {str(e)}", "error"))
            raise

    # 在新线程中运行函数
    thread = threading.Thread(target=run_function, daemon=True)
    thread.start()


class AsyncTask:
    """异步任务管理器"""

    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.loading_window = None
        self.progress = None
        self.result_queue = queue.Queue()

    def run_task(self, func: Callable, *args, **kwargs) -> Any:
        """
        运行异步任务

        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数

        Returns:
            函数执行结果
        """
        # 显示加载窗口
        message = kwargs.pop('loading_message', '正在处理...')
        self.show_loading(message)

        # 在后台线程中运行任务
        def task_wrapper():
            try:
                result = func(*args, **kwargs)
                self.result_queue.put(('success', result))
            except Exception as e:
                self.result_queue.put(('error', str(e)))

        thread = threading.Thread(target=task_wrapper, daemon=True)
        thread.start()

        # 定期检查结果
        self.parent.after(100, self.check_result)

        return None

    def show_loading(self, message: str = "正在处理..."):
        """显示加载指示器"""
        if self.loading_window:
            self.hide_loading()

        self.loading_window, self.progress = show_loading(self.parent, message)

    def hide_loading(self):
        """隐藏加载指示器"""
        if self.loading_window:
            hide_loading(self.loading_window, self.progress)
            self.loading_window = None
            self.progress = None

    def check_result(self):
        """检查任务结果"""
        try:
            if not self.result_queue.empty():
                status, result = self.result_queue.get_nowait()
                self.hide_loading()

                if status == 'success':
                    self.on_success(result)
                else:
                    self.on_error(result)
            else:
                # 继续检查
                self.parent.after(100, self.check_result)
        except queue.Empty:
            self.parent.after(100, self.check_result)

    def on_success(self, result):
        """成功回调，子类可重写"""
        pass

    def on_error(self, error):
        """错误回调，子类可重写"""
        show_message(self.parent, "错误", f"操作失败: {error}", "error")


class ProgressDialog:
    """进度对话框"""

    def __init__(self, parent: tk.Widget, title: str = "进度", message: str = "正在处理..."):
        self.parent = parent
        self.window = create_centered_window(title, 400, 150)
        self.window.transient(parent)
        self.window.grab_set()

        # 创建界面
        frame = ttk.Frame(self.window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        # 消息标签
        self.message_label = ttk.Label(frame, text=message)
        self.message_label.pack(pady=(0, 10))

        # 进度条
        self.progress = ttk.Progressbar(frame, mode='determinate', length=300)
        self.progress.pack(pady=(0, 10))

        # 百分比标签
        self.percentage_label = ttk.Label(frame, text="0%")
        self.percentage_label.pack()

        # 取消按钮
        self.cancel_button = ttk.Button(frame, text="取消", command=self.cancel)
        self.cancel_button.pack(pady=(10, 0))

        self.cancelled = False

    def update_progress(self, value: int, message: str = None):
        """
        更新进度

        Args:
            value: 进度值 (0-100)
            message: 新的消息文本
        """
        if not self.cancelled:
            self.progress['value'] = value
            self.percentage_label.config(text=f"{value}%")

            if message:
                self.message_label.config(text=message)

            self.window.update_idletasks()

    def cancel(self):
        """取消操作"""
        self.cancelled = True
        self.window.grab_release()
        self.window.destroy()

    def close(self):
        """关闭对话框"""
        if not self.cancelled:
            self.window.grab_release()
            self.window.destroy()

    def is_cancelled(self) -> bool:
        """检查是否已取消"""
        return self.cancelled


def create_tooltip(widget: tk.Widget, text: str) -> None:
    """
    为组件创建工具提示

    Args:
        widget: 目标组件
        text: 提示文本
    """
    def on_enter(event):
        tooltip = tk.Toplevel()
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")

        label = tk.Label(tooltip, text=text, background="lightyellow",
                        relief="solid", borderwidth=1, font=("Arial", 9))
        label.pack()

        widget.tooltip = tooltip

    def on_leave(event):
        if hasattr(widget, 'tooltip'):
            widget.tooltip.destroy()
            del widget.tooltip

    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)


def set_window_icon(window: tk.Tk, icon_path: str) -> None:
    """
    设置窗口图标

    Args:
        window: 窗口对象
        icon_path: 图标文件路径
    """
    try:
        window.iconbitmap(icon_path)
    except:
        # 如果设置失败，静默忽略
        pass


def create_separator(parent: tk.Widget, orient: str = "horizontal") -> ttk.Separator:
    """
    创建分隔线

    Args:
        parent: 父组件
        orient: 方向 ("horizontal" 或 "vertical")

    Returns:
        分隔线组件
    """
    separator = ttk.Separator(parent, orient=orient)
    return separator


def add_padding(widget: tk.Widget, padx: int = 5, pady: int = 5, **pack_options) -> None:
    """
    为组件添加内边距

    Args:
        widget: 目标组件
        padx: 水平内边距
        pady: 垂直内边距
        **pack_options: pack布局选项
    """
    frame = ttk.Frame(widget)
    frame.pack(**pack_options)

    widget.pack(in_=frame, padx=padx, pady=pady)


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小

    Args:
        size_bytes: 文件大小（字节）

    Returns:
        格式化的文件大小字符串
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f} {size_names[i]}"


def validate_number(value: str, min_val: float = None, max_val: float = None) -> bool:
    """
    验证数字输入

    Args:
        value: 输入值
        min_val: 最小值
        max_val: 最大值

    Returns:
        是否有效
    """
    try:
        num = float(value)
        if min_val is not None and num < min_val:
            return False
        if max_val is not None and num > max_val:
            return False
        return True
    except ValueError:
        return False