#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理组件 - 处理模型保存、加载和导入导出
"""

import os
import pickle
import json
import shutil
import tempfile
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional
import warnings

class ModelManager:
    """模型管理器"""

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            # 默认在项目根目录下的models文件夹
            self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        else:
            self.models_dir = models_dir

        # 确保模型目录存在
        os.makedirs(self.models_dir, exist_ok=True)

    def save_model(self, model: Any, model_name: str, model_info: Optional[Dict] = None) -> str:
        """
        保存模型

        Args:
            model: 要保存的模型对象
            model_name: 模型名称
            model_info: 模型信息字典

        Returns:
            保存的模型文件路径
        """
        try:
            # 生成唯一的模型文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model_name = model_name.replace(" ", "_").replace("/", "_")
            model_filename = f"{safe_model_name}_{timestamp}.pkl"
            model_path = os.path.join(self.models_dir, model_filename)

            # 保存模型
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            # 保存模型信息
            if model_info:
                info_filename = model_filename.replace('.pkl', '_info.json')
                info_path = os.path.join(self.models_dir, info_filename)

                # 添加保存时间
                model_info['saved_at'] = datetime.now().isoformat()
                model_info['filename'] = model_filename

                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(model_info, f, ensure_ascii=False, indent=2, default=str)

            return model_path

        except Exception as e:
            raise Exception(f"模型保存失败: {str(e)}")

    def load_model(self, model_name: str) -> Any:
        """
        加载模型

        Args:
            model_name: 模型名称（不包含扩展名）

        Returns:
            加载的模型对象
        """
        try:
            # 尝试找到模型文件
            model_path = self._find_model_file(model_name)

            if not model_path:
                raise FileNotFoundError(f"未找到模型文件: {model_name}")

            # 加载模型
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            return model

        except Exception as e:
            raise Exception(f"模型加载失败: {str(e)}")

    def _find_model_file(self, model_name: str) -> Optional[str]:
        """查找模型文件"""
        # 尝试不同的文件名格式
        possible_names = [
            f"{model_name}.pkl",
            f"{model_name}.model",
            model_name
        ]

        for name in possible_names:
            path = os.path.join(self.models_dir, name)
            if os.path.exists(path):
                return path

        # 如果直接路径不存在，尝试模糊匹配
        model_files = self.get_available_models()
        for model_file in model_files:
            if model_name in model_file:
                return os.path.join(self.models_dir, f"{model_file}.pkl")

        return None

    def get_available_models(self) -> List[str]:
        """
        获取所有可用的模型列表

        Returns:
            模型名称列表（不包含扩展名）
        """
        models = []

        try:
            if os.path.exists(self.models_dir):
                for file in os.listdir(self.models_dir):
                    if file.endswith('.pkl'):
                        model_name = file.replace('.pkl', '')
                        models.append(model_name)

                # 按修改时间排序，最新的在前
                models.sort(key=lambda x: os.path.getmtime(os.path.join(self.models_dir, f"{x}.pkl")), reverse=True)

        except Exception as e:
            warnings.warn(f"获取模型列表失败: {str(e)}")

        return models

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        获取模型信息

        Args:
            model_name: 模型名称

        Returns:
            模型信息字典
        """
        try:
            # 尝试找到信息文件
            info_path = self._find_info_file(model_name)

            if info_path and os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 如果没有信息文件，返回基本信息
                model_path = self._find_model_file(model_name)
                if model_path:
                    stat = os.stat(model_path)
                    return {
                        'model_name': model_name,
                        'filename': os.path.basename(model_path),
                        'file_size': stat.st_size,
                        'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    }

        except Exception as e:
            warnings.warn(f"获取模型信息失败: {str(e)}")

        return None

    def _find_info_file(self, model_name: str) -> Optional[str]:
        """查找模型信息文件"""
        possible_names = [
            f"{model_name}_info.json",
            f"{model_name}.json"
        ]

        for name in possible_names:
            path = os.path.join(self.models_dir, name)
            if os.path.exists(path):
                return path

        # 模糊匹配
        model_files = os.listdir(self.models_dir)
        for file in model_files:
            if file.endswith('_info.json') and model_name in file:
                return os.path.join(self.models_dir, file)

        return None

    def delete_model(self, model_name: str) -> bool:
        """
        删除模型

        Args:
            model_name: 模型名称

        Returns:
            是否删除成功
        """
        try:
            deleted = False

            # 删除模型文件
            model_path = self._find_model_file(model_name)
            if model_path and os.path.exists(model_path):
                os.remove(model_path)
                deleted = True

            # 删除信息文件
            info_path = self._find_info_file(model_name)
            if info_path and os.path.exists(info_path):
                os.remove(info_path)
                deleted = True

            return deleted

        except Exception as e:
            warnings.warn(f"删除模型失败: {str(e)}")
            return False

    def export_model(self, model_name: str, export_path: str) -> bool:
        """
        导出模型为zip文件

        Args:
            model_name: 模型名称
            export_path: 导出文件路径

        Returns:
            是否导出成功
        """
        try:
            model_path = self._find_model_file(model_name)
            if not model_path:
                raise FileNotFoundError(f"未找到模型: {model_name}")

            info_path = self._find_info_file(model_name)

            with tempfile.TemporaryDirectory() as temp_dir:
                # 复制模型文件
                shutil.copy2(model_path, temp_dir)

                # 复制信息文件（如果存在）
                if info_path and os.path.exists(info_path):
                    shutil.copy2(info_path, temp_dir)

                # 创建zip文件
                with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in os.listdir(temp_dir):
                        file_path = os.path.join(temp_dir, file)
                        zipf.write(file_path, file)

            return True

        except Exception as e:
            raise Exception(f"模型导出失败: {str(e)}")

    def import_model(self, zip_path: str) -> bool:
        """
        从zip文件导入模型

        Args:
            zip_path: zip文件路径

        Returns:
            是否导入成功
        """
        try:
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"文件不存在: {zip_path}")

            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(self.models_dir)

            return True

        except Exception as e:
            raise Exception(f"模型导入失败: {str(e)}")

    def get_model_statistics(self) -> Dict:
        """
        获取模型统计信息

        Returns:
            统计信息字典
        """
        try:
            models = self.get_available_models()
            total_size = 0
            model_types = {}
            creation_times = []

            for model_name in models:
                model_path = self._find_model_file(model_name)
                if model_path and os.path.exists(model_path):
                    stat = os.stat(model_path)
                    total_size += stat.st_size
                    creation_times.append(stat.st_ctime)

                    # 获取模型类型
                    model_info = self.get_model_info(model_name)
                    if model_info and 'task_type' in model_info:
                        task_type = model_info['task_type']
                        model_types[task_type] = model_types.get(task_type, 0) + 1

            return {
                'total_models': len(models),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'model_types': model_types,
                'oldest_model': datetime.fromtimestamp(min(creation_times)).isoformat() if creation_times else None,
                'newest_model': datetime.fromtimestamp(max(creation_times)).isoformat() if creation_times else None,
                'models_directory': self.models_dir
            }

        except Exception as e:
            warnings.warn(f"获取模型统计失败: {str(e)}")
            return {}

    def backup_models(self, backup_path: str) -> bool:
        """
        备份所有模型

        Args:
            backup_path: 备份文件路径

        Returns:
            是否备份成功
        """
        try:
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(self.models_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.models_dir)
                        zipf.write(file_path, arcname)

            return True

        except Exception as e:
            raise Exception(f"模型备份失败: {str(e)}")

    def cleanup_old_models(self, keep_days: int = 30) -> int:
        """
        清理旧模型文件

        Args:
            keep_days: 保留天数

        Returns:
            删除的文件数量
        """
        try:
            cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 3600)
            deleted_count = 0

            for file in os.listdir(self.models_dir):
                file_path = os.path.join(self.models_dir, file)
                if os.path.isfile(file_path):
                    if os.path.getmtime(file_path) < cutoff_time:
                        os.remove(file_path)
                        deleted_count += 1

            return deleted_count

        except Exception as e:
            warnings.warn(f"清理旧模型失败: {str(e)}")
            return 0

    def validate_model(self, model_name: str) -> Dict:
        """
        验证模型文件完整性

        Args:
            model_name: 模型名称

        Returns:
            验证结果字典
        """
        result = {
            'valid': False,
            'error': None,
            'info': None
        }

        try:
            model_path = self._find_model_file(model_name)
            if not model_path:
                result['error'] = "模型文件不存在"
                return result

            # 检查文件大小
            file_size = os.path.getsize(model_path)
            if file_size == 0:
                result['error'] = "模型文件为空"
                return result

            # 尝试加载模型
            try:
                with open(model_path, 'rb') as f:
                    pickle.load(f)

                result['valid'] = True
                result['info'] = self.get_model_info(model_name)

            except Exception as e:
                result['error'] = f"模型文件损坏: {str(e)}"

        except Exception as e:
            result['error'] = f"验证过程出错: {str(e)}"

        return result