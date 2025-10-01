#!/usr/bin/env python3
"""
特征库管理模块 - 负责特征选择和配置管理
"""
import json
import streamlit as st
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, asdict
import os


@dataclass
class FeatureConfig:
    """特征配置数据类"""
    name: str
    display_name: str
    category: str
    description: str
    enabled: bool = True
    sub_features: Optional[List[str]] = None  # 对于复合特征（如价格特征需要多天数据）

    def __post_init__(self):
        if self.sub_features is None:
            self.sub_features = []


class FeatureLibrary:
    """特征库管理类"""

    def __init__(self):
        self.features = self._initialize_feature_library()
        self.config_file = "feature_configs.json"

    def _initialize_feature_library(self) -> Dict[str, FeatureConfig]:
        """初始化特征库"""
        features = {}

        # 基础价格特征
        basic_features = [
            ("open", "开盘价", "基础价格数据"),
            ("high", "最高价", "基础价格数据"),
            ("low", "最低价", "基础价格数据"),
            ("close", "收盘价", "基础价格数据"),
            ("volume", "成交量", "基础交易数据"),
        ]

        for feature_name, display_name, description in basic_features:
            features[feature_name] = FeatureConfig(
                name=feature_name,
                display_name=display_name,
                category="基础数据",
                description=description
            )

        # 移动平均线指标
        ma_features = [
            ("MA5", "5日移动平均线", "趋势指标"),
            ("MA10", "10日移动平均线", "趋势指标"),
            ("MA20", "20日移动平均线", "趋势指标"),
        ]

        for feature_name, display_name, category in ma_features:
            features[feature_name] = FeatureConfig(
                name=feature_name,
                display_name=display_name,
                category=category,
                description=f"{display_name}，用于判断价格趋势"
            )

        # 技术指标
        technical_features = [
            ("RSI", "相对强弱指数", "动量指标", "衡量超买超卖状态"),
            ("MACD", "MACD指标", "动量指标", "趋势跟踪指标"),
            ("Signal", "MACD信号线", "动量指标", "MACD的信号线"),
            ("Histogram", "MACD柱状图", "动量指标", "MACD与信号线的差值"),
            ("BB_upper", "布林带上轨", "波动率指标", "价格压力位"),
            ("BB_middle", "布林带中轨", "波动率指标", "20日移动平均线"),
            ("BB_lower", "布林带下轨", "波动率指标", "价格支撑位"),
            ("Williams_R", "威廉指标", "动量指标", "判断超买超卖"),
            ("K_value", "KDJ随机指标K值", "动量指标", "随机指标的核心线"),
            ("D_value", "KDJ随机指标D值", "动量指标", "K值的移动平均"),
            ("J_value", "KDJ随机指标J值", "动量指标", "KDJ的辅助指标"),
            ("momentum", "动量指标", "动量指标", "价格变化率"),
            ("price_acceleration", "价格加速指标", "动量指标", "价格变化的变化率"),
            ("VWAP", "成交量加权平均价", "成交量指标", "成交量加权的平均价格"),
            ("ATR", "平均真实波幅", "波动率指标", "衡量价格波动幅度"),
            ("CCI", "商品通道指数", "动量指标", "识别价格偏离"),
            ("OBV", "能量潮指标", "成交量指标", "成交量与价格关系"),
        ]

        for feature_name, display_name, category, description in technical_features:
            features[feature_name] = FeatureConfig(
                name=feature_name,
                display_name=display_name,
                category=category,
                description=description
            )

        # 价格变化特征
        price_change_features = [
            ("price_change", "单日价格变化率", "价格变化"),
            ("price_change_3d", "3日价格变化率", "价格变化"),
            ("price_change_5d", "5日价格变化率", "价格变化"),
        ]

        for feature_name, display_name, category in price_change_features:
            features[feature_name] = FeatureConfig(
                name=feature_name,
                display_name=display_name,
                category=category,
                description=f"{display_name}，反映价格变化趋势"
            )

        # 成交量特征
        volume_features = [
            ("volume_MA5", "5日成交量平均", "成交量指标"),
            ("volume_MA10", "10日成交量平均", "成交量指标"),
            ("volume_ratio", "成交量比率", "成交量指标"),
            ("price_position", "价格位置", "价格特征"),
            ("volatility", "波动率", "波动率指标"),
        ]

        for feature_name, display_name, category in volume_features:
            features[feature_name] = FeatureConfig(
                name=feature_name,
                display_name=display_name,
                category=category,
                description=f"{display_name}，用于成交量分析"
            )

        return features

    def get_features_by_category(self) -> Dict[str, List[FeatureConfig]]:
        """按类别获取特征"""
        categories = {}
        for feature_config in self.features.values():
            if feature_config.category not in categories:
                categories[feature_config.category] = []
            categories[feature_config.category].append(feature_config)
        return categories

    def get_enabled_features(self) -> Set[str]:
        """获取启用的特征名称集合"""
        return {name for name, config in self.features.items() if config.enabled}

    def set_feature_enabled(self, feature_name: str, enabled: bool):
        """设置特征启用状态"""
        if feature_name in self.features:
            self.features[feature_name].enabled = enabled

    def enable_features_by_category(self, category: str, enabled: bool):
        """按类别启用/禁用特征"""
        for feature_config in self.features.values():
            if feature_config.category == category:
                feature_config.enabled = enabled

    def get_feature_config(self, feature_name: str) -> Optional[FeatureConfig]:
        """获取特定特征的配置"""
        return self.features.get(feature_name)

    def save_config(self):
        """保存特征配置到文件"""
        try:
            config_data = {name: asdict(config) for name, config in self.features.items()}
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            st.error(f"保存特征配置失败: {str(e)}")
            return False

    def load_config(self):
        """从文件加载特征配置"""
        if not os.path.exists(self.config_file):
            return False

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            for name, config_dict in config_data.items():
                if name in self.features:
                    # 更新现有特征的配置
                    self.features[name].enabled = config_dict.get('enabled', True)

            return True
        except Exception as e:
            st.error(f"加载特征配置失败: {str(e)}")
            return False

    def create_preset_configs(self) -> Dict[str, Dict[str, bool]]:
        """创建预设的特征配置"""
        presets = {
            "基础配置": {
                "open": True, "high": True, "low": True, "close": True, "volume": True,
                "MA5": True, "MA10": True, "MA20": True, "RSI": True, "MACD": True
            },
            "技术分析": {
                "open": True, "high": True, "low": True, "close": True, "volume": True,
                "MA5": True, "MA10": True, "MA20": True, "RSI": True, "MACD": True,
                "Signal": True, "Histogram": True, "BB_upper": True, "BB_middle": True,
                "BB_lower": True, "Williams_R": True, "K_value": True, "D_value": True,
                "price_change": True, "volume_ratio": True, "volatility": True
            },
            "全特征": {name: True for name in self.features.keys()},
            "仅价格": {
                "open": True, "high": True, "low": True, "close": True,
                "MA5": True, "MA10": True, "MA20": True,
                "price_change": True, "price_change_3d": True, "price_change_5d": True
            },
            "仅技术指标": {
                "RSI": True, "MACD": True, "Signal": True, "Histogram": True,
                "BB_upper": True, "BB_middle": True, "BB_lower": True,
                "Williams_R": True, "K_value": True, "D_value": True, "J_value": True,
                "momentum": True, "VWAP": True, "ATR": True, "CCI": True, "OBV": True
            }
        }
        return presets

    def apply_preset(self, preset_name: str):
        """应用预设配置"""
        presets = self.create_preset_configs()
        if preset_name in presets:
            preset_config = presets[preset_name]
            for feature_name, enabled in preset_config.items():
                if feature_name in self.features:
                    self.features[feature_name].enabled = enabled
            return True
        return False

    def get_training_features_list(self) -> List[str]:
        """获取用于训练的特征列表（按启用的特征）"""
        enabled_features = []
        for name, config in self.features.items():
            if config.enabled:
                enabled_features.append(name)
        return enabled_features

    def get_summary_stats(self) -> Dict[str, Any]:
        """获取特征库统计信息"""
        total_features = len(self.features)
        enabled_features = len(self.get_enabled_features())
        categories = len(set(config.category for config in self.features.values()))

        return {
            "总特征数": total_features,
            "已启用特征": enabled_features,
            "未启用特征": total_features - enabled_features,
            "特征类别": categories,
            "启用率": f"{(enabled_features / total_features * 100):.1f}%" if total_features > 0 else "0%"
        }


# 全局特征库实例
feature_library = FeatureLibrary()