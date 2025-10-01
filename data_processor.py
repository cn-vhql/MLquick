#!/usr/bin/env python3
"""
数据处理模块 - 负责特征工程和技术指标计算
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Optional


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算技术指标

    Args:
        df: 包含OHLCV数据的DataFrame

    Returns:
        添加技术指标的DataFrame
    """
    df_processed = df.copy()

    # 确保数据按日期排序
    df_processed = df_processed.sort_index()

    st.info("正在计算技术指标...")

    # 移动平均线
    df_processed['MA5'] = df_processed['close'].rolling(window=5).mean()
    df_processed['MA10'] = df_processed['close'].rolling(window=10).mean()
    df_processed['MA20'] = df_processed['close'].rolling(window=20).mean()

    # RSI指标
    delta = df_processed['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_processed['RSI'] = 100 - (100 / (1 + rs))

    # MACD指标
    exp1 = df_processed['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_processed['close'].ewm(span=26, adjust=False).mean()
    df_processed['MACD'] = exp1 - exp2
    df_processed['Signal'] = df_processed['MACD'].ewm(span=9, adjust=False).mean()
    df_processed['Histogram'] = df_processed['MACD'] - df_processed['Signal']

    # 布林带
    df_processed['BB_middle'] = df_processed['close'].rolling(window=20).mean()
    bb_std = df_processed['close'].rolling(window=20).std()
    df_processed['BB_upper'] = df_processed['BB_middle'] + (bb_std * 2)
    df_processed['BB_lower'] = df_processed['BB_middle'] - (bb_std * 2)

    # 价格变化率
    df_processed['price_change'] = df_processed['close'].pct_change()
    df_processed['price_change_3d'] = df_processed['close'].pct_change(3)
    df_processed['price_change_5d'] = df_processed['close'].pct_change(5)

    # 成交量相关指标
    if 'volume' in df_processed.columns:
        df_processed['volume_MA5'] = df_processed['volume'].rolling(window=5).mean()
        df_processed['volume_MA10'] = df_processed['volume'].rolling(window=10).mean()
        # 避免除零错误
        df_processed['volume_ratio'] = df_processed['volume'] / df_processed['volume_MA5'].replace(0, 1)

    # 价格位置指标（避免除零错误）
    high_low_diff = df_processed['high'] - df_processed['low']
    df_processed['price_position'] = (df_processed['close'] - df_processed['low']) / high_low_diff.replace(0, 1)

    # 波动率
    df_processed['volatility'] = df_processed['close'].rolling(window=10).std()

    # === 新增技术指标 ===

    # 1. 威廉指标 %R (Williams %R)
    high_14 = df_processed['high'].rolling(window=14).max()
    low_14 = df_processed['low'].rolling(window=14).min()
    df_processed['Williams_R'] = ((high_14 - df_processed['close']) / (high_14 - low_14)) * -100

    # 2. 随机指标 KDJ
    low_9 = df_processed['low'].rolling(window=9).min()
    high_9 = df_processed['high'].rolling(window=9).max()
    rsv = (df_processed['close'] - low_9) / (high_9 - low_9) * 100
    df_processed['K_value'] = rsv.ewm(com=2).mean()
    df_processed['D_value'] = df_processed['K_value'].ewm(com=2).mean()
    df_processed['J_value'] = 3 * df_processed['K_value'] - 2 * df_processed['D_value']

    # 3. 动量指标 (Momentum)
    df_processed['momentum'] = df_processed['close'] / df_processed['close'].shift(10) - 1

    # 4. 价格加速指标 (Price Acceleration)
    df_processed['price_acceleration'] = df_processed['close'].pct_change().diff()

    # 5. 成交量加权平均价 (VWAP)
    typical_price = (df_processed['high'] + df_processed['low'] + df_processed['close']) / 3
    vwap = (typical_price * df_processed['volume']).rolling(window=20).sum() / df_processed['volume'].rolling(window=20).sum()
    df_processed['VWAP'] = vwap

    # 6. 平均真实波幅 (ATR)
    high_low = df_processed['high'] - df_processed['low']
    high_close = abs(df_processed['high'] - df_processed['close'].shift())
    low_close = abs(df_processed['low'] - df_processed['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_processed['ATR'] = true_range.rolling(window=14).mean()

    # 7. 商品通道指数 (CCI)
    tp = (df_processed['high'] + df_processed['low'] + df_processed['close']) / 3
    sma_tp = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df_processed['CCI'] = (tp - sma_tp) / (0.015 * mad)

    # 8. 能量潮指标 (OBV)
    obv = np.where(df_processed['close'] > df_processed['close'].shift(), df_processed['volume'],
                  np.where(df_processed['close'] < df_processed['close'].shift(), -df_processed['volume'], 0)).cumsum()
    df_processed['OBV'] = obv

    # 处理无穷大值
    df_processed = df_processed.replace([np.inf, -np.inf], np.nan)

    # 计算技术指标后的数据质量检查
    original_count = len(df_processed)
    df_processed = df_processed.dropna()
    cleaned_count = len(df_processed)

    if cleaned_count < original_count:
        st.warning(f"技术指标计算后数据量从 {original_count} 减少到 {cleaned_count} (减少了 {original_count - cleaned_count} 条)")

    st.info(f"技术指标计算完成，保留 {cleaned_count} 条有效数据")

    return df_processed


def create_features_targets(df: pd.DataFrame, historical_days: int = 7,
                          prediction_days: int = 3, task_type: str = 'regression') -> Tuple[pd.DataFrame, pd.Series]:
    """
    创建特征矩阵和目标变量

    Args:
        df: 包含技术指标的DataFrame
        historical_days: 使用过去多少天的数据作为特征
        prediction_days: 预测未来多少天
        task_type: 任务类型 ('regression' 或 'classification')

    Returns:
        特征矩阵和目标变量
    """
    # 特征列
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'Signal', 'Histogram',
        'BB_upper', 'BB_middle', 'BB_lower',
        'price_change', 'price_change_3d', 'price_change_5d'
    ]

    # 添加成交量相关特征
    volume_features = ['volume_MA5', 'volume_MA10', 'volume_ratio', 'price_position', 'volatility']
    for feature in volume_features:
        if feature in df.columns:
            feature_columns.append(feature)

    # 添加新的技术指标特征
    new_features = [
        'Williams_R', 'K_value', 'D_value', 'J_value', 'momentum',
        'price_acceleration', 'VWAP', 'ATR', 'CCI', 'OBV'
    ]
    for feature in new_features:
        if feature in df.columns:
            feature_columns.append(feature)

    # 确保所有特征列都存在
    available_features = [col for col in feature_columns if col in df.columns]

    # 创建特征矩阵
    features_list = []
    targets = []

    for i in range(historical_days, len(df) - prediction_days):
        # 获取历史数据作为特征
        historical_data = df.iloc[i-historical_days:i][available_features]

        # 检查是否有NaN值
        if historical_data.isnull().any().any():
            continue  # 跳过包含NaN的数据

        # 将历史数据展平为一维特征向量
        feature_vector = historical_data.values.flatten()

        # 检查展平后的特征向量是否有NaN
        if np.any(np.isnan(feature_vector)):
            continue  # 跳过包含NaN的特征向量

        # 获取未来数据作为目标
        future_data = df.iloc[i:i+prediction_days]

        if task_type == 'regression':
            # 回归任务：预测未来价格变化百分比
            current_price = df.iloc[i-1]['close']
            future_price = future_data.iloc[-1]['close']
            price_change_pct = ((future_price - current_price) / current_price) * 100
            target = price_change_pct
        else:
            # 分类任务：预测涨跌方向
            current_price = df.iloc[i-1]['close']
            future_price = future_data.iloc[-1]['close']
            price_change_pct = ((future_price - current_price) / current_price) * 100

            if price_change_pct > 0.1:
                target = 2  # 上涨
            elif price_change_pct < -0.1:
                target = 0  # 下跌
            else:
                target = 1  # 震荡

        # 检查目标值是否有效
        if not np.isnan(target) and not np.isinf(target):
            features_list.append(feature_vector)
            targets.append(target)

    # 创建特征矩阵
    feature_names = []
    for day in range(historical_days):
        for feature in available_features:
            feature_names.append(f'{feature}_day_{day+1}')

    X = pd.DataFrame(features_list, columns=feature_names)
    y = pd.Series(targets)

    # 最终检查：确保没有NaN值
    if X.isnull().any().any():
        st.warning(f"在特征矩阵中发现NaN值，正在清理...")
        # 删除包含NaN的行
        valid_indices = ~X.isnull().any(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]

    if len(X) == 0:
        st.error("特征工程后没有有效数据，请检查数据质量或调整参数")
        return pd.DataFrame(), pd.Series()

    st.info(f"特征工程完成: {len(X)}个样本, {len(X.columns)}个特征")

    return X, y


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据预处理

    Args:
        df: 原始数据DataFrame

    Returns:
        预处理后的DataFrame
    """
    df_processed = df.copy()

    # 处理缺失值
    df_processed = df_processed.dropna()

    # 处理异常值
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        # 使用IQR方法处理异常值
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 将异常值替换为边界值
        df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)

    return df_processed


def split_data_by_date(df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按时间顺序分割数据

    Args:
        df: 数据DataFrame
        train_ratio: 训练集比例

    Returns:
        训练集和测试集
    """
    split_index = int(len(df) * train_ratio)
    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]

    return train_data, test_data


def get_feature_importance_data(X: pd.DataFrame, feature_importance: np.ndarray) -> pd.DataFrame:
    """
    整理特征重要性数据

    Args:
        X: 特征矩阵
        feature_importance: 特征重要性数组

    Returns:
        包含特征重要性的DataFrame
    """
    feature_names = X.columns

    # 按历史天数分组特征
    feature_groups = {}
    for i, feature_name in enumerate(feature_names):
        if '_day_' in feature_name:
            base_feature = feature_name.split('_day_')[0]
            day = feature_name.split('_day_')[1]

            if base_feature not in feature_groups:
                feature_groups[base_feature] = []
            feature_groups[base_feature].append((day, feature_importance[i]))

    # 计算每个基础特征的平均重要性
    avg_importance = {}
    for base_feature, importance_list in feature_groups.items():
        avg_importance[base_feature] = np.mean([imp for _, imp in importance_list])

    # 创建结果DataFrame
    importance_df = pd.DataFrame([
        {'Feature': feature, 'Importance': importance}
        for feature, importance in avg_importance.items()
    ])

    importance_df = importance_df.sort_values('Importance', ascending=False)

    return importance_df


def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    验证数据质量

    Args:
        df: 数据DataFrame

    Returns:
        包含数据质量信息的字典
    """
    quality_info = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'date_range': {
            'start': df.index.min() if len(df) > 0 else None,
            'end': df.index.max() if len(df) > 0 else None
        },
        'columns': list(df.columns),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns)
    }

    # 检查数据完整性
    if len(df) > 0:
        quality_info['completeness'] = (1 - quality_info['missing_values'] / (len(df) * len(df.columns))) * 100
    else:
        quality_info['completeness'] = 0

    return quality_info