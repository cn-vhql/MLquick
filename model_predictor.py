#!/usr/bin/env python3
"""
模型预测模块 - 负责未来趋势预测和报告生成
"""
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from data_processor import calculate_technical_indicators, create_features_targets


def predict_future_trend(model: Any, df: pd.DataFrame, historical_days: int = 7,
                        prediction_days: int = 3, task_type: str = 'regression') -> Dict[str, Any]:
    """
    使用训练好的模型预测未来趋势

    Args:
        model: 训练好的机器学习模型
        df: 历史数据DataFrame
        historical_days: 用于预测的历史数据天数
        prediction_days: 预测未来天数
        task_type: 任务类型 ('regression' 或 'classification')

    Returns:
        包含预测结果的字典
    """
    try:
        # 计算技术指标
        df_processed = calculate_technical_indicators(df)

        if len(df_processed) < historical_days + prediction_days:
            st.error(f"数据不足，需要至少 {historical_days + prediction_days} 天数据")
            return {}

        # 获取最近的历史数据用于预测
        recent_data = df_processed.tail(historical_days)

        # 特征列（需要与训练时保持一致）
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'Signal', 'Histogram',
            'BB_upper', 'BB_middle', 'BB_lower',
            'price_change', 'price_change_3d', 'price_change_5d'
        ]

        # 添加成交量相关特征
        volume_features = ['volume_MA5', 'volume_MA10', 'volume_ratio', 'price_position', 'volatility']
        for feature in volume_features:
            if feature in df_processed.columns:
                feature_columns.append(feature)

        # 确保所有特征列都存在
        available_features = [col for col in feature_columns if col in df_processed.columns]

        # 检查是否有足够的历史数据
        if len(recent_data) < historical_days:
            st.error(f"历史数据不足，只有 {len(recent_data)} 天，需要 {historical_days} 天")
            return {}

        # 创建预测特征
        prediction_features = []
        dates = []
        current_price = df_processed['close'].iloc[-1]

        # 逐步预测
        temp_df = df_processed.copy()

        for day in range(prediction_days):
            # 获取最新的历史数据
            latest_data = temp_df.tail(historical_days)[available_features]

            # 展平为一维特征向量
            feature_vector = latest_data.values.flatten()

            # 确保特征向量长度与训练时一致
            if len(feature_vector) != len(available_features) * historical_days:
                st.error(f"特征维度不匹配: 期望 {len(available_features) * historical_days}, 实际 {len(feature_vector)}")
                return {}

            # 预测
            feature_vector_reshaped = feature_vector.reshape(1, -1)
            prediction = model.predict(feature_vector_reshaped)[0]

            prediction_features.append(prediction)

            # 计算预测日期
            last_date = temp_df.index[-1]
            future_date = last_date + timedelta(days=1)
            dates.append(future_date)

            # 更新temp_df，添加预测的数据点
            if task_type == 'regression':
                # 回归：预测价格变化百分比
                price_change_pct = prediction
                predicted_price = current_price * (1 + price_change_pct / 100)
            else:
                # 分类：预测涨跌方向
                if prediction == 2:  # 上涨
                    price_change_pct = 2.0  # 假设上涨2%
                elif prediction == 0:  # 下跌
                    price_change_pct = -2.0  # 假设下跌2%
                else:  # 震荡
                    price_change_pct = 0.0  # 假设不变
                predicted_price = current_price * (1 + price_change_pct / 100)

            # 创建新的数据点
            new_row = {
                'open': predicted_price,
                'high': predicted_price * 1.01,  # 假设最高价略高于收盘价
                'low': predicted_price * 0.99,   # 假设最低价略低于收盘价
                'close': predicted_price,
                'volume': temp_df['volume'].tail(5).mean()  # 使用最近5天平均成交量
            }

            # 添加到temp_df
            new_row_df = pd.DataFrame([new_row], index=[future_date])
            temp_df = pd.concat([temp_df, new_row_df])

            # 重新计算技术指标
            temp_df = calculate_technical_indicators(temp_df)
            current_price = predicted_price

        return {
            'predictions': prediction_features,
            'dates': dates,
            'current_price': df_processed['close'].iloc[-1],
            'task_type': task_type,
            'prediction_days': prediction_days,
            'historical_data': df_processed.tail(30)  # 保留最近30天历史数据用于可视化
        }

    except Exception as e:
        st.error(f"预测过程出错: {str(e)}")
        return {}


def calculate_prediction_confidence(predictions: List[float], task_type: str) -> Dict[str, float]:
    """
    计算预测置信度

    Args:
        predictions: 预测结果列表
        task_type: 任务类型

    Returns:
        包含置信度指标的字典
    """
    if not predictions:
        return {'confidence': 0.0, 'volatility': 0.0}

    if task_type == 'regression':
        # 回归任务：基于预测值的分布计算置信度
        pred_array = np.array(predictions)
        mean_pred = np.mean(pred_array)
        std_pred = np.std(pred_array)

        # 置信度基于标准差，标准差越小置信度越高
        confidence = max(0, 1 - (std_pred / abs(mean_pred + 1e-6)))
        volatility = std_pred

        return {
            'confidence': min(confidence, 1.0),
            'volatility': volatility,
            'mean_prediction': mean_pred
        }
    else:
        # 分类任务：基于类别分布计算置信度
        pred_array = np.array(predictions)
        unique, counts = np.unique(pred_array, return_counts=True)
        most_common_count = np.max(counts)

        # 置信度基于最常见类别的比例
        confidence = most_common_count / len(predictions)

        return {
            'confidence': confidence,
            'class_distribution': dict(zip(unique, counts)),
            'most_common_class': unique[np.argmax(counts)]
        }


def plot_prediction_results(historical_data: pd.DataFrame, predictions: List[float],
                          dates: List[datetime], task_type: str = 'regression',
                          show_confidence: bool = True) -> plt.Figure:
    """
    绘制预测结果图

    Args:
        historical_data: 历史数据
        predictions: 预测结果
        dates: 预测日期
        task_type: 任务类型
        show_confidence: 是否显示置信区间

    Returns:
        matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制历史价格
    ax.plot(historical_data.index, historical_data['close'],
           label='Historical Price', color='blue', linewidth=2)

    # 获取最后一个历史价格
    last_price = historical_data['close'].iloc[-1]
    last_date = historical_data.index[-1]

    if task_type == 'regression':
        # 回归任务：绘制价格预测
        predicted_prices = []
        current_price = last_price

        for pred_change in predictions:
            predicted_price = current_price * (1 + pred_change / 100)
            predicted_prices.append(predicted_price)
            current_price = predicted_price

        # 绘制预测价格
        ax.plot(dates, predicted_prices,
               label='Predicted Price', color='red', linewidth=2, linestyle='--')

        # 绘制置信区间
        if show_confidence and len(predictions) > 1:
            pred_array = np.array(predictions)
            std_pred = np.std(pred_array)

            # 计算置信区间
            confidence_prices = []
            current_price_upper = last_price
            current_price_lower = last_price

            for pred_change in predictions:
                # 上限
                upper_change = pred_change + std_pred
                predicted_price_upper = current_price_upper * (1 + upper_change / 100)
                confidence_prices.append(predicted_price_upper)
                current_price_upper = predicted_price_upper

            current_price_upper = last_price
            for pred_change in predictions:
                # 下限
                lower_change = pred_change - std_pred
                predicted_price_lower = current_price_lower * (1 + lower_change / 100)
                confidence_prices.append(predicted_price_lower)
                current_price_lower = predicted_price_lower

            # 绘制置信区间
            ax.fill_between(dates,
                          [p - std_pred for p in predicted_prices],
                          [p + std_pred for p in predicted_prices],
                          alpha=0.3, color='red', label='Confidence Interval')

    else:
        # 分类任务：绘制趋势预测
        predicted_prices = []
        current_price = last_price

        for pred_class in predictions:
            if pred_class == 2:  # 上涨
                price_change = 0.02  # 假设上涨2%
                color = 'green'
            elif pred_class == 0:  # 下跌
                price_change = -0.02  # 假设下跌2%
                color = 'red'
            else:  # 震荡
                price_change = 0.0
                color = 'orange'

            predicted_price = current_price * (1 + price_change)
            predicted_prices.append(predicted_price)
            current_price = predicted_price

        # 绘制预测价格
        ax.plot(dates, predicted_prices,
               label='Predicted Trend', color='red', linewidth=2, linestyle='--', marker='o')

    # 设置图表样式
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Futures Price Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 格式化日期轴
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig


def generate_prediction_report(prediction_results: Dict[str, Any], symbol: str) -> str:
    """
    生成详细的预测报告

    Args:
        prediction_results: 预测结果字典
        symbol: 期货品种代码

    Returns:
        预测报告文本
    """
    if not prediction_results:
        return "预测失败，无法生成报告"

    current_price = prediction_results['current_price']
    predictions = prediction_results['predictions']
    dates = prediction_results['dates']
    task_type = prediction_results['task_type']
    prediction_days = prediction_results['prediction_days']

    # 计算置信度
    confidence_info = calculate_prediction_confidence(predictions, task_type)

    report = f"""
# {symbol} 期货行情预测报告

## 基本信息
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 当前价格: {current_price:.2f}
- 预测天数: {prediction_days} 天
- 预测类型: {'价格预测' if task_type == 'regression' else '趋势预测'}

## 预测结果
"""

    if task_type == 'regression':
        report += "\n### 价格变化预测\n"
        predicted_prices = []
        current_price_temp = current_price

        for i, (pred_change, date) in enumerate(zip(predictions, dates)):
            predicted_price = current_price_temp * (1 + pred_change / 100)
            predicted_prices.append(predicted_price)
            price_change = predicted_price - current_price
            change_pct = (price_change / current_price) * 100

            direction = "📈 Up" if change_pct > 0 else "📉 Down" if change_pct < 0 else "➡️ Sideways"

            report += f"""
**第{i+1}天 ({date.strftime('%Y-%m-%d')})**
- 预测价格: {predicted_price:.2f}
- 价格变化: {price_change:+.2f} ({change_pct:+.2f}%)
- 趋势: {direction}
"""
            current_price_temp = predicted_price

        total_change = (predicted_prices[-1] - current_price) / current_price * 100
        report += f"""
### 总体预测
- 期末价格: {predicted_prices[-1]:.2f}
- 总变化: {total_change:+.2f}%
- 置信度: {confidence_info['confidence']:.2f}
- 预测波动性: {confidence_info['volatility']:.4f}
"""

    else:
        report += "\n### 趋势方向预测\n"
        trend_names = {0: "Down 📉", 1: "Sideways ➡️", 2: "Up 📈"}

        for i, (pred_class, date) in enumerate(zip(predictions, dates)):
            trend_name = trend_names.get(pred_class, f"未知 ({pred_class})")
            report += f"""
**第{i+1}天 ({date.strftime('%Y-%m-%d')})**
- 预测趋势: {trend_name}
- 置信度: {confidence_info['confidence']:.2f}
"""

        most_common_class = confidence_info.get('most_common_class', predictions[0])
        overall_trend = trend_names.get(most_common_class, f"未知 ({most_common_class})")

        report += f"""
### 总体预测
- 主要趋势: {overall_trend}
- 平均置信度: {confidence_info['confidence']:.2f}
"""

    report += f"""
## 重要提示
⚠️ **投资风险提示**:
- 本预测基于历史数据和机器学习模型，仅供参考
- 期货市场存在较大风险，请谨慎投资
- 预测结果不构成任何投资建议
- 请结合其他分析方法进行综合判断
- 投资有风险，入市需谨慎

## 技术说明
- 本预测使用的技术指标包括：移动平均线(MA5,MA10,MA20)、RSI、MACD、布林带等
- 模型基于最近{prediction_results.get('historical_data_length', '未知')}天的历史数据进行训练
- 预测结果会随着市场情况的变化而调整

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*本报告由AI系统生成，仅供参考*
"""

    return report


def create_prediction_summary_table(prediction_results: Dict[str, Any]) -> pd.DataFrame:
    """
    创建预测结果汇总表

    Args:
        prediction_results: 预测结果字典

    Returns:
        包含预测结果的DataFrame
    """
    if not prediction_results:
        return pd.DataFrame()

    predictions = prediction_results['predictions']
    dates = prediction_results['dates']
    current_price = prediction_results['current_price']
    task_type = prediction_results['task_type']

    summary_data = []

    for i, (pred, date) in enumerate(zip(predictions, dates)):
        if task_type == 'regression':
            predicted_price = current_price * (1 + pred / 100)
            price_change = predicted_price - current_price
            change_pct = (price_change / current_price) * 100
            direction = "Up" if change_pct > 0 else "Down" if change_pct < 0 else "Sideways"

            summary_data.append({
                '预测天数': i + 1,
                '日期': date.strftime('%Y-%m-%d'),
                '预测价格': f"{predicted_price:.2f}",
                '价格变化': f"{price_change:+.2f}",
                '变化百分比': f"{change_pct:+.2f}%",
                '趋势方向': direction
            })
        else:
            trend_names = {0: "Down", 1: "Sideways", 2: "Up"}
            trend_name = trend_names.get(pred, f"未知({pred})")

            summary_data.append({
                '预测天数': i + 1,
                '日期': date.strftime('%Y-%m-%d'),
                '预测趋势': trend_name,
                '预测类别': pred
            })

    return pd.DataFrame(summary_data)