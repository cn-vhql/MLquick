#!/usr/bin/env python3
"""
主界面模块 - 负责Streamlit界面和用户交互
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.patches as patches
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_fetcher import get_futures_data, get_supported_futures_symbols, validate_futures_symbol
from data_processor import calculate_technical_indicators, validate_data_quality, get_feature_importance_data
from model_trainer import train_complete_workflow, plot_model_comparison, plot_prediction_scatter, plot_confusion_matrix, evaluate_model_performance, get_feature_importance_from_model
from model_predictor import predict_future_trend, plot_prediction_results, generate_prediction_report, create_prediction_summary_table

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.unicode_minus'] = False

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = {}
if 'best_model' not in st.session_state:
    st.session_state.best_model = None


def plot_matplotlib_candlestick(df, title="Candlestick Chart"):
    """使用Matplotlib绘制K线图"""
    try:
        # 设置字体配置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 确保数据按日期排序
        df = df.sort_index().reset_index()
        df = df.rename(columns={'index': 'date'})

        # 限制显示最近200天数据以提高性能
        if len(df) > 200:
            df = df.tail(200)
            st.info(f"数据量较大，仅显示最近200天数据")

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10),
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       sharex=True)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # 转换日期格式
        dates = mdates.date2num(df['date'])

        # 绘制K线
        for i in range(len(df)):
            date = dates[i]
            open_price = df.iloc[i]['open']
            high_price = df.iloc[i]['high']
            low_price = df.iloc[i]['low']
            close_price = df.iloc[i]['close']

            # 设置颜色：红色上涨，绿色下跌（中国期货市场惯例）
            color = 'red' if close_price >= open_price else 'green'

            # 绘制上下影线
            ax1.plot([date, date], [low_price, high_price], color=color, linewidth=1)

            # 绘制实体
            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)

            rect = patches.Rectangle((date - 0.3, bottom), 0.6, height,
                                   facecolor=color, edgecolor=color, alpha=0.8)
            ax1.add_patch(rect)

        # 绘制移动平均线
        if 'MA5' in df.columns and not df['MA5'].isnull().all():
            ax1.plot(dates, df['MA5'], 'b-', linewidth=1.5, label='MA5', alpha=0.8)

        if 'MA10' in df.columns and not df['MA10'].isnull().all():
            ax1.plot(dates, df['MA10'], 'orange', linewidth=1.5, label='MA10', alpha=0.8)

        if 'MA20' in df.columns and not df['MA20'].isnull().all():
            ax1.plot(dates, df['MA20'], 'purple', linewidth=1.5, label='MA20', alpha=0.8)

        # 设置价格图表格式
        ax1.set_title('Price Trend', fontsize=14)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 绘制成交量
        for i in range(len(df)):
            date = dates[i]
            volume = df.iloc[i]['volume']
            close_price = df.iloc[i]['close']
            open_price = df.iloc[i]['open']

            # 成交量颜色对应K线
            color = 'red' if close_price >= open_price else 'green'

            ax2.bar(date, volume, width=0.6, color=color, alpha=0.8)

        # 设置成交量图表格式
        ax2.set_title('Volume', fontsize=14)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # 设置x轴日期格式
        ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # 调整布局
        plt.tight_layout()

        return fig

    except Exception as e:
        st.error(f"Matplotlib图表绘制错误: {str(e)}")
        return None


def render_sidebar():
    """渲染侧边栏"""
    st.sidebar.title("⚙️ 参数配置")

    # 自定义期货代码输入
    symbol = st.sidebar.text_input("输入自定义期货代码", value="")

    st.sidebar.markdown(f"**当前选择**: {symbol}")

    # 时间范围选择
    st.sidebar.subheader("📅 时间范围")
    use_date_range = st.sidebar.checkbox("自定义日期范围", value=False)

    if use_date_range:
        start_date = st.sidebar.date_input("开始日期", datetime.now() - timedelta(days=90))
        end_date = st.sidebar.date_input("结束日期", datetime.now())
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        days = None
    else:
        days = st.sidebar.slider("获取最近天数", min_value=30, max_value=365, value=90)
        start_date_str = None
        end_date_str = None

    # 预测参数
    st.sidebar.subheader("🔮 预测参数")
    historical_days = st.sidebar.slider("历史数据天数", min_value=5, max_value=30, value=7)
    prediction_days = st.sidebar.slider("预测天数", min_value=1, max_value=15, value=1)
    train_size = st.sidebar.slider("训练集比例", min_value=0.6, max_value=0.9, value=0.7)

    # 预测类型
    task_type = st.sidebar.radio(
        "预测类型",
        options=["分类预测","回归预测"],
        index=0,
        help="分类预测：预测涨跌方向\n回归预测：预测价格变化百分比"
    )
    task_type_value = 'regression' if task_type == "回归预测" else 'classification'

    return {
        'symbol': symbol,
        'days': days,
        'start_date': start_date_str,
        'end_date': end_date_str,
        'historical_days': historical_days,
        'prediction_days': prediction_days,
        'train_size': train_size,
        'task_type': task_type_value
    }


def render_data_preview_tab(df):
    """渲染数据预览标签页"""
    st.header("📊 原始数据")

    if df is not None and len(df) > 0:
        # 数据基本信息
        st.subheader("数据基本信息")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("数据条数", len(df))
        with col2:
            st.metric("开始日期", df.index.min().strftime('%Y-%m-%d'))
        with col3:
            st.metric("结束日期", df.index.max().strftime('%Y-%m-%d'))
        with col4:
            latest_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2] if len(df) > 1 else latest_price
            price_change = latest_price - prev_price
            price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
            st.metric("最新价格", f"{latest_price:.2f}", f"{price_change_pct:+.2f}%")

        # 数据预览
        st.subheader("数据预览")
        st.dataframe(df)

        # 数据质量报告
        st.subheader("数据质量报告")
        quality_info = validate_data_quality(df)

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**完整性**: {quality_info['completeness']:.1f}%")
            st.write(f"**缺失值**: {quality_info['missing_values']}")
            st.write(f"**重复行**: {quality_info['duplicate_rows']}")

        with col2:
            st.write(f"**总行数**: {quality_info['total_rows']}")
            st.write(f"**数值列数**: {len(quality_info['numeric_columns'])}")
            st.write(f"**总列数**: {len(quality_info['columns'])}")

    else:
        st.warning("暂无数据，请先获取期货数据")


def render_price_chart_tab(df):
    """渲染价格图表标签页"""
    st.header("📈 价格走势图")

    if df is not None and len(df) > 0:
        # 绘制K线图
        fig = plot_matplotlib_candlestick(df, f"Futures Candlestick Chart")
        if fig:
            st.pyplot(fig)
            plt.close()

        # 数据摘要
        st.subheader("数据摘要")
        latest_data = df.iloc[-1]
        first_data = df.iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("开盘价", f"{latest_data['open']:.2f}")
        with col2:
            st.metric("收盘价", f"{latest_data['close']:.2f}")
        with col3:
            st.metric("最高价", f"{latest_data['high']:.2f}")
        with col4:
            st.metric("最低价", f"{latest_data['low']:.2f}")

        # 期间涨跌幅
        total_change = (latest_data['close'] - first_data['close']) / first_data['close'] * 100
        st.metric("期间涨跌幅", f"{total_change:+.2f}%")

    else:
        st.warning("暂无数据，请先获取期货数据")


def render_feature_engineering_tab(df, params):
    """渲染特征工程标签页"""
    st.header("🔧 特征工程")

    if df is not None and len(df) > 0:
        # 计算技术指标
        processed_df = calculate_technical_indicators(df)
        st.session_state.processed_data = processed_df

        # 技术指标预览
        st.subheader("技术指标预览")
        indicator_cols = ['MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'Signal', 'BB_upper', 'BB_middle', 'BB_lower']
        available_indicators = [col for col in indicator_cols if col in processed_df.columns]

        if available_indicators:
            st.dataframe(processed_df[available_indicators].tail(20))
        else:
            st.warning("无法计算技术指标，数据可能不足")

        # 技术指标统计
        st.subheader("技术指标统计")
        if available_indicators:
            st.write(processed_df[available_indicators].describe())

    else:
        st.warning("暂无数据，请先获取期货数据")


def render_model_training_tab(params):
    """渲染模型训练标签页"""
    st.header("🤖 模型训练与预测")

    if st.session_state.processed_data is None:
        st.warning("请先完成数据处理步骤")
        return

    # 训练按钮
    if st.button("🚀 开始训练模型", type="primary"):
        with st.spinner("正在训练模型..."):
            try:
                results = train_complete_workflow(
                    st.session_state.processed_data,
                    historical_days=params['historical_days'],
                    prediction_days=params['prediction_days'],
                    task_type=params['task_type'],
                    train_size=params['train_size']
                )

                if results:
                    st.session_state.training_results = results
                    st.session_state.best_model = results['best_model']
                    st.success("模型训练完成！")
                else:
                    st.error("模型训练失败")
            except Exception as e:
                st.error(f"训练过程出错: {str(e)}")

    # 显示训练结果
    if st.session_state.training_results:
        results = st.session_state.training_results

        # 模型性能对比
        st.subheader("📊 模型性能对比")
        performance_df = evaluate_model_performance(results['results'], results['task_type'])
        if not performance_df.empty:
            st.dataframe(performance_df, use_container_width=True)

        # 预测结果可视化
        if results['task_type'] == 'regression':
            st.subheader("📈 预测散点图")
            for model_name, result in results['results'].items():
                if result is not None:
                    fig = plot_prediction_scatter(
                        results['y_test'], result['predictions'], model_name
                    )
                    if fig:
                        st.pyplot(fig)
                        plt.close()
                        break  # 只显示最佳模型的图
        else:
            st.subheader("🎯 混淆矩阵")
            for model_name, result in results['results'].items():
                if result is not None:
                    metrics = result['metrics']
                    fig = plot_confusion_matrix(
                        metrics['confusion_matrix'],
                        metrics['class_names'],
                        model_name
                    )
                    if fig:
                        st.pyplot(fig)
                        plt.close()
                        break  # 只显示最佳模型的图


def render_feature_importance_tab():
    """渲染特征重要性标签页"""
    st.header("📊 特征重要性分析")

    if not st.session_state.training_results:
        st.warning("请先完成模型训练")
        return

    results = st.session_state.training_results

    if results['best_model'] is not None:
        # 获取特征重要性
        feature_names = list(results['X'].columns)
        importance_df = get_feature_importance_from_model(results['best_model'], feature_names)

        if not importance_df.empty:
            # 按基础特征分组
            importance_grouped = get_feature_importance_data(results['X'], importance_df['Importance'].values)

            st.subheader("特征重要性排名 (Top 20)")
            if len(importance_grouped) > 0:
                st.dataframe(importance_grouped.head(20), use_container_width=True)

                # # 绘制特征重要性图
                # fig, ax = plt.subplots(figsize=(12, 8))
                # top_features = importance_grouped.head(15)
                # bars = ax.barh(range(len(top_features)), top_features['Importance'])
                # ax.set_yticks(range(len(top_features)))
                # ax.set_yticklabels(top_features['Feature'])
                # ax.set_xlabel('Importance')
                # ax.set_title('Feature Importance Analysis')
                # ax.grid(True, alpha=0.3)

                ## 添加数值标签
                # for i, bar in enumerate(bars):
                #     width = bar.get_width()
                #     ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                #            f'{width:.3f}', ha='left', va='center')

                # plt.tight_layout()
                # st.pyplot(fig)
                # plt.close()
            else:
                st.warning("无法获取特征重要性信息")
        else:
            st.warning("该模型不支持特征重要性分析")
    else:
        st.warning("没有可用的最佳模型")


def render_future_prediction_tab(params):
    """渲染未来预测标签页"""
    st.header("🔮 未来预测报告")

    if st.session_state.best_model is None:
        st.warning("请先完成模型训练")
        return

    if st.session_state.processed_data is None:
        st.warning("请先完成数据处理")
        return

    # 预测参数控制
    st.subheader("预测参数设置")
    col1, col2, col3 = st.columns(3)

    with col1:
        pred_days = st.slider("预测天数", min_value=1, max_value=15, value=params['prediction_days'])
    with col2:
        show_confidence = st.checkbox("显示置信区间", value=True)
    # with col3:
    #     generate_report = st.checkbox("生成详细报告", value=True)

    # 生成预测按钮
    if st.button("🔮 生成未来预测", type="primary"):
        with st.spinner("正在生成未来预测..."):
            try:
                prediction_results = predict_future_trend(
                    st.session_state.best_model,
                    st.session_state.processed_data,
                    historical_days=params['historical_days'],
                    prediction_days=pred_days,
                    task_type=params['task_type']
                )

                if prediction_results:
                    st.session_state.prediction_results = prediction_results
                    st.success("预测生成完成！")
                else:
                    st.error("预测生成失败")
            except Exception as e:
                st.error(f"预测过程出错: {str(e)}")

    # 显示预测结果
    if 'prediction_results' in st.session_state and st.session_state.prediction_results:
        pred_results = st.session_state.prediction_results

        # 预测摘要
        st.subheader("📋 预测摘要")
        current_price = pred_results['current_price']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("当前价格", f"{current_price:.2f}")
        with col2:
            if pred_results['task_type'] == 'regression':
                final_prediction = pred_results['predictions'][-1]
                final_price = current_price * (1 + final_prediction / 100)
                total_change = final_prediction
                st.metric("预测总变化", f"{total_change:+.2f}%")
            else:
                trend_names = {0: "Down", 1: "Sideways", 2: "Up"}
                most_common = max(set(pred_results['predictions']), key=pred_results['predictions'].count)
                st.metric("主要趋势", trend_names.get(most_common, "未知"))

        with col3:
            if pred_results['predictions']:
                confidence = min(0.9, max(0.1, 1 - np.std(pred_results['predictions']) / (np.mean(np.abs(pred_results['predictions'])) + 1e-6)))
                st.metric("预测置信度", f"{confidence:.2f}")

        # 预测数据表
        st.subheader("📊 详细预测数据")
        summary_df = create_prediction_summary_table(pred_results)
        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True)

        # 可视化预测结果
        st.subheader("📈 预测趋势图")
        fig = plot_prediction_results(
            pred_results['historical_data'],
            pred_results['predictions'],
            pred_results['dates'],
            pred_results['task_type'],
            show_confidence
        )
        if fig:
            st.pyplot(fig)
            plt.close()

def main():
    """主函数"""
    # 设置页面配置
    st.set_page_config(
        page_title="期货行情预测平台",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 页面标题
    st.title("🚀 期货行情预测平台")
    st.markdown("---")

    # 渲染侧边栏
    params = render_sidebar()

    # 获取数据按钮
    if st.sidebar.button("📊 获取期货数据", type="primary"):
        with st.spinner("正在获取期货数据..."):
            try:
                # 验证期货代码
                if not validate_futures_symbol(params['symbol']):
                    st.error(f"期货代码 {params['symbol']} 可能无效，请检查后重试")
                else:
                    data = get_futures_data(
                        symbol=params['symbol'],
                        days=params['days'],
                        start_date=params['start_date'],
                        end_date=params['end_date']
                    )
                    if data is not None and len(data) > 0:
                        st.session_state.data = data
                        st.success(f"成功获取 {params['symbol']} 的期货数据")
                    else:
                        st.error("获取期货数据失败")
            except Exception as e:
                st.error(f"数据获取出错: {str(e)}")

    # 主标签页
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 原始数据",
        "📈 价格走势图",
        "🔧 特征工程",
        "🤖 模型训练与预测",
        "📊 特征重要性",
        "🔮 未来预测报告"
    ])

    with tab1:
        render_data_preview_tab(st.session_state.data)

    with tab2:
        render_price_chart_tab(st.session_state.data)

    with tab3:
        render_feature_engineering_tab(st.session_state.data, params)

    with tab4:
        render_model_training_tab(params)

    with tab5:
        render_feature_importance_tab()

    with tab6:
        render_future_prediction_tab(params)

    # 页脚
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 12px;'>
        ⚠️ 投资风险提示：本平台预测结果仅供参考，不构成投资建议。期货市场风险较大，请谨慎投资。
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()