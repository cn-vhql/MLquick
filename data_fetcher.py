#!/usr/bin/env python3
"""
数据获取模块 - 负责从akshare获取期货数据
"""
import pandas as pd
import akshare as ak
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional


def get_futures_data(symbol: str, days: int = 90, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    获取期货历史数据

    Args:
        symbol: 期货品种代码 (如 'CU0')
        days: 获取天数 (默认90天)
        start_date: 开始日期 (格式: 'YYYY-MM-DD')
        end_date: 结束日期 (格式: 'YYYY-MM-DD')

    Returns:
        包含期货历史数据的DataFrame
    """
    try:
        with st.spinner(f'正在获取{symbol}的期货数据...'):
            # 获取原始数据
            df = ak.futures_main_sina(symbol=symbol)

            if df is None or df.empty:
                st.error(f"无法获取{symbol}的数据，数据为空")
                return pd.DataFrame()

            # 显示数据获取信息
            st.info(f"成功获取{symbol}原始数据: {df.shape[0]}行 x {df.shape[1]}列")

            # 处理不同的日期列名
            date_columns = ['date', '日期', 'datetime', 'time', '时间']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break

            if date_col is None:
                st.error(f"无法找到日期列，可用列名: {list(df.columns)}")
                return pd.DataFrame()

            # 重命名日期列为'date'
            if date_col != 'date':
                df = df.rename(columns={date_col: 'date'})

            # 确保日期列是datetime类型
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

            # 删除日期为NaT的行
            df = df.dropna(subset=['date'])

            if len(df) == 0:
                st.error("数据中没有有效的日期")
                return pd.DataFrame()

            # 日期范围过滤
            if start_date and end_date:
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            else:
                # 按日期排序并获取最近N天
                df = df.sort_values('date')
                df = df.tail(days)

            # 数据清洗和预处理
            df = df.dropna()

            # 处理价格列名（支持中英文）
            column_mapping = {
                # 日期列（已处理）
                'date': 'date',
                '日期': 'date',
                'datetime': 'date',
                '时间': 'date',

                # 价格列
                'open': 'open', '开盘': 'open', '开盘价': 'open',
                'high': 'high', '最高': 'high', '最高价': 'high',
                'low': 'low', '最低': 'low', '最低价': 'low',
                'close': 'close', '收盘': 'close', '收盘价': 'close',

                # 成交量列
                'volume': 'volume', '成交量': 'volume', 'vol': 'volume',

                # 持仓量列
                'hold': 'hold', '持仓量': 'hold', '持仓': 'hold',
                'open_interest': 'hold', '持仓量': 'hold',

                # 结算价列
                'settlement': 'settlement', '结算': 'settlement', '结算价': 'settlement',
                '动态结算价': 'settlement'
            }

            # 重命名列
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns and old_name != new_name:
                    df = df.rename(columns={old_name: new_name})

            # 处理后的列信息

            # 确保必要的列都存在
            required_columns = ['date', 'open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.error(f"数据中缺少必要的列: {missing_columns}")
                st.error(f"可用列名: {list(df.columns)}")
                return pd.DataFrame()

            # 处理成交量列（可选）
            if 'volume' not in df.columns:
                df['volume'] = 0
                st.warning("数据中没有成交量信息，已设置为0")

            # 确保数值列的数据类型正确
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 删除含有NaN的行
            df = df.dropna(subset=required_columns)

            if len(df) == 0:
                st.error("数据清洗后没有有效数据")
                return pd.DataFrame()

            # 设置日期为索引
            df = df.set_index('date')

            # 按日期排序
            df = df.sort_index()

            st.success(f'成功获取{len(df)}条数据')
            return df

    except Exception as e:
        st.error(f'获取期货数据失败: {str(e)}')
        return pd.DataFrame()


def validate_futures_symbol(symbol: str) -> bool:
    """
    验证期货品种代码是否有效

    Args:
        symbol: 期货品种代码

    Returns:
        bool: 代码是否有效
    """
    try:
        # 尝试获取少量数据来验证代码
        df = ak.futures_main_sina(symbol=symbol)
        return len(df) > 0
    except:
        return False


def get_futures_info(symbol: str) -> dict:
    """
    获取期货品种信息

    Args:
        symbol: 期货品种代码

    Returns:
        包含期货信息的字典
    """
    try:
        df = ak.futures_main_sina(symbol=symbol)
        if len(df) > 0:
            latest_data = df.iloc[-1]
            return {
                'symbol': symbol,
                'latest_price': latest_data.get('close', 0),
                'latest_date': latest_data.get('date', ''),
                'data_count': len(df)
            }
        else:
            return {'error': f'无法获取{symbol}的数据'}
    except Exception as e:
        return {'error': f'获取{symbol}信息失败: {str(e)}'}


def get_supported_futures_symbols() -> dict:
    """
    获取支持的期货品种列表

    Returns:
        包含期货品种代码和名称的字典
    """
    return {
        '沪铜主力': 'CU0',
        '沪铝主力': 'AL0',
        '沪锌主力': 'ZN0',
        '沪镍主力': 'NI0',
        '沪锡主力': 'SN0',
        '沪铅主力': 'PB0',
        '螺纹钢主力': 'RB0',
        '热轧卷板主力': 'HC0',
        '铁矿石主力': 'I0',
        '焦炭主力': 'J0',
        '焦煤主力': 'JM0',
        '动力煤主力': 'ZC0',
        '原油主力': 'SC0',
        '燃油主力': 'FU0',
        '沥青主力': 'BU0',
        'PTA主力': 'TA0',
        '甲醇主力': 'MA0',
        'PVC主力': 'V0',
        'PP主力': 'PP0',
        'LLDPE主力': 'L0',
        '豆粕主力': 'M0',
        '豆油主力': 'Y0',
        '棕榈油主力': 'P0',
        '玉米主力': 'C0',
        '淀粉主力': 'CS0',
        '棉花主力': 'CF0',
        '白糖主力': 'SR0',
        '苹果主力': 'AP0',
        '鸡蛋主力': 'JD0',
        '黄金主力': 'AU0',
        '白银主力': 'AG0',
        '沪深300股指': 'IF0',
        '上证50股指': 'IH0',
        '中证500股指': 'IC0',
        '十年国债': 'T0',
        '五年国债': 'TF0'
    }