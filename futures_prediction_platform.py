import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Attempt to import st_echarts

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.unicode_minus'] = False

# Matplotlib candlestick chart function
def plot_matplotlib_candlestick(df, title="Candlestick Chart"):
    """Draw candlestick chart using matplotlib"""
    try:
        from matplotlib import dates as mdates
        from matplotlib.dates import DateFormatter
        import matplotlib.patches as patches

        # Set font configuration
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # Ensure data is sorted by date
        df = df.sort_values('date').reset_index(drop=True)

        # Create chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10),
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       sharex=True)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Convert date format
        dates = mdates.date2num(df['date'])

        # Draw candlesticks
        for i in range(len(df)):
            date = dates[i]
            open_price = df.iloc[i]['open']
            high_price = df.iloc[i]['high']
            low_price = df.iloc[i]['low']
            close_price = df.iloc[i]['close']

            # Set colors: red for up, green for down (Chinese futures market convention)
            color = 'red' if close_price >= open_price else 'green'

            # Draw upper and lower shadows
            ax1.plot([date, date], [low_price, high_price], color=color, linewidth=1)

            # Draw body
            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)

            rect = patches.Rectangle((date - 0.3, bottom), 0.6, height,
                                   facecolor=color, edgecolor=color, alpha=0.8)
            ax1.add_patch(rect)

        # Draw moving averages
        if 'MA5' in df.columns and not df['MA5'].isnull().all():
            ax1.plot(dates, df['MA5'], 'b-', linewidth=1.5, label='MA5', alpha=0.8)

        if 'MA10' in df.columns and not df['MA10'].isnull().all():
            ax1.plot(dates, df['MA10'], 'orange', linewidth=1.5, label='MA10', alpha=0.8)

        if 'MA20' in df.columns and not df['MA20'].isnull().all():
            ax1.plot(dates, df['MA20'], 'purple', linewidth=1.5, label='MA20', alpha=0.8)

        # Set price chart format
        ax1.set_title('Price Trend', fontsize=14)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Draw volume
        for i in range(len(df)):
            date = dates[i]
            volume = df.iloc[i]['volume']
            close_price = df.iloc[i]['close']
            open_price = df.iloc[i]['open']

            # Volume color corresponds to candlestick
            color = 'red' if close_price >= open_price else 'green'

            ax2.bar(date, volume, width=0.6, color=color, alpha=0.8)

        # Set volume chart format
        ax2.set_title('Volume', fontsize=14)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Set x-axis date format
        ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # Adjust layout
        plt.tight_layout()

        return fig

    except Exception as e:
        st.error(f"matplotlib chart plotting error: {str(e)}")
        return None

# ECharts candlestick chart function
def plot_echarts_candlestick(df, title="Candlestick Chart"):
    """Draw professional candlestick chart using ECharts"""
    # Ensure data is sorted by date
    df = df.sort_values('date').reset_index(drop=True)

    # Prepare candlestick data
    kline_data = []
    for i in range(len(df)):
        kline_data.append([
            float(df.iloc[i]['open']),
            float(df.iloc[i]['close']),
            float(df.iloc[i]['low']),
            float(df.iloc[i]['high'])
        ])

    # Prepare x-axis data (dates)
    xaxis_data = [df.iloc[i]['date'].strftime('%m-%d') for i in range(len(df))]

    # Prepare volume data
    volume_data = []
    for i in range(len(df)):
        volume_value = int(df.iloc[i]['volume']) if pd.notna(df.iloc[i]['volume']) else 0
        volume_data.append({
            'value': volume_value,
            'itemStyle': {
                'color': '#ef232a' if df.iloc[i]['close'] > df.iloc[i]['open'] else '#14b143'
            }
        })

    # Prepare moving average data
    series_list = [
        {
            'name': 'Candlestick',
            'type': 'candlestick',
            'data': kline_data,
            'xAxisIndex': 0,
            'yAxisIndex': 0,
            'itemStyle': {
                'color': '#ef232a',      # Bullish color (red)
                'color0': '#14b143',     # Bearish color (green)
                'borderColor': '#ef232a',
                'borderColor0': '#14b143'
            }
        }
    ]

    # Add moving averages
    if 'MA5' in df.columns and not df['MA5'].isnull().all():
        ma5_data = []
        for x in df['MA5']:
            if pd.notna(x):
                ma5_data.append(float(x))
            else:
                ma5_data.append('-')  # Use '-' instead of None
        series_list.append({
            'name': 'MA5',
            'type': 'line',
            'data': ma5_data,
            'smooth': True,
            'lineStyle': {
                'color': '#3982c4',
                'width': 1
            },
            'xAxisIndex': 0,
            'yAxisIndex': 0
        })

    if 'MA10' in df.columns and not df['MA10'].isnull().all():
        ma10_data = []
        for x in df['MA10']:
            if pd.notna(x):
                ma10_data.append(float(x))
            else:
                ma10_data.append('-')  # Use '-' instead of None
        series_list.append({
            'name': 'MA10',
            'type': 'line',
            'data': ma10_data,
            'smooth': True,
            'lineStyle': {
                'color': '#f67317',
                'width': 1
            },
            'xAxisIndex': 0,
            'yAxisIndex': 0
        })

    if 'MA20' in df.columns and not df['MA20'].isnull().all():
        ma20_data = []
        for x in df['MA20']:
            if pd.notna(x):
                ma20_data.append(float(x))
            else:
                ma20_data.append('-')  # Use '-' instead of None
        series_list.append({
            'name': 'MA20',
            'type': 'line',
            'data': ma20_data,
            'smooth': True,
            'lineStyle': {
                'color': '#8a2be2',
                'width': 1
            },
            'xAxisIndex': 0,
            'yAxisIndex': 0
        })

    # Add volume
    series_list.append({
        'name': 'Volume',
        'type': 'bar',
        'data': volume_data,
        'xAxisIndex': 1,
        'yAxisIndex': 1
    })

    # ECharts configuration
    option = {
        'title': {
            'text': title,
            'left': 'center',
            'textStyle': {
                'fontSize': 16,
                'fontWeight': 'bold'
            }
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'cross'
            }
        },
        'legend': {
            'data': ['Candlestick', 'MA5', 'MA10', 'MA20', 'Volume'],
            'top': '8%'
        },
        'grid': [
            {
                'left': '10%',
                'right': '8%',
                'height': '60%',
                'top': '15%'
            },
            {
                'left': '10%',
                'right': '8%',
                'top': '80%',
                'height': '15%'
            }
        ],
        'xAxis': [
            {
                'type': 'category',
                'data': xaxis_data,
                'scale': True,
                'boundaryGap': False,
                'axisLine': {'onZero': False},
                'splitLine': {'show': False},
                'min': 'dataMin',
                'max': 'dataMax'
            },
            {
                'type': 'category',
                'gridIndex': 1,
                'data': xaxis_data,
                'scale': True,
                'boundaryGap': False,
                'axisLine': {'onZero': False},
                'axisTick': {'show': False},
                'splitLine': {'show': False},
                'axisLabel': {'show': False},
                'min': 'dataMin',
                'max': 'dataMax'
            }
        ],
        'yAxis': [
            {
                'scale': True,
                'splitArea': {
                    'show': True
                }
            },
            {
                'scale': True,
                'gridIndex': 1,
                'splitNumber': 2,
                'axisLabel': {'show': False},
                'axisLine': {'show': False},
                'axisTick': {'show': False},
                'splitLine': {'show': False}
            }
        ],
        'dataZoom': [
            {
                'type': 'inside',
                'xAxisIndex': [0, 1],
                'start': 50,
                'end': 100
            },
            {
                'show': True,
                'xAxisIndex': [0, 1],
                'type': 'slider',
                'top': '92%',
                'start': 50,
                'end': 100
            }
        ],
        'series': series_list
    }

    return option



# Set page configuration
st.set_page_config(
    page_title="Futures Price Prediction Platform",
    page_icon="📈",
    layout="wide"
)

# Futures data retrieval function
def get_futures_data(symbol, period="daily", start_date=None, end_date=None, days=30):
    """
    Retrieve futures market data
    :param symbol: Futures contract code, e.g., 'CU0' for copper main contract
    :param period: Period, 'daily', 'weekly', 'monthly'
    :param start_date: Start date, format YYYYMMDD
    :param end_date: End date, format YYYYMMDD
    :param days: Default number of days to retrieve when start_date and end_date are None
    """
    try:
        if period == "daily":
            # Retrieve data
            try:
                df = ak.futures_main_sina(symbol=symbol)
            except Exception as e:
                st.error(f"Failed to fetch data from akshare: {str(e)}")
                st.warning("Please check:")
                st.write("1. Network connection")
                st.write("2. Contract code validity")
                st.write("3. Data source availability")
                return None

            if df is None or df.empty:
                st.error(f"Cannot retrieve data for {symbol}")
                st.warning("This could be due to:")
                st.write("1. Invalid futures contract code")
                st.write("2. Contract has been delisted or suspended")
                st.write("3. Data source is temporarily unavailable")
                st.info("Try common contract codes like: CU0 (Copper), RB0 (Rebar), AU0 (Gold)")
                return None

            # Standardize column names for easier processing
            # Handle both Chinese and English column names
            column_mapping = {
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'open_interest': 'open_interest',
                'settlement': 'settlement',
                # Chinese column names
                '日期': 'date',
                '开盘价': 'open',
                '最高价': 'high',
                '最低价': 'low',
                '收盘价': 'close',
                '成交量': 'volume',
                '持仓量': 'open_interest',
                '动态结算价': 'settlement'
            }

            df = df.rename(columns=column_mapping)

            # Ensure date column is datetime type
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            else:
                st.error("Incorrect data format, date column not found")
                st.write("**Debugging Information:**")
                st.write("Available columns in data:", list(df.columns))
                st.write("Data shape:", df.shape)
                st.write("First few rows:")
                st.dataframe(df.head())
                st.info("The data source may have changed format. Please check the available columns above.")
                return None

            # Filter by date range if specified
            if start_date is not None and end_date is not None:
                start_dt = pd.to_datetime(start_date, format='%Y%m%d')
                end_dt = pd.to_datetime(end_date, format='%Y%m%d')

                # Filter data by date range
                mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
                filtered_df = df[mask].copy()

                if filtered_df.empty:
                    st.warning(f"No data found for {symbol} between {start_date} and {end_date}")
                    st.info("Showing recent data for reference")
                    return df.tail(days)

                # Sort by date
                filtered_df = filtered_df.sort_values('date').reset_index(drop=True)
                return filtered_df
            else:
                # If no date range specified, return recent data
                return df.tail(days)

        return None
    except Exception as e:
        st.error(f"Failed to retrieve data: {e}")
        return None

# Technical indicators calculation
def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    df = df.copy()

    # Handle column name mapping (both Chinese and English)
    column_mapping = {
        'date': 'date',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'open_interest': 'open_interest',
        'settlement': 'settlement',
        # Chinese column names
        '日期': 'date',
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
        '成交量': 'volume',
        '持仓量': 'open_interest',
        '动态结算价': 'settlement'
    }

    df = df.rename(columns=column_mapping)

    # Ensure correct data types
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])

    # Moving averages
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Histogram'] = df['MACD'] - df['Signal']

    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)

    # Price change rates
    df['price_change'] = df['close'].pct_change()
    df['price_change_3d'] = df['close'].pct_change(3)
    df['price_change_5d'] = df['close'].pct_change(5)

    return df

# Generate features and target variables
def create_features_targets(df, historical_days=7, prediction_days=3, task_type='regression'):
    """
    Generate features and target variables
    :param df: Original data
    :param historical_days: Number of historical data days
    :param prediction_days: Number of prediction days
    :param task_type: 'regression' or 'classification'
    """
    df_features = calculate_technical_indicators(df)

    # Create features: historical price and technical indicators
    feature_list = []
    target_list = []

    for i in range(historical_days, len(df_features) - prediction_days):
        # Historical features
        features = []
        historical_data = df_features.iloc[i-historical_days:i]

        # Price features
        features.extend(historical_data['close'].values)
        features.extend(historical_data['volume'].values)
        features.extend(historical_data['MA5'].values[-5:])
        features.extend(historical_data['MA10'].values[-5:])
        features.extend(historical_data['RSI'].values[-5:])
        features.extend(historical_data['MACD'].values[-5:])

        # Add current price information
        current_price = df_features.iloc[i]['close']
        features.append(current_price)

        feature_list.append(features)

        # Target variable
        future_price = df_features.iloc[i + prediction_days]['close']
        if task_type == 'regression':
            # Regression task: predict price change percentage
            target = (future_price - current_price) / current_price * 100
        else:
            # Classification task: predict up/sideways/down
            price_change = (future_price - current_price) / current_price * 100
            if price_change > 1.0:  # Price increase > 1%
                target = 2  # Up
            elif price_change < -1.0:  # Price decrease > 1%
                target = 0  # Down
            else:
                target = 1  # Sideways

        target_list.append(target)

    # Create feature names
    feature_names = []
    for i in range(historical_days):
        feature_names.extend([f'close_{i}', f'volume_{i}'])
    feature_names.extend([f'MA5_{i}' for i in range(5)])
    feature_names.extend([f'MA10_{i}' for i in range(5)])
    feature_names.extend([f'RSI_{i}' for i in range(5)])
    feature_names.extend([f'MACD_{i}' for i in range(5)])
    feature_names.append('current_price')

    X = pd.DataFrame(feature_list, columns=feature_names)
    y = pd.Series(target_list)

    # Handle missing values
    X = X.dropna()
    y = y[X.index]  # Keep X and y consistent

    # Reset index
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y


# Regression prediction model
def regression_prediction(X, y, train_size=0.7):
    """Use multiple regression models for prediction"""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.svm import SVR
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt

    # Split training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    # Define models - using models that can better handle data
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    best_model = None
    best_score = -float('inf')

    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)
            # Predict
            y_pred = model.predict(X_test)
            # Evaluate
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            results[name] = {'r2': r2, 'mse': mse, 'model': model}

            if r2 > best_score:
                best_score = r2
                best_model = model

        except Exception as e:
            st.warning(f"Model {name} training failed: {e}")

    return results, best_model, X_test, y_test

# Classification prediction model
def classification_prediction(X, y, train_size=0.7):
    """Use multiple classification models for prediction"""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import seaborn as sns

    # Split training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    # Define models - using models that can better handle data
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }

    results = {}
    best_model = None
    best_score = 0

    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)
            # Predict
            y_pred = model.predict(X_test)
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {'accuracy': accuracy, 'model': model, 'predictions': y_pred}

            if accuracy > best_score:
                best_score = accuracy
                best_model = model

        except Exception as e:
            st.warning(f"Model {name} training failed: {e}")

    return results, best_model, X_test, y_test

# Visualization function
def plot_predictions(y_true, y_pred, title):
    """Plot prediction results comparison chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, color='blue', s=30)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Values (%)', fontsize=12)
    ax.set_ylabel('Predicted Values (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# Main interface
def main():
    st.title("📈 Futures Price Prediction Platform")

    # Sidebar parameter settings
    st.sidebar.header("Parameter Settings")

    # Futures contract selection - user custom input
    st.sidebar.subheader("📊 Futures Contract Selection")

    # User input for futures contract code
    futures_symbol = st.sidebar.text_input(
        "🔤 Enter Futures Contract Code",
        value="CU0",
        help="Please enter futures contract code, e.g., CU0 (Copper main), RB0 (Rebar main), etc."
    ).upper().strip()

    # Add common contract suggestions
    with st.sidebar.expander("📋 Common Contract Codes", expanded=False):
        st.write("**Metals:**")
        st.code("CU0 - Copper Main")
        st.code("AL0 - Aluminum Main")
        st.code("ZN0 - Zinc Main")
        st.code("AU0 - Gold Main")
        st.code("AG0 - Silver Main")

        st.write("**Energy & Chemicals:**")
        st.code("SC0 - Crude Oil Main")
        st.code("FU0 - Fuel Oil Main")
        st.code("TA0 - PTA Main")

        st.write("**Agriculture:**")
        st.code("M0 - Soybean Meal Main")
        st.code("Y0 - Soybean Oil Main")
        st.code("C0 - Corn Main")
        st.code("CF0 - Cotton Main")

    # Market data time range selection
    st.sidebar.subheader("📅 Market Data Time Range")

    # Date range selection
    today = datetime.now().date()
    default_start = today - timedelta(days=90)

    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start,
        max_value=today,
        help="Select the start date for market data"
    )

    end_date = st.sidebar.date_input(
        "End Date",
        value=today,
        max_value=today,
        min_value=start_date,
        help="Select the end date for market data"
    )

    # Calculate number of days
    data_days = (end_date - start_date).days
    if data_days < 30:
        st.sidebar.warning("Data period is less than 30 days, which may affect model training performance")

    st.sidebar.info(f"Selected time range: {data_days} days")

    # Simplified futures contract name mapping (for display only)
    common_futures_names = {
        'CU0': 'Copper Main', 'RB0': 'Rebar Main', 'AU0': 'Gold Main', 'SC0': 'Crude Oil Main'
    }

    selected_future = common_futures_names.get(futures_symbol, f'{futures_symbol} Futures')

    # Validate input
    if not futures_symbol:
        st.sidebar.error("Please enter a futures contract code")
        return

    if len(futures_symbol) < 2:
        st.sidebar.error("Futures contract code format is incorrect, usually 2-3 characters")
        return

    # Historical data days
    historical_days = st.sidebar.slider("Historical Data Days", min_value=3, max_value=30, value=7)

    # Prediction days
    prediction_days = st.sidebar.slider("Prediction Days", min_value=1, max_value=10, value=3)

    # Prediction task type
    task_type = st.sidebar.radio("Prediction Type", ["Regression (Price Change %)", "Classification (Trend Direction)"])

    # Training set ratio
    train_size = st.sidebar.slider("Training Set Ratio", min_value=0.5, max_value=0.9, value=0.7)

    # Main interface
    if st.sidebar.button("Start Prediction"):
        with st.spinner(f"Retrieving {selected_future}({futures_symbol}) data..."):
            # Retrieve data
            start_date_str = start_date.strftime('%Y%m%d')
            end_date_str = end_date.strftime('%Y%m%d')
            df = get_futures_data(futures_symbol, start_date=start_date_str, end_date=end_date_str, days=data_days)

            if df is None:
                st.error(f"❌ Cannot retrieve {selected_future}({futures_symbol}) data")
                st.warning("Possible reasons:")
                st.write("1. Incorrect futures contract code")
                st.write("2. The futures contract has been delisted or suspended")
                st.write("3. Network connection issues")
                st.write("4. Data source temporarily unavailable")
                st.info("💡 Tip: Please check if the futures contract code is correct, or try other contract codes")
                return

            # Display actual retrieved data range
            actual_start = df['date'].min().strftime('%Y-%m-%d')
            actual_end = df['date'].max().strftime('%Y-%m-%d')
            st.success(f"✅ Successfully retrieved {selected_future}({futures_symbol}) data: {len(df)} records")
            st.info(f"📅 Actual data range: {actual_start} to {actual_end}")

            if len(df) <= historical_days + prediction_days:
                st.error(f"❌ Insufficient data! Retrieved {len(df)} records, but need at least {historical_days + prediction_days + 1} records")
                st.warning("Suggested solutions:")
                st.write("1. Increase data retrieval days")
                st.write("2. Reduce historical data days")
                st.write("3. Reduce prediction days")
                return

            # Create main analysis workflow tabs
            main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs([
                "📊 Raw Data",
                "📈 Price Charts",
                "🔧 Feature Engineering",
                "🤖 Model Training & Prediction",
                "📊 Feature Importance"
            ])

            with main_tab1:
                st.subheader("📊 Raw Data Preview")
                st.info(f"Showing the last 10 data records for {selected_future}({futures_symbol})")
                st.dataframe(df.tail(10), width='stretch')

                # Data basic information
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Records", len(df))
                with col2:
                    st.metric("Data Columns", len(df.columns))
                with col3:
                    if len(df) > 0:
                        date_range = (df['date'].max() - df['date'].min()).days
                        st.metric("Time Range", f"{date_range} days")

            with main_tab2:
                st.subheader("📈 Price Charts")

                df_processed = calculate_technical_indicators(df)

                # Display all data within user-selected time range
                df_chart = df_processed.copy()

                # If data is too large, limit to last 200 days for chart performance
                if len(df_chart) > 200:
                    st.info(f"📊 Large dataset ({len(df_chart)} days), for chart performance, showing the last 200 days of data")
                    df_chart = df_chart.tail(200).copy()

                # Display data range information
                actual_start = df_chart['date'].min().strftime('%Y-%m-%d')
                actual_end = df_chart['date'].max().strftime('%Y-%m-%d')
                st.info(f"📅 Data Range: {actual_start} to {actual_end} ({len(df_chart)} records)")

                # Draw candlestick chart using matplotlib
                try:
                    chart_title = f'{selected_future} Candlestick Chart ({actual_start} to {actual_end})'
                    fig = plot_matplotlib_candlestick(df_chart, chart_title)

                    if fig is not None:
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)  # Close figure to free memory

                        # Display data summary below chart
                        with st.expander("📊 View Data Summary", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Open Price (Latest)", f"{df_chart['open'].iloc[-1]:,.0f}")
                            with col2:
                                st.metric("Close Price (Latest)", f"{df_chart['close'].iloc[-1]:,.0f}")
                            with col3:
                                change = (df_chart['close'].iloc[-1] - df_chart['open'].iloc[0]) / df_chart['open'].iloc[0] * 100
                                st.metric("Period Change (%)", f"{change:+.2f}%")

                            st.write("**Last 5 days data:**")
                            recent_data = df_chart[['date', 'open', 'high', 'low', 'close', 'volume']].tail(5).copy()
                            recent_data['date'] = recent_data['date'].dt.strftime('%Y-%m-%d')
                            st.dataframe(recent_data, hide_index=True, width='stretch')
                    else:
                        raise Exception("matplotlib chart generation failed")

                except Exception as e:
                    st.error(f"❌ Candlestick chart plotting failed: {str(e)}")
                    st.write("📊 Data preview (showing raw data due to chart plotting failure)")
                    fallback_data = df_chart[['date', 'open', 'high', 'low', 'close', 'volume']].tail(20).copy()
                    fallback_data['date'] = fallback_data['date'].dt.strftime('%Y-%m-%d')
                    st.dataframe(fallback_data, hide_index=True, width='stretch')

            with main_tab3:
                st.subheader("🔧 Feature Engineering")
                task = 'regression' if task_type == "Regression (Price Change %)" else 'classification'
                X, y = create_features_targets(df, historical_days, prediction_days, task)

                st.write(f"Feature matrix shape: {X.shape}")
                st.write(f"Target variable shape: {y.shape}")

                # Feature engineering data details
                if task == 'regression':
                    with st.expander("🔍 View Feature Engineering Details", expanded=True):
                        st.write("**Feature matrix example (first 5 rows):**")
                        display_X = X.head().copy()
                        display_y = y.head().copy()
                        display_X['Target Variable (Change %)'] = display_y.round(4)
                        st.dataframe(display_X, width='stretch')

                        st.write("**Feature Description:**")
                        feature_desc = {
                            'Historical Prices': f'Closing prices for the last {historical_days} days',
                            'Historical Volume': f'Volume for the last {historical_days} days',
                            'MA5 Indicator': '5-day moving average for the last 5 days',
                            'MA10 Indicator': '10-day moving average for the last 10 days',
                            'RSI Indicator': 'Relative Strength Index for the last 5 days',
                            'MACD Indicator': 'MACD indicator for the last 5 days',
                            'Current Price': 'Closing price on the prediction base date'
                        }

                        desc_df = pd.DataFrame({
                            'Feature Type': list(feature_desc.keys()),
                            'Description': list(feature_desc.values())
                        })
                        st.dataframe(desc_df, hide_index=True)

                        st.write("**Feature List:**")
                        feature_list_text = ", ".join(list(X.columns))
                        st.code(feature_list_text)

                        st.write("**Target Variable Statistics:**")
                        stats_df = pd.DataFrame({
                            'Statistic': ['Min', 'Max', 'Mean', 'Std Dev'],
                            'Change (%)': [
                                f"{y.min():.2f}%",
                                f"{y.max():.2f}%",
                                f"{y.mean():.2f}%",
                                f"{y.std():.2f}%"
                            ]
                        })
                        st.dataframe(stats_df, hide_index=True)

                # Classification task target variable distribution
                if task == 'classification':
                    st.write("**Target Variable Distribution:**")
                    label_mapping = {0: 'Down', 1: 'Sideways', 2: 'Up'}
                    distribution = y.value_counts().sort_index()
                    distribution_labeled = distribution.rename(index=label_mapping)

                    dist_df = pd.DataFrame({
                        'Price Trend': distribution_labeled.index,
                        'Sample Count': distribution_labeled.values,
                        'Percentage': (distribution_labeled.values / len(y) * 100).round(2)
                    })

                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(dist_df, hide_index=True)

                    with col2:
                        try:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            colors = ['#ef232a', '#ff9500', '#14b143']
                            bars = ax.bar(dist_df['Price Trend'], dist_df['Sample Count'], color=colors)
                            ax.set_title('Target Variable Distribution', fontsize=12)
                            ax.set_ylabel('Sample Count', fontsize=10)

                            # Ensure ticks match data
                            ax.set_xticks(range(len(dist_df)))
                            ax.set_xticklabels(dist_df['Price Trend'])

                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                       f'{int(height)}\n({height/len(y)*100:.1f}%)',
                                       ha='center', va='bottom', fontsize=8)

                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error creating distribution chart: {str(e)}")
                            st.write("Distribution data:")
                            st.dataframe(dist_df)

                    if len(y.unique()) < 3:
                        st.warning("Target variable classes are imbalanced, which may affect classification performance")

            with main_tab4:
                # Check if data is sufficient
                if len(X) < 10:
                    st.error(f"❌ Insufficient data samples ({len(X)} available), please increase data retrieval days or reduce historical data days")
                else:
                    st.subheader("🤖 Model Training & Prediction")

                    if task == 'regression':
                        # Regression prediction sub-tabs
                        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📊 Model Performance", "📈 Prediction Results", "🎯 Model Evaluation"])

                        results, best_model, X_test, y_test = regression_prediction(X, y, train_size)

                        with sub_tab1:
                            st.write("**Model Performance Comparison:**")
                            model_performance = []
                            for name, result in results.items():
                                model_performance.append({
                                    'Model': name,
                                    'R² Score': result['r2'],
                                    'MSE': result['mse']
                                })
                            performance_df = pd.DataFrame(model_performance)
                            st.dataframe(performance_df, width='stretch')

                            best_r2 = max(r['r2'] for r in results.values())
                            best_model_name = [name for name, r in results.items() if r['r2'] == best_r2][0]
                            st.success(f"🏆 Best Model: {best_model_name} (R² = {best_r2:.4f})")

                        with sub_tab2:
                            if best_model is not None:
                                st.write("**Prediction Results Visualization:**")
                                y_pred = best_model.predict(X_test)

                                fig = plot_predictions(y_test, y_pred, 'Regression Prediction Comparison')
                                st.pyplot(fig)
                                st.caption("📊 Chart Description: Points closer to the diagonal indicate more accurate predictions")

                                stats_df = pd.DataFrame({
                                    'Statistical Metric': ['Actual Min', 'Actual Max', 'Actual Mean', 'Predicted Min', 'Predicted Max', 'Predicted Mean'],
                                    'Change (%)': [
                                        f"{y_test.min():.2f}%", f"{y_test.max():.2f}%", f"{y_test.mean():.2f}%",
                                        f"{y_pred.min():.2f}%", f"{y_pred.max():.2f}%", f"{y_pred.mean():.2f}%"
                                    ]
                                })
                                st.dataframe(stats_df, hide_index=True, width='stretch')

                        with sub_tab3:
                            st.write("**Detailed Model Evaluation:**")
                            best_r2 = max(r['r2'] for r in results.values())
                            best_mse = min(r['mse'] for r in results.values())

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Best R² Score", f"{best_r2:.4f}")
                                st.caption("R² closer to 1 indicates better model fit")

                            with col2:
                                st.metric("Minimum MSE", f"{best_mse:.6f}")
                                st.caption("Smaller MSE indicates smaller prediction error")

                            models = list(results.keys())
                            r2_scores = [results[m]['r2'] for m in models]

                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.bar(models, r2_scores, color='skyblue', alpha=0.7)
                            ax.set_title('Model R² Score Comparison', fontsize=14, fontweight='bold')
                            ax.set_ylabel('R² Score', fontsize=12)
                            ax.set_xlabel('Model', fontsize=12)
                            ax.grid(True, alpha=0.3)
                            plt.xticks(rotation=45)

                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                       f'{height:.4f}', ha='center', va='bottom')

                            plt.tight_layout()
                            st.pyplot(fig)

                    else:
                        # Classification prediction sub-tabs
                        sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(["📊 Model Performance", "🎯 Confusion Matrix", "📋 Classification Report", "🔍 Prediction Analysis"])

                        results, best_model, X_test, y_test = classification_prediction(X, y, train_size)

                        with sub_tab1:
                            st.write("**Model Accuracy Comparison:**")
                            model_performance = []
                            for name, result in results.items():
                                model_performance.append({
                                    'Model': name,
                                    'Accuracy': result['accuracy']
                                })
                            performance_df = pd.DataFrame(model_performance)
                            st.dataframe(performance_df, width='stretch')

                            best_accuracy = max(r['accuracy'] for r in results.values())
                            best_model_name = [name for name, r in results.items() if r['accuracy'] == best_accuracy][0]
                            st.success(f"🏆 Best Model: {best_model_name} (Accuracy = {best_accuracy:.4f})")

                        with sub_tab2:
                            if best_model is not None:
                                st.write("**Confusion Matrix:**")
                                best_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
                                y_pred = results[best_name]['predictions']

                                from sklearn.metrics import confusion_matrix
                                cm = confusion_matrix(y_test, y_pred)
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
                                ax.set_xlabel('Predicted', fontsize=12)
                                ax.set_ylabel('Actual', fontsize=12)
                                ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

                                # Set appropriate labels based on actual classes present
                                unique_labels = sorted(list(set(y_test) | set(y_pred)))
                                label_names = {0: 'Down', 1: 'Sideways', 2: 'Up'}
                                tick_labels = [label_names[i] for i in unique_labels]

                                # Ensure the number of ticks matches the number of labels
                                ax.set_xticks(range(len(tick_labels)))
                                ax.set_yticks(range(len(tick_labels)))
                                ax.set_xticklabels(tick_labels)
                                ax.set_yticklabels(tick_labels)
                                st.pyplot(fig)
                                st.caption("📊 Confusion Matrix Description: Diagonal numbers indicate correctly predicted samples")

                        with sub_tab3:
                            if best_model is not None:
                                st.write("**Classification Report:**")
                                from sklearn.metrics import classification_report

                                # Create target names based on actual classes present
                                unique_labels = sorted(list(set(y_test) | set(y_pred)))
                                label_names = {0: 'Down', 1: 'Sideways', 2: 'Up'}
                                target_names = [label_names[i] for i in unique_labels]

                                classification_rep = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

                                classification_df = pd.DataFrame(classification_rep).transpose()
                                # Use actual class names that exist in the data
                                display_rows = target_names + ['macro avg', 'weighted avg']
                                display_df = classification_df.loc[display_rows, ['precision', 'recall', 'f1-score', 'support']]
                                st.dataframe(display_df.astype({'support': 'int'}).round(4), width='stretch')

                                st.info("""
                                📊 **Metric Description:**
                                - **Precision**: Of all predicted positive cases, how many are actually positive
                                - **Recall**: Of all actual positive cases, how many are correctly predicted
                                - **F1-Score**: Harmonic mean of precision and recall
                                - **Support**: Number of actual samples in each class
                                """)

                        with sub_tab4:
                            if best_model is not None:
                                st.write("**Prediction Analysis:**")
                                actual_counts = pd.Series(y_test).value_counts().sort_index()
                                pred_counts = pd.Series(y_pred).value_counts().sort_index()

                                label_mapping = {0: 'Down', 1: 'Sideways', 2: 'Up'}

                                comparison_df = pd.DataFrame({
                                    'Price Trend': [label_mapping[i] for i in actual_counts.index],
                                    'Actual Count': actual_counts.values,
                                    'Predicted Count': [pred_counts.get(i, 0) for i in actual_counts.index],
                                    'Accuracy': [results[best_name]['predictions'][y_test == i].tolist().count(i) / len(y_test[y_test == i]) for i in actual_counts.index]
                                })
                                comparison_df['Accuracy'] = (comparison_df['Accuracy'] * 100).round(2)
                                comparison_df['Accuracy'] = comparison_df['Accuracy'].astype(str) + '%'

                                st.dataframe(comparison_df, hide_index=True, width='stretch')

                                overall_accuracy = results[best_name]['accuracy']
                                st.metric("Overall Accuracy", f"{overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

            with main_tab5:
                st.subheader("📊 Feature Importance Analysis")
                # Feature importance analysis
                if len(X) >= 10 and 'best_model' in locals() and best_model is not None and hasattr(best_model, 'feature_importances_'):
                    st.info("🔍 Showing the most important features from model training (only supported by tree-based models like Random Forest, Gradient Boosting, etc.)")

                    feature_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': best_model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)

                    # Visualize feature importance
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Feature Importance Chart:**")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.barh(range(len(feature_importance)), feature_importance['Importance'], color='skyblue', alpha=0.8)
                        ax.set_yticks(range(len(feature_importance)))
                        ax.set_yticklabels(feature_importance['Feature'], fontsize=10)
                        ax.set_xlabel('Importance Score', fontsize=12)
                        ax.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3)

                        # Add values on bars
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                                   f'{width:.4f}', ha='left', va='center', fontsize=9)

                        plt.tight_layout()
                        st.pyplot(fig)

                    with col2:
                        st.write("**Feature Importance Table:**")
                        st.dataframe(feature_importance, hide_index=True, width='stretch')

                    # Feature type analysis
                    st.write("**Feature Type Analysis:**")
                    feature_types = {
                        'Price Features': [col for col in feature_importance['Feature'] if 'close' in col or 'current_price' in col],
                        'Volume Features': [col for col in feature_importance['Feature'] if 'volume' in col],
                        'Technical Indicators': [col for col in feature_importance['Feature'] if any(ind in col for ind in ['MA5', 'MA10', 'RSI', 'MACD'])]
                    }

                    type_importance = {}
                    for ftype, features in feature_types.items():
                        if features:
                            total_importance = feature_importance[feature_importance['Feature'].isin(features)]['Importance'].sum()
                            type_importance[ftype] = total_importance

                    if type_importance:
                        type_df = pd.DataFrame({
                            'Feature Type': list(type_importance.keys()),
                            'Total Importance': list(type_importance.values()),
                            'Percentage': [(v / sum(type_importance.values()) * 100) for v in type_importance.values()]
                        })
                        type_df['Percentage'] = type_df['Percentage'].round(2).astype(str) + '%'
                        st.dataframe(type_df, hide_index=True)

                    st.caption("📊 Feature Description: close=Closing Price, volume=Volume, MA=Moving Average, RSI=Relative Strength Index, MACD=Indicator")

                else:
                    st.info("ℹ️ Feature importance analysis requires the following conditions:")
                    st.write("1. Sufficient data samples")
                    st.write("2. Model training completed")
                    st.write("3. Model supports feature importance analysis (e.g., Random Forest, Gradient Boosting)")
                    st.write("4. Linear models (e.g., Linear Regression, Logistic Regression) do not support this feature")

if __name__ == "__main__":
    main()