import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                           classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')


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
            if price_change > 0.5:  # Price increase > 1%
                target = 2  # Up
            elif price_change < -0.5:  # Price decrease > 1%
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


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

# Regression prediction model
def regression_prediction(X, y, train_size=0.7):
    """Use multiple regression models for prediction"""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge

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
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

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


# Future prediction function
def predict_future_trend(model, df, historical_days, prediction_days, task_type='regression'):
    """
    Use trained model to predict future trends
    :param model: Trained ML model
    :param df: Original data with technical indicators
    :param historical_days: Number of historical days for features
    :param prediction_days: Number of days to predict into future
    :param task_type: 'regression' or 'classification'
    :return: DataFrame with predictions and confidence scores
    """
    try:
        df_features = calculate_technical_indicators(df)

        # Get the most recent data for prediction
        latest_data = df_features.tail(historical_days).copy()

        # Check if we have enough data
        if len(latest_data) < historical_days:
            raise ValueError(f"Insufficient historical data: need {historical_days} days, got {len(latest_data)}")

        # Prepare features for prediction
        features = []

        # Historical prices and volumes
        features.extend(latest_data['close'].fillna(0).values)
        features.extend(latest_data['volume'].fillna(0).values)

        # Technical indicators (last 5 values) - handle NaN values
        features.extend(latest_data['MA5'].fillna(0).values[-5:])
        features.extend(latest_data['MA10'].fillna(0).values[-5:])
        features.extend(latest_data['RSI'].fillna(50).values[-5:])  # RSI default 50
        features.extend(latest_data['MACD'].fillna(0).values[-5:])

        # Current price
        current_price = latest_data['close'].iloc[-1]
        features.append(current_price)

        # Convert to DataFrame
        feature_names = []
        for i in range(historical_days):
            feature_names.extend([f'close_{i}', f'volume_{i}'])
        feature_names.extend([f'MA5_{i}' for i in range(5)])
        feature_names.extend([f'MA10_{i}' for i in range(5)])
        feature_names.extend([f'RSI_{i}' for i in range(5)])
        feature_names.extend([f'MACD_{i}' for i in range(5)])
        feature_names.append('current_price')

        X_pred = pd.DataFrame([features], columns=feature_names)

        # Handle any remaining NaN values
        X_pred = X_pred.fillna(0)

    except Exception as e:
        raise ValueError(f"Error preparing prediction features: {str(e)}")

    # Make prediction
    if task_type == 'regression':
        predicted_change = model.predict(X_pred)[0]
        predicted_prices = []
        confidence_intervals = []

        # Calculate predicted prices for each future day
        for day in range(1, prediction_days + 1):
            # Assume compound growth based on predicted daily change
            daily_change = predicted_change / prediction_days
            predicted_price = current_price * (1 + daily_change / 100) ** day
            predicted_prices.append(predicted_price)

            # Simple confidence interval (±20% of predicted change)
            margin = abs(predicted_change) * 0.2 * (day / prediction_days)
            confidence_intervals.append([predicted_price - margin, predicted_price + margin])

        # Create prediction results DataFrame
        future_dates = [latest_data['date'].iloc[-1] + timedelta(days=i) for i in range(1, prediction_days + 1)]

        results = pd.DataFrame({
            'date': future_dates,
            'predicted_price': predicted_prices,
            'confidence_lower': [ci[0] for ci in confidence_intervals],
            'confidence_upper': [ci[1] for ci in confidence_intervals],
            'predicted_change_pct': [((p - current_price) / current_price * 100) for p in predicted_prices]
        })

    else:  # classification
        predicted_class = model.predict(X_pred)[0]
        class_probabilities = model.predict_proba(X_pred)[0] if hasattr(model, 'predict_proba') else [1.0, 0.0, 0.0]

        # Map class to trend
        class_names = {0: 'Down', 1: 'Sideways', 2: 'Up'}
        predicted_trend = class_names.get(predicted_class, 'Unknown')
        confidence = max(class_probabilities) * 100

        # For classification, create simple trend projection
        future_dates = [latest_data['date'].iloc[-1] + timedelta(days=i) for i in range(1, prediction_days + 1)]
        predicted_prices = []

        # Calculate trend-based price projection
        if predicted_class == 2:  # Up
            daily_change = 0.5  # 0.5% per day
        elif predicted_class == 0:  # Down
            daily_change = -0.5  # -0.5% per day
        else:  # Sideways
            daily_change = 0.0  # No change

        for day in range(1, prediction_days + 1):
            predicted_price = current_price * (1 + daily_change / 100) ** day
            predicted_prices.append(predicted_price)

        results = pd.DataFrame({
            'date': future_dates,
            'predicted_price': predicted_prices,
            'predicted_trend': [predicted_trend] * prediction_days,
            'confidence': [confidence] * prediction_days,
            'predicted_change_pct': [((p - current_price) / current_price * 100) for p in predicted_prices]
        })

    return results, current_price


# Generate prediction report
def generate_prediction_report(prediction_results, current_price, task_type, symbol, prediction_days):
    """Generate a comprehensive prediction report"""

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"🔮 FUTURES PREDICTION REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Contract: {symbol}")
    report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Current Price: {current_price:.2f}")
    report_lines.append(f"Prediction Period: {prediction_days} days")
    report_lines.append(f"Prediction Type: {task_type.title()}")
    report_lines.append("")

    if task_type == 'regression':
        final_prediction = prediction_results['predicted_price'].iloc[-1]
        total_change = prediction_results['predicted_change_pct'].iloc[-1]

        report_lines.append("📈 PREDICTION RESULTS:")
        report_lines.append(f"Target Price (Day {prediction_days}): {final_prediction:.2f}")
        report_lines.append(f"Expected Change: {total_change:+.2f}%")
        report_lines.append("")

        # Confidence intervals
        final_lower = prediction_results['confidence_lower'].iloc[-1]
        final_upper = prediction_results['confidence_upper'].iloc[-1]
        report_lines.append("📊 CONFIDENCE INTERVAL:")
        report_lines.append(f"Lower Bound: {final_lower:.2f}")
        report_lines.append(f"Upper Bound: {final_upper:.2f}")
        report_lines.append("")

        # Daily breakdown
        report_lines.append("📅 DAILY PROJECTIONS:")
        for _, row in prediction_results.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            price = row['predicted_price']
            change = row['predicted_change_pct']
            lower = row['confidence_lower']
            upper = row['confidence_upper']

            report_lines.append(f"{date_str}: {price:.2f} ({change:+.2f}%)")
            report_lines.append(f"  Confidence: [{lower:.2f} - {upper:.2f}]")

    else:  # classification
        predicted_trend = prediction_results['predicted_trend'].iloc[0]
        confidence = prediction_results['confidence'].iloc[0]
        final_prediction = prediction_results['predicted_price'].iloc[-1]
        total_change = prediction_results['predicted_change_pct'].iloc[-1]

        report_lines.append("📈 PREDICTION RESULTS:")
        report_lines.append(f"Predicted Trend: {predicted_trend}")
        report_lines.append(f"Confidence Level: {confidence:.1f}%")
        report_lines.append(f"Target Price (Day {prediction_days}): {final_prediction:.2f}")
        report_lines.append(f"Expected Change: {total_change:+.2f}%")
        report_lines.append("")

        # Daily breakdown
        report_lines.append("📅 DAILY PROJECTIONS:")
        for _, row in prediction_results.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            price = row['predicted_price']
            change = row['predicted_change_pct']

            report_lines.append(f"{date_str}: {price:.2f} ({change:+.2f}%)")

    report_lines.append("")
    report_lines.append("⚠️  DISCLAIMER:")
    report_lines.append("This prediction is based on historical data and machine learning models.")
    report_lines.append("It should not be considered as financial advice.")
    report_lines.append("Market conditions can change rapidly and past performance")
    report_lines.append("does not guarantee future results.")
    report_lines.append("=" * 60)

    return "\n".join(report_lines)


# ============================================================================
# STREAMLIT INTERFACE
# ============================================================================

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
            main_tab1, main_tab2, main_tab3, main_tab4, main_tab5, main_tab6 = st.tabs([
                "📊 Raw Data",
                "📈 Price Charts",
                "🔧 Feature Engineering",
                "🤖 Model Training & Prediction",
                "📊 Feature Importance",
                "🔮 Future Prediction Report"
            ])

            with main_tab1:
                st.subheader("📊 Raw Data Preview")
                st.info(f"Showing the last 10 data records for {selected_future}({futures_symbol})")
                st.dataframe(df.tail(10), width='stretch')


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
                # Initialize best_model for future prediction tab
                best_model = None

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

            with main_tab6:
                st.subheader("🔮 Future Prediction Report")

                # Simple test version first
                st.write("🔮 Future Prediction Tab Loaded Successfully")

                # Check if model training was completed
                if 'X' not in locals():
                    st.error("❌ Data not loaded. Please run the analysis first.")
                    st.stop()
                elif len(X) < 10:
                    st.error(f"❌ Insufficient data samples ({len(X)} available), please increase data retrieval days or reduce historical data days")
                    st.stop()
                elif 'best_model' not in locals() or best_model is None:
                    st.warning("⚠️ Please complete model training in the 'Model Training & Prediction' tab first")
                    st.info("💡 Train the model by going to the '🤖 Model Training & Prediction' tab and running the analysis")
                    st.stop()
                else:
                    st.success("✅ All requirements met - Ready for prediction!")

                    # Simple prediction controls
                    st.write("**Basic Configuration:**")
                    future_prediction_days = st.slider(
                        "Future Prediction Days",
                        min_value=1,
                        max_value=7,  # Reduced for testing
                        value=3,
                        help="Number of days to predict into the future"
                    )

                    # Generate predictions
                    if st.button("🧪 Test Simple Prediction", key="test_pred"):
                        st.write("🔄 Starting simple prediction test...")

                        try:
                            # Simple test prediction
                            current_price = df['close'].iloc[-1]

                            # Create simple mock prediction results for testing
                            future_dates = [df['date'].iloc[-1] + timedelta(days=i) for i in range(1, future_prediction_days + 1)]

                            # Simple linear projection
                            daily_change = 0.5  # 0.5% per day
                            predicted_prices = []
                            for day in range(1, future_prediction_days + 1):
                                price = current_price * (1 + daily_change / 100) ** day
                                predicted_prices.append(price)

                            simple_results = pd.DataFrame({
                                'date': future_dates,
                                'predicted_price': predicted_prices,
                                'predicted_change_pct': [((p - current_price) / current_price * 100) for p in predicted_prices]
                            })

                            st.success("✅ Simple prediction test successful!")
                            st.dataframe(simple_results)

                            # Simple chart
                            fig, ax = plt.subplots(figsize=(10, 5))

                            # Historical data
                            hist_data = df.tail(10)
                            ax.plot(hist_data['date'], hist_data['close'], 'b-', label='Historical')

                            # Predictions
                            ax.plot(simple_results['date'], simple_results['predicted_price'], 'r--', label='Predicted')

                            ax.set_title('Simple Prediction Test')
                            ax.legend()
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                            plt.close(fig)

                        except Exception as e:
                            st.error(f"❌ Simple prediction failed: {str(e)}")
                            st.write("Error details:", str(e))

                    # Full prediction button
                    if st.button("🚀 Generate Full Future Predictions", key="full_pred"):
                        st.write("🔄 Starting full prediction process...")

                        try:
                            with st.spinner("Generating predictions..."):
                                # Call the actual prediction function
                                prediction_results, current_price = predict_future_trend(
                                    best_model, df, historical_days, future_prediction_days, task
                                )

                            st.success("✅ Full predictions generated successfully!")
                            st.dataframe(prediction_results)

                        except Exception as e:
                            st.error(f"❌ Full prediction failed: {str(e)}")
                            st.write("Error details:", str(e))
                            import traceback
                            st.write("Full traceback:")
                            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()