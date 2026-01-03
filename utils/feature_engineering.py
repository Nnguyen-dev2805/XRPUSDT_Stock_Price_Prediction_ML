import pandas as pd
import numpy as np


def create_advanced_features(data):
    feature_data = data.copy()
    
    # ==================== PRICE-BASED FEATURES ====================
    # Returns = % thay đổi giá
    feature_data['Return_1d'] = feature_data['Price'].pct_change(1) * 100
    feature_data['Return_3d'] = feature_data['Price'].pct_change(3) * 100
    feature_data['Return_7d'] = feature_data['Price'].pct_change(7) * 100
    
    # Price ranges
    feature_data['HL_Range'] = feature_data['High'] - feature_data['Low']
    feature_data['HL_Range_Pct'] = (feature_data['HL_Range'] / feature_data['Low']) * 100
    feature_data['OC_Range'] = abs(feature_data['Open'] - feature_data['Price'])
    feature_data['OC_Range_Pct'] = (feature_data['OC_Range'] / feature_data['Open']) * 100
    
    # Price position
    feature_data['Price_Position'] = (feature_data['Price'] - feature_data['Low']) / (feature_data['HL_Range'] + 1e-8)
    
    # ==================== MOVING AVERAGES ====================
    for period in [5, 7, 10, 14, 20, 30]:
        feature_data[f'SMA_{period}'] = feature_data['Price'].rolling(window=period).mean()
        feature_data[f'Price_to_SMA_{period}'] = (feature_data['Price'] / feature_data[f'SMA_{period}'] - 1) * 100
    
    for period in [5, 10, 20]:
        feature_data[f'EMA_{period}'] = feature_data['Price'].ewm(span=period, adjust=False).mean()
        feature_data[f'Price_to_EMA_{period}'] = (feature_data['Price'] / feature_data[f'EMA_{period}'] - 1) * 100
    
    # ==================== BOLLINGER BANDS ====================
    for period in [10, 20]:
        bb_middle = feature_data['Price'].rolling(window=period).mean()
        bb_std = feature_data['Price'].rolling(window=period).std()
        
        feature_data[f'BB_Upper_{period}'] = bb_middle + (2 * bb_std)
        feature_data[f'BB_Lower_{period}'] = bb_middle - (2 * bb_std)
        feature_data[f'BB_Width_{period}'] = feature_data[f'BB_Upper_{period}'] - feature_data[f'BB_Lower_{period}']
        feature_data[f'BB_Position_{period}'] = (feature_data['Price'] - feature_data[f'BB_Lower_{period}']) / (feature_data[f'BB_Width_{period}'] + 1e-8)
    
    # ==================== MOMENTUM INDICATORS ====================
    # RSI
    for period in [7, 14]:
        price_delta = feature_data['Price'].diff()
        gain = (price_delta.where(price_delta > 0, 0)).rolling(window=period).mean()
        loss = (-price_delta.where(price_delta < 0, 0)).rolling(window=period).mean()
        rs_ratio = gain / loss
        feature_data[f'RSI_{period}'] = 100 - (100 / (1 + rs_ratio))
    
    # Stochastic
    for period in [14]:
        low_min = feature_data['Low'].rolling(window=period).min()
        high_max = feature_data['High'].rolling(window=period).max()
        feature_data[f'Stoch_{period}'] = 100 * (feature_data['Price'] - low_min) / (high_max - low_min + 1e-8)
    
    # ROC
    for period in [5, 10]:
        feature_data[f'ROC_{period}'] = ((feature_data['Price'] - feature_data['Price'].shift(period)) / feature_data['Price'].shift(period)) * 100
    
    # ==================== MACD ====================
    ema_12 = feature_data['Price'].ewm(span=12, adjust=False).mean()
    ema_26 = feature_data['Price'].ewm(span=26, adjust=False).mean()
    feature_data['MACD'] = ema_12 - ema_26
    feature_data['MACD_Signal'] = feature_data['MACD'].ewm(span=9, adjust=False).mean()
    feature_data['MACD_Histogram'] = feature_data['MACD'] - feature_data['MACD_Signal']
    
    # ==================== VOLATILITY ====================
    for period in [5, 10, 20]:
        feature_data[f'Volatility_{period}'] = feature_data['Return_1d'].rolling(window=period).std()
        feature_data[f'ATR_{period}'] = feature_data['HL_Range'].rolling(window=period).mean()
    
    # ==================== VOLUME FEATURES ====================
    feature_data['Vol_Change'] = feature_data['Vol'].pct_change() * 100
    
    for period in [5, 10, 20]:
        feature_data[f'Vol_SMA_{period}'] = feature_data['Vol'].rolling(window=period).mean()
        feature_data[f'Vol_Ratio_{period}'] = feature_data['Vol'] / (feature_data[f'Vol_SMA_{period}'] + 1e-8)
    
    feature_data['Vol_Price_Corr_20'] = feature_data['Vol'].rolling(window=20).corr(feature_data['Price'])
    
    # ==================== LAG FEATURES ====================
    for lag in [1, 2, 3, 5, 7]:
        feature_data[f'Price_Lag_{lag}'] = feature_data['Price'].shift(lag)
        feature_data[f'Return_Lag_{lag}'] = feature_data['Return_1d'].shift(lag)
        feature_data[f'Vol_Lag_{lag}'] = feature_data['Vol'].shift(lag)
    
    # ==================== STATISTICAL FEATURES ====================
    for period in [7, 14, 30]:
        feature_data[f'Price_Mean_{period}'] = feature_data['Price'].rolling(window=period).mean()
        feature_data[f'Price_Std_{period}'] = feature_data['Price'].rolling(window=period).std()
        feature_data[f'Price_Min_{period}'] = feature_data['Price'].rolling(window=period).min()
        feature_data[f'Price_Max_{period}'] = feature_data['Price'].rolling(window=period).max()
        feature_data[f'Price_Range_{period}'] = feature_data[f'Price_Max_{period}'] - feature_data[f'Price_Min_{period}']
    
    # ==================== TIME FEATURES ====================
    feature_data['Day_of_Week'] = feature_data['Date'].dt.dayofweek
    feature_data['Day_of_Month'] = feature_data['Date'].dt.day
    feature_data['Month'] = feature_data['Date'].dt.month
    feature_data['Quarter'] = feature_data['Date'].dt.quarter
    feature_data['Year'] = feature_data['Date'].dt.year
    
    # ==================== TARGET VARIABLE ====================
    feature_data['Target_Price'] = feature_data['Price'].shift(-1)
    
    return feature_data



#########
def create_lstm_features(data):
    """
    Tạo các features chuyên biệt cho mô hình LSTM
    Dựa trên notebook LSTM_Price_Forecast_Pipeline.ipynb
    """
    df = data.copy()
    EPS = 1e-8
    
    # Feature Engineering: VVR, VWAP, Lag Features
    df['VVR'] = df['Vol'] / (df['High'] - df['Low'] + EPS)
    df['VWAP'] = ((df['High'] + df['Low'] + df['Price']) / 3 * df['Vol']).cumsum() / df['Vol'].cumsum()
    
    for lag in [1, 2, 3, 5, 7]:
        df[f'Lag_{lag}'] = df['Price'].shift(lag)
        
    df['Price_Change'] = df['Price'].pct_change()
    df['Volatility'] = df['High'] - df['Low']
    df['MA5'] = df['Price'].rolling(5).mean()
    df['MA10'] = df['Price'].rolling(10).mean()
    
    return df


def get_feature_columns():
    """
    Trả về danh sách các feature columns để train model
    """
    base_features = ['Open', 'High', 'Low', 'Vol']
    
    # Price features
    price_features = ['Return_1d', 'Return_3d', 'Return_7d', 
                     'HL_Range', 'HL_Range_Pct', 'OC_Range', 'OC_Range_Pct', 'Price_Position']
    
    # Moving averages
    ma_features = []
    for period in [5, 7, 10, 14, 20, 30]:
        ma_features.extend([f'SMA_{period}', f'Price_to_SMA_{period}'])
    for period in [5, 10, 20]:
        ma_features.extend([f'EMA_{period}', f'Price_to_EMA_{period}'])
    
    # Bollinger Bands
    bb_features = []
    for period in [10, 20]:
        bb_features.extend([f'BB_Upper_{period}', f'BB_Lower_{period}', 
                           f'BB_Width_{period}', f'BB_Position_{period}'])
    
    # Momentum
    momentum_features = ['RSI_7', 'RSI_14', 'Stoch_14', 'ROC_5', 'ROC_10',
                        'MACD', 'MACD_Signal', 'MACD_Histogram']
    
    # Volatility
    volatility_features = []
    for period in [5, 10, 20]:
        volatility_features.extend([f'Volatility_{period}', f'ATR_{period}'])
    
    # Volume
    volume_features = ['Vol_Change', 'Vol_Price_Corr_20']
    for period in [5, 10, 20]:
        volume_features.extend([f'Vol_SMA_{period}', f'Vol_Ratio_{period}'])
    
    # Lag features
    lag_features = []
    for lag in [1, 2, 3, 5, 7]:
        lag_features.extend([f'Price_Lag_{lag}', f'Return_Lag_{lag}', f'Vol_Lag_{lag}'])
    
    # Statistical features
    stat_features = []
    for period in [7, 14, 30]:
        stat_features.extend([f'Price_Mean_{period}', f'Price_Std_{period}',
                             f'Price_Min_{period}', f'Price_Max_{period}', f'Price_Range_{period}'])
    
    # Time features
    time_features = ['Day_of_Week', 'Day_of_Month', 'Month', 'Quarter', 'Year']
    
    all_features = (base_features + price_features + ma_features + bb_features + 
                   momentum_features + volatility_features + volume_features + 
                   lag_features + stat_features + time_features)
    
    return all_features
