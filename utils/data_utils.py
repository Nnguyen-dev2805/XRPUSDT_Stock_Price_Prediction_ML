"""
Data Utilities Module
Xử lý load, validate và append dữ liệu
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_data(filepath):
    """
    Load và preprocess dữ liệu từ CSV
    
    Args:
        filepath: Đường dẫn đến file CSV
    
    Returns:
        DataFrame đã được xử lý
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Drop Change % column if exists
    if 'Change %' in df.columns:
        df = df.drop(columns=['Change %'])
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df


def get_latest_row(df):
    """
    Lấy dòng dữ liệu mới nhất
    
    Args:
        df: DataFrame
    
    Returns:
        Series chứa dòng dữ liệu mới nhất
    """
    return df.iloc[-1]


def get_latest_n_rows(df, n=10):
    """
    Lấy n dòng dữ liệu mới nhất
    
    Args:
        df: DataFrame
        n: Số dòng cần lấy
    
    Returns:
        DataFrame chứa n dòng mới nhất
    """
    return df.tail(n)


def append_prediction_to_csv(filepath, prediction_data):
    """
    Thêm dữ liệu dự đoán vào CSV
    
    Args:
        filepath: Đường dẫn đến file CSV
        prediction_data: Dict chứa dữ liệu dự đoán
            {
                'Date': datetime,
                'Price': float,
                'Open': float,
                'High': float,
                'Low': float,
                'Vol': float
            }
    
    Returns:
        bool: True nếu thành công
    """
    try:
        # Load existing data
        df = pd.read_csv(filepath)
        
        # Create new row
        new_row = pd.DataFrame([prediction_data])
        
        # Convert Date sang định dạng ISO (YYYY-MM-DD) để đồng nhất
        if not pd.api.types.is_datetime64_any_dtype(new_row['Date']):
            new_row['Date'] = pd.to_datetime(new_row['Date'], dayfirst=False, format='mixed')
        
        new_row['Date'] = new_row['Date'].dt.strftime('%Y-%m-%d')
        
        # Append to dataframe (thêm vào cuối)
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Save back to CSV
        df.to_csv(filepath, index=False)
        
        return True
    except Exception as e:
        print(f"Error appending to CSV: {e}")
        return False


def validate_data(df):
    """
    Kiểm tra tính hợp lệ của dữ liệu
    
    Args:
        df: DataFrame cần validate
    
    Returns:
        tuple: (is_valid, error_message)
    """
    required_columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Vol']
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"
    
    # Check for null values in critical columns
    if df[required_columns].isnull().any().any():
        return False, "Found null values in data"
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        return False, "Date column is not datetime type"
    
    # Check for negative prices
    if (df['Price'] < 0).any() or (df['Open'] < 0).any() or (df['High'] < 0).any() or (df['Low'] < 0).any():
        return False, "Found negative prices"
    
    # Check High >= Low
    if (df['High'] < df['Low']).any():
        return False, "High price is less than Low price"
    
    return True, "Data is valid"


def get_next_trading_date(last_date):
    """
    Tính ngày giao dịch tiếp theo (bỏ qua weekend)
    
    Args:
        last_date: datetime của ngày cuối cùng
    
    Returns:
        datetime của ngày giao dịch tiếp theo
    """
    next_date = last_date + timedelta(days=1)
    
    # Skip weekends (crypto trades 24/7, but we can keep daily pattern)
    # For crypto, we just add 1 day
    return next_date


def format_number(num, decimals=4):
    """
    Format số với số chữ số thập phân
    """
    return f"{num:.{decimals}f}"


def calculate_change_percent(current, previous):
    """
    Tính % thay đổi
    """
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100
