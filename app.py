"""
XRP/USDT Price Prediction Web Application
Dự đoán giá XRP sử dụng Machine Learning (Layer 1: RandomForest)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add utils to path
sys.path.append(os.path.dirname(__file__))

from utils import (
    load_data, get_latest_row, get_latest_n_rows,
    create_advanced_features, get_feature_columns,
    train_layer1_model, load_model, save_model,
    predict_next_day_layer1, create_prediction_with_confidence,
    evaluate_model, get_feature_importance, prepare_data_for_training,
    plot_price_history, plot_candlestick, plot_volume,
    plot_technical_indicators, plot_prediction_result, plot_feature_importance,
    get_next_trading_date, format_number, calculate_change_percent,
    append_prediction_to_csv, validate_data
)

# Page config
st.set_page_config(
    page_title="Dự đoán giá XRP",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00D9FF 0%, #FF6B6B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .stButton>button {
        background: linear-gradient(90deg, #00D9FF 0%, #4ECDC4 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0, 217, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Constants
DATA_PATH = './data/XRPUSDT_train.csv'
MODEL_PATH = './models/layer1_rf_model.pkl'
SCALER_PATH = './models/layer1_scaler.pkl'

# Session state initialization
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'df_features' not in st.session_state:
    st.session_state.df_features = None


def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">DỰ ĐOÁN GIÁ XRP/USDT</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - Only Layer selection
    with st.sidebar:
        st.title("Mô hình")
        
        # Layer selection
        selected_layer = st.radio(
            "Chọn Layer",
            ["Layer 1 - RandomForest"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.info("**Layer 1**: RandomForest Regressor\n\n**Features**: 90+ Chỉ số kỹ thuật")
    
    # Main content
    if selected_layer == "Layer 1 - RandomForest":
        display_layer1_content()


def display_layer1_content():
    """Display Layer 1 content with controls and dashboard"""
    
    # Control buttons at top
    st.subheader("Điều khiển")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Tải dữ liệu", use_container_width=True, type="primary"):
            load_and_process_data()
    
    with col2:
        if st.button("Train mô hình mới", use_container_width=True, disabled=st.session_state.df_features is None):
            train_model()
    
    with col3:
        if st.button("Load mô hình đã lưu", use_container_width=True):
            load_saved_model()
    
    with col4:
        if st.button("Dự đoán ngày tiếp theo", use_container_width=True, disabled=not st.session_state.model_trained):
            make_prediction()
    
    st.markdown("---")
    
    # Display dashboard if data is loaded
    if st.session_state.df_features is not None:
        display_dashboard()
    else:
        st.info("Vui lòng nhấn **Tải dữ liệu** để bắt đầu")


def load_and_process_data():
    """Load and process data with features"""
    with st.spinner("Đang tải và xử lý dữ liệu..."):
        try:
            # Load data
            df = load_data(DATA_PATH)
            
            # Validate
            is_valid, msg = validate_data(df)
            if not is_valid:
                st.error(f"Dữ liệu không hợp lệ: {msg}")
                return
            
            # Create features
            df_features = create_advanced_features(df)
            
            # Store in session state
            st.session_state.df_features = df_features
            
            st.success(f"Đã tải {len(df)} dòng dữ liệu với {len(df_features.columns)} features!")
            
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu: {e}")


def train_model():
    """Train Layer 1 RandomForest model"""
    if st.session_state.df_features is None:
        st.warning("Vui lòng tải dữ liệu trước!")
        return
    
    with st.spinner("Đang huấn luyện mô hình RandomForest... Có thể mất vài phút."):
        try:
            # Get feature columns
            feature_cols = get_feature_columns()
            
            # Prepare data
            X_train, X_test, y_train, y_test = prepare_data_for_training(
                st.session_state.df_features,
                feature_cols,
                target_column='Target_Price',
                test_size=0.2
            )
            
            # Train model (returns both model and scaler)
            model, scaler = train_layer1_model(X_train, y_train)
            
            # Evaluate
            metrics = evaluate_model(model, scaler, X_test, y_test)
            
            # Save model and scaler
            save_model(model, MODEL_PATH)
            save_model(scaler, MODEL_PATH.replace('_model.pkl', '_scaler.pkl'))
            
            # Add predictions to dataframe
            # Get all data with features (not just train/test split)
            df_clean = st.session_state.df_features.dropna(subset=feature_cols + ['Target_Price'])
            X_all = df_clean[feature_cols]
            
            # Scale and predict
            X_all_scaled = scaler.transform(X_all)
            predictions = model.predict(X_all_scaled)
            
            # Add RF_Pred_Tomorrow to the cleaned dataframe
            st.session_state.df_features.loc[df_clean.index, 'RF_Pred_Tomorrow'] = predictions
            
            # Create RF_Pred_Today by shifting RF_Pred_Tomorrow
            st.session_state.df_features['RF_Pred_Today'] = st.session_state.df_features['RF_Pred_Tomorrow'].shift(1)
            
            # Store in session state
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.model_trained = True
            st.session_state.metrics = metrics
            st.session_state.feature_cols = feature_cols
            
            # Display metrics
            last_date = df_clean['Date'].max().strftime('%d/%m/%Y')
            st.success(f"Huấn luyện mô hình thành công! Dữ liệu huấn luyện đến ngày: **{last_date}**")
            st.info("Đã thêm cột RF_Pred_Tomorrow và RF_Pred_Today vào dữ liệu")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAE", f"{metrics['MAE']:.6f}")
            with col2:
                st.metric("RMSE", f"{metrics['RMSE']:.6f}")
            with col3:
                st.metric("R²", f"{metrics['R2']:.4f}")
            with col4:
                st.metric("Độ chính xác hướng", f"{metrics['Direction_Accuracy']:.2f}%")
            
        except Exception as e:
            st.error(f"Lỗi khi huấn luyện mô hình: {e}")
            import traceback
            st.error(traceback.format_exc())


def load_saved_model():
    """Load pre-trained model"""
    with st.spinner("Đang tải mô hình đã lưu..."):
        try:
            model = load_model(MODEL_PATH)
            scaler = load_model(MODEL_PATH.replace('_model.pkl', '_scaler.pkl'))
            
            if model is None or scaler is None:
                st.warning("Không tìm thấy mô hình đã lưu. Vui lòng train mô hình mới.")
                return
            
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.model_trained = True
            st.session_state.feature_cols = get_feature_columns()
            
            st.success("Đã load mô hình thành công!")
            
        except Exception as e:
            st.error(f"Lỗi khi load mô hình: {e}")


def make_prediction():
    """Make prediction for next day"""
    if st.session_state.model is None or st.session_state.df_features is None:
        st.warning("Vui lòng tải dữ liệu và mô hình trước!")
        return
    
    if 'scaler' not in st.session_state or st.session_state.scaler is None:
        st.warning("Không tìm thấy scaler. Vui lòng train lại mô hình!")
        return
    
    with st.spinner("Đang dự đoán..."):
        try:
            df = st.session_state.df_features
            latest_row = df.iloc[-1]
            
            # Check if RF_Pred_Tomorrow is NaN for the latest row
            is_prediction_missing = pd.isna(latest_row.get('RF_Pred_Tomorrow'))
            
            if is_prediction_missing:
                # Use the latest row's features to predict for its "tomorrow"
                df_clean = df.dropna(subset=st.session_state.feature_cols)
                if df_clean.empty:
                    st.error("Không đủ dữ liệu sạch để dự đoán!")
                    return
                
                # We want the features of the VERY LAST row to predict its tomorrow
                latest_features = df[st.session_state.feature_cols].iloc[-1:].values
                
                # Make prediction with confidence
                pred_result = create_prediction_with_confidence(
                    st.session_state.model, 
                    st.session_state.scaler,
                    latest_features
                )
                
                # Date of prediction is the date of the latest row + 1 day
                pred_date = get_next_trading_date(latest_row['Date'])
                
                # Store prediction for display
                st.session_state.prediction = {
                    'date': pred_date,
                    'predicted_price': pred_result['prediction'],
                    'lower_bound': pred_result['lower_bound'],
                    'upper_bound': pred_result['upper_bound'],
                    'std': pred_result['std'],
                    'current_price': latest_row['Price'],
                    'is_new_prediction': True
                }
                
                st.success(f"Đã tạo dự đoán mới cho ngày {pred_date.strftime('%d/%m/%Y')}!")
            else:
                st.info("Ngày cuối cùng đã có kết quả dự đoán trong dữ liệu.")
                # Optional: Show existing prediction
                st.session_state.prediction = {
                    'date': get_next_trading_date(latest_row['Date']),
                    'predicted_price': latest_row['RF_Pred_Tomorrow'],
                    'lower_bound': latest_row['RF_Pred_Tomorrow'] * 0.95, # Estimate
                    'upper_bound': latest_row['RF_Pred_Tomorrow'] * 1.05, # Estimate
                    'std': 0,
                    'current_price': latest_row['Price'],
                    'is_new_prediction': False
                }
            
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")
            import traceback
            st.error(traceback.format_exc())


def update_csv_with_prediction(prediction_val):
    """Update the latest row in CSV with the prediction value"""
    try:
        df_csv = pd.read_csv(DATA_PATH)
        # Assuming Date is unique and sorted
        df_csv.iloc[-1, df_csv.columns.get_loc('RF_Pred_Tomorrow')] = prediction_val
        df_csv.to_csv(DATA_PATH, index=False)
        return True
    except Exception as e:
        st.error(f"Lỗi khi cập nhật CSV: {e}")
        return False


def display_dashboard():
    """Display main dashboard"""
    df = st.session_state.df_features
    
    # Latest data section - Only show latest date and single row
    st.header("Dữ liệu mới nhất")
    
    latest = get_latest_row(df)
    
    # Display latest date prominently
    st.subheader(f"Ngày: {latest['Date'].strftime('%d/%m/%Y')}")
    
    # Metrics in one row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Giá đóng cửa",
            f"${format_number(latest['Price'])}",
            f"{format_number(latest['Return_1d'] if 'Return_1d' in latest else 0, 2)}%"
        )
    
    with col2:
        st.metric("Giá mở cửa", f"${format_number(latest['Open'])}")
    
    with col3:
        st.metric("Giá cao nhất", f"${format_number(latest['High'])}")
    
    with col4:
        st.metric("Giá thấp nhất", f"${format_number(latest['Low'])}")
    
    with col5:
        st.metric("Khối lượng", f"{int(latest['Vol']):,}")
    
    # Show only the latest row in a clean table
    st.subheader("Chi tiết dòng dữ liệu mới nhất")
    
    # Determine which columns to show as requested by user
    base_cols = ['Date', 'Price', 'Open', 'High', 'Low', 'Vol']
    
    # Add RF prediction columns if they exist in the dataframe
    display_cols = base_cols.copy()
    if 'RF_Pred_Tomorrow' in df.columns:
        display_cols.append('RF_Pred_Tomorrow')
    if 'RF_Pred_Today' in df.columns:
        display_cols.append('RF_Pred_Today')
    
    latest_row_df = df[display_cols].tail(1).copy()
    latest_row_df['Date'] = latest_row_df['Date'].dt.strftime('%d/%m/%Y')
    
    # Format numeric columns
    for col in ['Price', 'Open', 'High', 'Low', 'RF_Pred_Tomorrow', 'RF_Pred_Today']:
        if col in latest_row_df.columns:
            latest_row_df[col] = latest_row_df[col].apply(lambda x: f"${x:.4f}" if pd.notna(x) else "N/A")
    
    if 'Vol' in latest_row_df.columns:
        latest_row_df['Vol'] = latest_row_df['Vol'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
    
    st.dataframe(latest_row_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Display prediction if available (MOVED HERE - between data and charts)
    if 'prediction' in st.session_state:
        display_prediction_inline()
        st.markdown("---")
    
    # Charts section
    st.header("Phân tích giá")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Lịch sử giá", "Nến Nhật", "Khối lượng", "Chỉ số kỹ thuật"])
    
    with tab1:
        fig = plot_price_history(df, n_days=100)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = plot_candlestick(df, n_days=60)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = plot_volume(df, n_days=60)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        fig = plot_technical_indicators(df, n_days=60)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model performance
    if st.session_state.model_trained and 'metrics' in st.session_state:
        st.markdown("---")
        st.header("Hiệu suất mô hình")
        
        col1, col2 = st.columns(2)
        
        with col1:
            metrics = st.session_state.metrics
            st.metric("Mean Absolute Error", f"{metrics['MAE']:.6f}")
            st.metric("Root Mean Squared Error", f"{metrics['RMSE']:.6f}")
        
        with col2:
            st.metric("R² Score", f"{metrics['R2']:.4f}")
            st.metric("Độ chính xác hướng", f"{metrics['Direction_Accuracy']:.2f}%")
        
        # Feature importance
        if st.checkbox("Hiển thị độ quan trọng của features"):
            feature_imp = get_feature_importance(st.session_state.model, st.session_state.feature_cols, top_n=20)
            fig = plot_feature_importance(feature_imp, top_n=20)
            st.plotly_chart(fig, use_container_width=True)


def display_prediction_inline():
    """Display prediction results inline (between data and charts)"""
    if 'prediction' not in st.session_state:
        return
    
    pred = st.session_state.prediction
    
    st.header("Kết quả dự đoán")
    
    # Beautiful prediction card with gradient
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 3rem 2rem;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
            margin: 2rem 0;
        ">
            <h3 style="color: white; margin-bottom: 1rem; font-size: 1.3rem;">
                Giá dự đoán cho ngày
            </h3>
            <h2 style="color: white; margin-bottom: 2rem; font-size: 1.5rem;">
                {pred['date'].strftime('%Y-%m-%d')}
            </h2>
            <h1 style="color: white; font-size: 3.5rem; font-weight: bold; margin: 1.5rem 0;">
                ${format_number(pred['predicted_price'])}
            </h1>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-top: 2rem;">
                Khoảng tin cậy 95%
            </p>
            <p style="color: white; font-size: 1.3rem; font-weight: 500; margin-top: 0.5rem;">
                ${format_number(pred['lower_bound'])} - ${format_number(pred['upper_bound'])}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Comparison metrics in 3 columns
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    change = pred['predicted_price'] - pred['current_price']
    change_pct = (change / pred['current_price']) * 100
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <p style="color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">Giá hiện tại</p>
            <h2 style="color: #333; font-size: 2rem; margin: 0;">${format_number(pred['current_price'])}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "#00c853" if change >= 0 else "#ff1744"
        arrow = "↑" if change >= 0 else "↓"
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <p style="color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">Thay đổi dự đoán</p>
            <h2 style="color: {color}; font-size: 2rem; margin: 0;">
                ${format_number(abs(change))}
            </h2>
            <p style="color: {color}; font-size: 1.2rem; margin-top: 0.3rem;">
                {arrow} {format_number(abs(change_pct), 2)}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <p style="color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">Độ lệch chuẩn</p>
            <h2 style="color: #333; font-size: 2rem; margin: 0;">${format_number(pred['std'])}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Action button
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.markdown("""
        <style>
        .stButton > button {
            background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
            color: white;
            font-weight: bold;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 30px;
            font-size: 1.1rem;
            width: 100%;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 210, 255, 0.4);
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button("Lưu dự đoán vào CSV", use_container_width=True):
            save_prediction_to_csv()


def save_prediction_to_csv():
    """Save prediction to CSV file"""
    if 'prediction' not in st.session_state:
        st.warning("Không có dự đoán để lưu!")
        return
    
    pred = st.session_state.prediction
    is_new_prediction = pred.get('is_new_prediction', True)
    
    if is_new_prediction:
        # Check if we should update an existing row (where RF_Pred_Tomorrow was NaN)
        # or append a completely new row.
        # If the prediction date matches the "tomorrow" of the last row in df
        df = st.session_state.df_features
        latest_date = df.iloc[-1]['Date']
        
        # If the prediction is indeed for the 'tomorrow' of the last existing row
        # we update that row's RF_Pred_Tomorrow column
        success = update_csv_with_prediction(pred['predicted_price'])
        
        if success:
            st.success(f"Đã cập nhật dự đoán cho ngày {pred['date'].strftime('%d/%m/%Y')} vào dữ liệu hiện có!")
            load_and_process_data() # Reload to show updated data
        else:
            # Fallback to append if update fails or logic dictates
            prediction_data = {
                'Date': pred['date'],
                'Price': pred['predicted_price'],
                'Open': pred['predicted_price'],
                'High': pred['upper_bound'],
                'Low': pred['lower_bound'],
                'Vol': 0
            }
            if append_prediction_to_csv(DATA_PATH, prediction_data):
                st.success("Đã thêm dòng dự đoán mới vào CSV!")
                load_and_process_data()
            else:
                st.error("Lưu dự đoán thất bại")
    else:
        st.info("Dự đoán này đã tồn tại trong tệp dữ liệu.")


if __name__ == "__main__":
    main()
