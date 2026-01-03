import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import traceback

sys.path.append(os.path.dirname(__file__))

from utils import (
    load_data, get_latest_row, get_latest_n_rows,
    create_advanced_features, get_feature_columns, create_lstm_features,
    train_layer1_model, train_svr_model, train_layer2_model, load_model, save_model,
    predict_next_day_layer1, predict_layer2, create_prediction_with_confidence,
    evaluate_model, get_feature_importance, prepare_data_for_training,
    predict_multi_step_layer1, train_multi_horizon_models, 
    train_lstm_model, prepare_lstm_data, predict_lstm,
    plot_price_history, plot_candlestick, plot_volume,
    plot_technical_indicators, plot_prediction_result, plot_feature_importance,
    plot_prediction_30d,
    get_next_trading_date, format_number, calculate_change_percent,
    append_prediction_to_csv, validate_data
)

# Import new modules for Layer 3 enhancements
from utils.regime_lstm import (
    RegimeLSTM, create_regime_features, prepare_regime_data,
    train_regime_lstm, predict_regime_lstm
)
from utils.ml_ensemble import (
    MLEnsembleForecaster, train_ml_ensemble, create_ml_features
)

# Page config
st.set_page_config(
    page_title="Hệ thống Dự báo Giá XRP Đa tầng",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Global Font & Theme */
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        color: #31333F; /* Dark text for light background */
    }
    
    /* Header styling */
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0E1117; /* Very dark blue-black */
        padding-bottom: 1rem;
        border-bottom: 2px solid #0E1117;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1F2937; /* Dark gray */
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        border-left: 5px solid #FF4B4B; /* Red Accent */
        padding-left: 10px;
    }
    
    /* Custom Containers */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background-color: #F8F9FA; /* Light gray background */
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Global Secondary Buttons (Blue) */
    .stButton > button:not([kind="primary"]) {
        background-color: #007BFF !important;
        color: white !important;
        border: none !important;
        transition: background-color 0.2s;
    }
    .stButton > button:not([kind="primary"]):hover {
        background-color: #0062cc !important;
    }
    
    /* Primary buttons styling (Force Red) */
    .stButton > button[kind="primary"] {
        background-color: #FF4B4B !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #e63939 !important;
    }
    
    /* Buttons Styling */
    .stButton > button {
        border-radius: 4px;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }

    /* Cards/Prediction Box */
    .prediction-box {
        background: #25262b;
        border: 1px solid #373a40;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #25262b;
        border: 1px solid #373a40;
        padding: 10px 15px;
        border-radius: 6px;
    }
    
    /* Tabs styling - Modern & Clean */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
        padding: 10px 0;
        border-bottom: 1px solid #E5E7EB;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 0;
        color: #6B7280; /* Neutral Gray */
        font-weight: 500;
        background-color: transparent;
        border: none;
        padding: 0 4px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #FF4B4B;
        background-color: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #FF4B4B !important; /* Professional Red */
        font-weight: 700;
        border-bottom: 2px solid #FF4B4B !important;
    }
    
    hr {
        margin: 2rem 0;
        border-color: #E5E7EB;
    }
    
    /* Dashboard Market Card */
    .market-card-container {
        background-color: #ffffff;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .market-label {
        color: #6B7280;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    
    .market-value-lg {
        color: #111827;
        font-size: 2.25rem;
        font-weight: 700;
        line-height: 2.5rem;
    }
    
    .market-value-sm {
        color: #1F2937;
        font-size: 1.25rem;
        font-weight: 600;
    }
    
    .trend-up { color: #059669; font-weight: 600; }
    .trend-down { color: #DC2626; font-weight: 600; }
    
    /* Input Form Styling */
    .input-form-box {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 20px;
    }

    /* Prediction Card Styling - Scientific/Professional Light Theme */
    .prediction-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    .prediction-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border-color: #D1D5DB;
    }
    .pred-title {
        color: #374151;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .pred-price {
        color: #111827;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 1rem 0;
        font-variant-numeric: tabular-nums;
    }
    .pred-sub {
        color: #6B7280;
        font-size: 0.875rem;
        margin-bottom: 1rem;
    }
    .confidence-box {
        background-color: #F3F4F6;
        border-radius: 6px;
        padding: 8px;
        margin-top: 1rem;
        font-size: 0.875rem;
        color: #4B5563;
    }
</style>
""", unsafe_allow_html=True)

# Helper to get absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DISPLAY_DATA_PATH = os.path.join(BASE_DIR, 'data', 'XRPUSDT_train.csv')
SOURCE_DATA_PATH = os.path.join(BASE_DIR, 'data', 'XRPUSDT20182024new.csv')

# Layer 1 paths
L1_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'layer1_rf_model.pkl')
L1_SCALER_PATH = os.path.join(BASE_DIR, 'models', 'layer1_scaler.pkl')
L1_MULTI_MODELS_PATH = os.path.join(BASE_DIR, 'models', 'layer1_multi_models.pkl')
L1_MULTI_SCALERS_PATH = os.path.join(BASE_DIR, 'models', 'layer1_multi_scalers.pkl')
L1_SVR_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'layer1_svr_model.pkl')
L1_SVR_SCALER_PATH = os.path.join(BASE_DIR, 'models', 'layer1_svr_scaler.pkl')
# Layer 2 paths (Single Ridge Stacking)
L2_RIDGE_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'l2_ridge_model.pkl')
L2_RIDGE_SCALER_PATH = os.path.join(BASE_DIR, 'models', 'l2_ridge_scaler.pkl')
# Layer 3 paths
L3_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'layer3_lstm_model.keras')
L3_SCALER_PATH = os.path.join(BASE_DIR, 'models', 'layer3_scaler.pkl')
L3_TARGET_SCALER_PATH = os.path.join(BASE_DIR, 'models', 'layer3_target_scaler.pkl')
ML_ENSEMBLE_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ml_ensemble_model.pkl')
REGIME_LSTM_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'regime_lstm_model.pkl')
REGIME_LSTM_SCALERS_PATH = os.path.join(BASE_DIR, 'models', 'regime_lstm_scalers.pkl')

# Session state initialization
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'l1_multi_models' not in st.session_state:
    st.session_state.l1_multi_models = None
if 'l1_multi_scalers' not in st.session_state:
    st.session_state.l1_multi_scalers = None
if 'svr_model' not in st.session_state:
    st.session_state.svr_model = None
if 'svr_scaler' not in st.session_state:
    st.session_state.svr_scaler = None
if 'svr_model_trained' not in st.session_state:
    st.session_state.svr_model_trained = False
if 'df_features' not in st.session_state:
    st.session_state.df_features = None
if 'show_manual_input' not in st.session_state:
    st.session_state.show_manual_input = False
if 'active_data_path' not in st.session_state:
    st.session_state.active_data_path = DISPLAY_DATA_PATH
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'svr_metrics' not in st.session_state:
    st.session_state.svr_metrics = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None

# Layer 2 Session States (Single Ridge Stacking)
if 'l2_ridge_model_trained' not in st.session_state:
    st.session_state.l2_ridge_model_trained = False
if 'l2_ridge_model' not in st.session_state:
    st.session_state.l2_ridge_model = None
if 'l2_ridge_scaler' not in st.session_state:
    st.session_state.l2_ridge_scaler = None

# Layer 3 Session States
if 'l3_model_trained' not in st.session_state:
    st.session_state.l3_model_trained = False
if 'l3_model' not in st.session_state:
    st.session_state.l3_model = None
if 'l3_scaler' not in st.session_state:
    st.session_state.l3_scaler = None
if 'l3_target_scaler' not in st.session_state:
    st.session_state.l3_target_scaler = None

# Regime LSTM Session States (Layer 3 - Tab 2)
if 'regime_lstm_trained' not in st.session_state:
    st.session_state.regime_lstm_trained = False
if 'regime_lstm_model' not in st.session_state:
    st.session_state.regime_lstm_model = None
if 'regime_lstm_scalers' not in st.session_state:
    st.session_state.regime_lstm_scalers = None
if 'regime_lstm_metrics' not in st.session_state:
    st.session_state.regime_lstm_metrics = None

# ML Ensemble Session States (Layer 3 - Tab 3)
if 'ml_ensemble_trained' not in st.session_state:
    st.session_state.ml_ensemble_trained = False
if 'ml_ensemble_model' not in st.session_state:
    st.session_state.ml_ensemble_model = None
if 'ml_ensemble_metrics' not in st.session_state:
    st.session_state.ml_ensemble_metrics = None

def main():

    # Header
    st.markdown('<h1 class="main-header">HỆ THỐNG DỰ ĐOÁN GIÁ CỔ PHIẾU XRP/USDT</h1>', unsafe_allow_html=True)
    
    # Sidebar
    # with st.sidebar:
    #     st.header("Bảng Điều Khiển")
    #     st.markdown("""
    #     **Kiến trúc Hệ thống:**
    #     1. **Lớp 1 (Máy học)**: Xác định Xu hướng (RandomForest, SVR)
    #     2. **Lớp 2 (Thống kê)**: Tinh chỉnh trong ngày (Ridge)
    #     3. **Lớp 3 (Học sâu)**: Dự báo chuỗi thời gian (LSTM)
    #     """)
        

    # Tabs for different Layers
    tab1, tab2, tab3 = st.tabs(["Dự đoán ngày tiếp theo", "Dự đoán trong ngày", "Dự đoán dài hạn"])
    
    with tab1:
        display_layer1_content()
    
    with tab2:
        display_layer2_content()
        
    with tab3:
        display_layer3_content()


def display_layer1_content():
    """Giao diện chính Layer 1 với bố cục lưới tối ưu"""
    
    # --- HÀNG 1: NHẬP LIỆU & HUẤN LUYỆN ---
    col_top_left, col_top_right = st.columns([1, 1.8])
    
    with col_top_left:
        st.markdown('<div class="section-header">1. NHẬP DỮ LIỆU</div>', unsafe_allow_html=True)
        with st.container(border=True):
            uploaded_file = st.file_uploader("CSV/Excel file", type=['csv', 'xlsx'], label_visibility="collapsed")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                # Nút Xử lý luôn hiện nhưng disabled nếu chưa chọn file để giữ layout
                if st.button("Xử lý", use_container_width=True, type="primary", disabled=(uploaded_file is None)):
                    load_and_process_data(uploaded_file)
            with col_btn2:
                # Nút Xem Data (sẽ tự động có màu xanh theo CSS secondary)
                if st.button("Xem Data", use_container_width=True, key="btn_view_data_l1"):
                    st.session_state.show_processed_data = not st.session_state.get('show_processed_data', False)

    with col_top_right:
        st.markdown('<div class="section-header">2. HUẤN LUYỆN MÔ HÌNH</div>', unsafe_allow_html=True)
        with st.container(border=True):
            # Trạng thái mô hình
            rf_status = "Đã train" if st.session_state.model_trained else "Chưa train"
            svr_status = "Đã train" if st.session_state.svr_model_trained else "Chưa train"
            st.caption(f"Trạng thái: RF [{rf_status}] | SVR [{svr_status}]")
            
            # Nút huấn luyện & Tinh chỉnh
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                with st.expander("Tham số RF", expanded=False):
                    rf_n_estimators = st.number_input("n_estimators", 10, 2000, 500, 50, key="rf_n")
                    rf_max_depth = st.number_input("max_depth", 1, 50, 8, 1, key="rf_d")
                    rf_min_leaf = st.number_input("min_samples_leaf", 1, 100, 20, 1, key="rf_l")
                
                if st.button("Train RandomForest", use_container_width=True, type="primary"):
                    params = {
                        'n_estimators': rf_n_estimators,
                        'max_depth': rf_max_depth,
                        'min_samples_leaf': rf_min_leaf,
                        'random_state': 42,
                        'n_jobs': -1
                    }
                    train_model(model_type="RF", custom_params=params)
                    
            with col_t2:
                with st.expander("Tham số SVR", expanded=False):
                    svr_c = st.number_input("C (Regularization)", 0.01, 1000.0, 100.0, 10.0, key="svr_c")
                    svr_epsilon = st.number_input("Epsilon", 0.001, 1.0, 0.01, 0.005, format="%.3f", key="svr_e")
                    svr_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], index=0, key="svr_k")
                
                if st.button("Train SVR (Vector)", use_container_width=True, type="primary"):
                    params = {
                        'C': svr_c,
                        'epsilon': svr_epsilon,
                        'kernel': svr_kernel,
                        'gamma': 'scale'
                    }
                    train_model(model_type="SVR", custom_params=params)
            
            st.divider()
            
            # Quản lý file mô hình (Dùng Expander để tiết kiệm diện tích)
            with st.expander("Tải mô hình", expanded=False):
                m_tab1, m_tab2 = st.tabs(["Tải Mô hình", "Xóa Mô hình"])
                with m_tab1:
                    model_options = ["Tất cả (All)", "RandomForest (1-Day)", "SVR (1-Day)", "Dự báo 7-Ngày"]
                    selected_load = st.selectbox("Chọn để tải:", model_options, label_visibility="collapsed")
                    if st.button("Tải Mô hình Đã chọn", use_container_width=True, type="primary"):
                        if selected_load == "Tất cả (All)":
                            load_saved_model(model_type="ALL")
                            load_saved_7day_models()
                        elif selected_load == "RandomForest (1-Day)":
                            load_saved_model(model_type="RF")
                        elif selected_load == "SVR (1-Day)":
                            load_saved_model(model_type="SVR")
                        elif selected_load == "Dự báo 7-Ngày":
                            load_saved_7day_models()
                with m_tab2:
                    files_to_delete = st.multiselect(
                        "File cần xóa:",
                        ["RandomForest", "SVR", "Dự báo 7-Ngày", "Layer 2 Models", "Layer 3 LSTM"],
                        default=[]
                    )
                    if st.button("Xóa Mô hình Đã chọn", type="primary", use_container_width=True):
                        if files_to_delete:
                            delete_selected_models(files_to_delete)
                
            # Thông tin mô hình hiện tại (Mới)
            if st.session_state.model_trained or st.session_state.svr_model_trained:
                with st.expander("Chi tiết mô hình đang tải", expanded=False):
                    if st.session_state.model_trained and st.session_state.model is not None:
                        st.markdown("**RandomForest (RF):**")
                        p = st.session_state.model.get_params()
                        st.json({k: p[k] for k in ['n_estimators', 'max_depth', 'min_samples_leaf'] if k in p})
                    
                    if st.session_state.svr_model_trained and st.session_state.svr_model is not None:
                        st.markdown("**SVR (Support Vector):**")
                        p = st.session_state.svr_model.get_params()
                        st.json({k: p[k] for k in ['C', 'epsilon', 'kernel', 'gamma'] if k in p})

    # Hiển thị bảng dữ liệu (Toggle) - Full width bên dưới Row 1
    if st.session_state.get('show_processed_data', False):
        if st.session_state.df_features is not None:
            st.toast(f"Đang hiển thị {len(st.session_state.df_features)} dòng đã xử lý.")
            st.dataframe(st.session_state.df_features, use_container_width=True, height=300)
        else:
            st.warning("Vui lòng tải dữ liệu trước.")

    st.divider()

    # --- HÀNG 2: THỊ TRƯỜNG & DỰ BÁO & CẬP NHẬT ---
    if st.session_state.df_features is not None:
        # Load df_display để dùng cho các component bên dưới
        df_display = None
        if os.path.exists(DISPLAY_DATA_PATH):
            try: df_display = pd.read_csv(DISPLAY_DATA_PATH)
            except: pass

        col_main_left, col_main_right = st.columns([2.3, 1])
        
        with col_main_left:
            # 1. DỮ LIỆU THỊ TRƯỜNG MỚI NHẤT (70% bề ngang)
            display_market_status_card(st.session_state.df_features, df_display)
            
            # 1.5 KẾT QUẢ DỰ BÁO SO SÁNH (Đưa vào khoảng trống bên trái)
            if 'prediction' in st.session_state:
                st.write("") # Spacer
                display_prediction_inline()
        
        with col_main_right:
            # 2. THỰC HIỆN DỰ ĐOÁN (Phần trên - 30% ngang)
            st.markdown('<div class="section-header">3. DỰ ĐOÁN</div>', unsafe_allow_html=True)
            # Hiển thị các nút xếp chồng theo hàng dọc
            with st.container(border=True):
                if st.button("DỰ ĐOÁN T+1", use_container_width=True, type="primary", 
                             disabled=not (st.session_state.model_trained or st.session_state.svr_model_trained)):
                    make_prediction()
                st.write("")
                if st.button("DỰ ĐOÁN T+7", use_container_width=True, 
                             disabled=not (st.session_state.model_trained or st.session_state.svr_model_trained)):
                    make_7day_prediction()
                st.caption("*Yêu cầu mô hình đã sẵn sàng.")

            # 3. CẬP NHẬT DỮ LIỆU THỰC TẾ (Phần dưới - 30% ngang)
            display_manual_input_form()

        # --- HÀNG 3: KẾT QUẢ & PHÂN TÍCH ---
        st.divider()
        display_prediction_results_and_charts(st.session_state.df_features, df_display)
    else:
        # Khi chưa có dữ liệu
        col1, col2 = st.columns([2.3, 1])
        with col1:
            st.markdown('<div class="section-header">DỮ LIỆU THỊ TRƯỜNG</div>', unsafe_allow_html=True)
            st.info("Vui lòng bắt đầu bằng việc **Tải dữ liệu** ở Mục 1.")
        with col2:
            st.markdown('<div class="section-header">3. DỰ ĐOÁN</div>', unsafe_allow_html=True)
            st.warning("Vui lòng tải dữ liệu trước.")
        # st.info("")


#### load dữ liệu
def load_and_process_data(file_buffer=None, target_path=None):
    with st.spinner("Đang tải và xử lý dữ liệu..."):
        try:
            # Load data
            if file_buffer is not None:
                df = pd.read_csv(file_buffer)
                df['Date'] = pd.to_datetime(df['Date'])
                df.drop(columns=['Change %'], errors='ignore', inplace=True)
                df = df.sort_values('Date').reset_index(drop=True)
            else:
                path = target_path if target_path else SOURCE_DATA_PATH
                df = load_data(path)
            
            # Validate
            is_valid, msg = validate_data(df)
            if not is_valid:
                st.error(f"Dữ liệu không hợp lệ: {msg}")
                return
            
            # Create features
            df_features = create_advanced_features(df)
            
            # # Sync RF & SVR predictions if they exist in the loaded file
            # if 'RF_Pred_Tomorrow' in df_features.columns:
            #     df_features['RF_Pred_Today'] = df_features['RF_Pred_Tomorrow'].shift(1)
            # if 'SVR_Pred_Tomorrow' in df_features.columns:
            #     df_features['SVR_Pred_Today'] = df_features['SVR_Pred_Tomorrow'].shift(1)
                
            # Store in session state
            st.session_state.df_features = df_features
            
            # Update active path
            if file_buffer is not None:
                # For uploaded files, save a local copy with its original name in the data folder
                save_filename = file_buffer.name
                save_path = os.path.join(BASE_DIR, 'data', save_filename)
                df.to_csv(save_path, index=False)
                st.session_state.active_data_path = save_path
            else:
                st.session_state.active_data_path = path
                
            st.toast(f"Đã tải {len(df)} dòng dữ liệu thành công! (Nguồn: {st.session_state.active_data_path})")
            
        except Exception as e:
            st.error(f"Lỗi khi xử lý dữ liệu: {e}")


#### train model
def train_model(model_type="RF", custom_params=None):
    if st.session_state.df_features is None:
        st.warning("Vui lòng tải dữ liệu trước!")
        return
    
    model_name = "RandomForest" if model_type == "RF" else "SVR"
    with st.spinner(f"Đang huấn luyện mô hình {model_name}..."):
        try:
            # Get feature columns
            # Lấy danh sách feature chuẩn (95 cột)
            feature_cols = get_feature_columns()
            st.session_state.feature_cols = feature_cols
            
            # st.info(f"Đang chuẩn bị dữ liệu với {len(feature_cols)} features...")
            st.toast(f"Đang chuẩn bị dữ liệu với {len(feature_cols)} features...")
            
            # Prepare data
            X_train, X_test, y_train, y_test, _ = prepare_data_for_training(
                st.session_state.df_features,
                feature_columns=feature_cols,
                target_column='Target_Price',
                test_size=0.2
            )
            
            # Lưu danh sách features vào session state để dùng khi dự đoán
            st.session_state.feature_cols = feature_cols
            
            if model_type == "RF":
                # Train RF
                model, scaler = train_layer1_model(X_train, y_train, params=custom_params)
                save_model(model, L1_MODEL_PATH)
                save_model(scaler, L1_SCALER_PATH)
                
                # Store in session state
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.model_trained = True
                
                # Add predictions to dataframe
                # df_clean = st.session_state.df_features.dropna(subset=feature_cols + ['Target_Price'])
                # X_all_scaled = scaler.transform(df_clean[feature_cols])

                # predictions = model.predict(X_all_scaled)
                # st.session_state.df_features.loc[df_clean.index, 'RF_Pred_Tomorrow'] = predictions
                # st.session_state.df_features['RF_Pred_Today'] = st.session_state.df_features['RF_Pred_Tomorrow'].shift(1)
            else:
                # Train SVR
                model, scaler = train_svr_model(X_train, y_train, params=custom_params)
                save_model(model, L1_SVR_MODEL_PATH)
                save_model(scaler, L1_SVR_SCALER_PATH)
                
                # Store in session state
                st.session_state.svr_model = model
                st.session_state.svr_scaler = scaler
                st.session_state.svr_model_trained = True
                
                # Add predictions to dataframe
                # df_clean = st.session_state.df_features.dropna(subset=feature_cols + ['Target_Price'])
                # X_all_scaled = scaler.transform(df_clean[feature_cols])
                # predictions = model.predict(X_all_scaled)
                # st.session_state.df_features.loc[df_clean.index, 'SVR_Pred_Tomorrow'] = predictions
                # st.session_state.df_features['SVR_Pred_Today'] = st.session_state.df_features['SVR_Pred_Tomorrow'].shift(1)

            # Evaluate
            metrics = evaluate_model(model, scaler, X_test, y_test)
            # st.session_state.feature_cols = feature_cols
            
            # Display metrics
            # Display metrics
            # st.success(f"Huấn luyện mô hình {model_name} thành công!")
            st.toast(f"Huấn luyện mô hình {model_name} thành công!")
            
            # Store metrics specifically
            if model_type == "RF":
                st.session_state.metrics = metrics
            else:
                st.session_state.svr_metrics = metrics
                
        except Exception as e:
            st.error(f"Lỗi khi huấn luyện mô hình {model_name}: {e}")
            import traceback
            st.error(traceback.format_exc())


def load_saved_model(model_type="ALL"):
    """Load pre-trained models Layer 1 (RF & SVR) based on selection"""
    with st.spinner(f"Đang tải quy trình mô hình: {model_type}..."):
        try:
            # Feature columns are shared - lấy 1 lần để dùng chung
            if st.session_state.feature_cols is None:
                st.session_state.feature_cols = get_feature_columns()
            
            loaded_any = False
            
            # --- Load RF ---
            if model_type in ["ALL", "RF"]:
                try:
                    rf_model = load_model(L1_MODEL_PATH)
                    rf_scaler = load_model(L1_SCALER_PATH)
                    
                    if rf_model and rf_scaler:
                        # Check feature consistency
                        if hasattr(rf_scaler, 'n_features_in_') and rf_scaler.n_features_in_ != len(st.session_state.feature_cols):
                            st.error(f"⚠️ Scaler RF cũ ({rf_scaler.n_features_in_} cột) không khớp với {len(st.session_state.feature_cols)} cột hiện tại. Vui lòng train lại!")
                        else:
                            st.session_state.model = rf_model
                            st.session_state.scaler = rf_scaler
                            st.session_state.model_trained = True
                            loaded_any = True
                            st.toast("Đã tải Random Forest thành công!")
                except Exception as e:
                    st.warning(f"Không thể tải RF: {e}")

            # --- Load SVR ---
            if model_type in ["ALL", "SVR"]:
                try:
                    svr_model = load_model(L1_SVR_MODEL_PATH)
                    svr_scaler = load_model(L1_SVR_SCALER_PATH)
                    
                    if svr_model and svr_scaler:
                        # Check feature consistency
                        if hasattr(svr_scaler, 'n_features_in_') and svr_scaler.n_features_in_ != len(st.session_state.feature_cols):
                            st.error(f"⚠️ Scaler SVR cũ ({svr_scaler.n_features_in_} cột) không khớp với {len(st.session_state.feature_cols)} cột hiện tại. Vui lòng train lại!")
                        else:
                            st.session_state.svr_model = svr_model
                            st.session_state.svr_scaler = svr_scaler
                            st.session_state.svr_model_trained = True
                            loaded_any = True
                            st.toast("Đã tải SVR thành công!")
                except Exception as e:
                    st.warning(f"Không thể tải SVR: {e}")
            
            if not loaded_any:
                st.warning(f"Không tìm thấy mô hình {model_type} hợp lệ nào đã lưu.")
            
        except Exception as e:
            st.error(f"Lỗi chung khi tải mô hình: {e}")


def load_saved_7day_models():
    """Load pre-trained 7-day models"""
    with st.spinner("Đang tải bộ mô hình dự báo 7 ngày..."):
        try:
            if st.session_state.feature_cols is None:
                st.session_state.feature_cols = get_feature_columns()

            multi_models = load_model(L1_MULTI_MODELS_PATH)
            multi_scalers = load_model(L1_MULTI_SCALERS_PATH)
            
            if multi_models and multi_scalers:
                st.session_state.l1_multi_models = multi_models
                st.session_state.l1_multi_scalers = multi_scalers
                st.toast("Đã tải thành công bộ mô hình dự báo 7 ngày!")
            else:
                st.warning("Không tìm thấy file mô hình 7 ngày đã lưu.")
        except Exception as e:
            st.error(f"Lỗi khi tải mô hình 7 ngày: {e}")


def delete_selected_models(files_to_delete):
    """Xóa các mô hình được chọn"""
    # Mapping tên hiển thị -> đường dẫn file
    mapping = {
        "RandomForest": [L1_MODEL_PATH, L1_SCALER_PATH],
        "SVR": [L1_SVR_MODEL_PATH, L1_SVR_SCALER_PATH],
        "Dự báo 7-Ngày": [L1_MULTI_MODELS_PATH, L1_MULTI_SCALERS_PATH],
        "Layer 2 Models": ["models/layer2_ridge_model.pkl", "models/layer2_ridge_scaler.pkl", "models/layer2_svr_model.pkl", "models/layer2_svr_scaler.pkl"],
        "Layer 3 LSTM": ["models/layer3_lstm_model.keras", "models/layer3_scaler.pkl", "models/layer3_target_scaler.pkl"]
    }
    
    deleted_count = 0
    for key in files_to_delete:
        paths = mapping.get(key, [])
        for p in paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                    deleted_count += 1
                except Exception as e:
                    st.error(f"Không xóa được {p}: {e}")
    
    if deleted_count > 0:
        st.toast(f"Đã xóa {deleted_count} file mô hình thành công!", icon="🗑️")
        
        # Cập nhật lại session state sau khi xóa
        if "RandomForest" in files_to_delete:
            st.session_state.model = None
            st.session_state.model_trained = False
        if "SVR" in files_to_delete:
            st.session_state.svr_model = None
            st.session_state.svr_model_trained = False
            
        time.sleep(1) # Delay nhẹ để hiển thị toast
        st.rerun()
    else:
        st.warning("Không tìm thấy file nào để xóa (có thể đã bị xóa trước đó).")

                



# Nhấn dự đoán 1 ngày
def make_prediction():
    if st.session_state.df_features is None:
        st.warning("Vui lòng tải dữ liệu trước!")
        return
        
    if not st.session_state.model_trained and not st.session_state.svr_model_trained:
        st.warning("Chưa có mô hình nào được huấn luyện!")
        return
    
    with st.spinner("Đang tính toán dự đoán..."):
        try:
            df = st.session_state.df_features
            # Xuất dữ liệu ra file CSV để kiểm tra
            df.to_csv('debug_df_features.csv', index=False)
            print(f"Đã xuất dữ liệu df_features ra file: debug_df_features.csv")
            
            latest_row = df.iloc[-1]
            print("\n" + "="*60)
            print("DEBUG: CHI TIẾT DÒNG DỮ LIỆU CUỐI CÙNG (LATEST ROW)")
            print("-" * 64)
            print(latest_row.to_string())
            print("-" * 64)
            print("="*60 + "\n")
            
            # Prepare feature data (handle NaNs) - CHỈ LẤY CÁC CỘT FEATURES (Loại bỏ Date)
            # Lấy dòng cuối cùng của df (dòng mới nhất người dùng vừa nhập hoặc tải lên)
            feature_cols = st.session_state.feature_cols
            df_cleaned = df[feature_cols].copy().ffill().fillna(0)
            
            latest_features = df_cleaned.iloc[-1:].values
            pred_date = get_next_trading_date(latest_row['Date'])
            
            comparison_results = {}
            
            # Predict with RF if available
            if st.session_state.model_trained:
                pred_rf = create_prediction_with_confidence(
                    st.session_state.model, 
                    st.session_state.scaler,
                    latest_features
                )
                comparison_results['RF'] = {
                    'price': pred_rf['prediction'],
                    'lower': pred_rf['lower_bound'],
                    'upper': pred_rf['upper_bound']
                }
                
            # Predict with SVR if available
            # Predict with SVR if available
            if st.session_state.svr_model_trained:
                try:
                    # Kiểm tra số features trước khi transform
                    if hasattr(st.session_state.svr_scaler, 'n_features_in_'):
                        expected = st.session_state.svr_scaler.n_features_in_
                        actual = latest_features.shape[1]
                        if expected != actual:
                            raise ValueError(f"SVR Model cũ mong đợi {expected} features nhưng code mới cung cấp {actual}. Cần Train lại SVR!")

                    svr_pred_scaled = st.session_state.svr_model.predict(
                        st.session_state.svr_scaler.transform(latest_features)
                    )[0]
                    comparison_results['SVR'] = {
                        'price': svr_pred_scaled,
                        'lower': svr_pred_scaled * 0.98, # Theoretical interval
                        'upper': svr_pred_scaled * 1.02
                    }
                except Exception as e:
                    st.error(f"Lỗi SVR: {e}")
                    st.warning("Mô hình SVR hiện tại không tương thích với dữ liệu mới. Hệ thống sẽ bỏ qua SVR trong lần này. Vui lòng nhấn nút 'Train SVR' để huấn luyện lại!")
                    # Tạm thời vô hiệu hóa SVR để không gây lỗi tiếp
                    # st.session_state.svr_model_trained = False 
            
            st.session_state.prediction = {
                'date': pred_date,
                'current_price': latest_row['Price'],
                'results': comparison_results
            }
            
            # Thêm các phím phẳng cho tính tương thích với hàm lưu CSV (mặc định lấy RF)
            if 'RF' in comparison_results:
                st.session_state.prediction.update({
                    'predicted_price': comparison_results['RF']['price'],
                    'upper_bound': comparison_results['RF']['upper'],
                    'lower_bound': comparison_results['RF']['lower']
                })
            elif 'SVR' in comparison_results:
                st.session_state.prediction.update({
                    'predicted_price': comparison_results['SVR']['price'],
                    'upper_bound': comparison_results['SVR']['upper'],
                    'lower_bound': comparison_results['SVR']['lower']
                })
            
            st.toast("Đã cập nhật dự đoán so sánh!")
            st.rerun() # Buộc Streamlit chạy lại để hiển thị kết quả ngay lập tức
            
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")


def make_7day_prediction():
    """Make 7-day prediction using multi-horizon models (train on demand if needed)"""
    if st.session_state.df_features is None:
        st.warning("Vui lòng tải dữ liệu trước!")
        return

    # Check if multi-horizon models are already trained/loaded
    if st.session_state.l1_multi_models is None:
        with st.status("Đang huấn luyện bộ 7 mô hình chuyên biệt cho dự báo 7 ngày...", expanded=True) as status:
            try:
                st.write("Dữ liệu đang được chuẩn bị...")
                feature_cols = get_feature_columns()
                
                st.write("Bắt đầu huấn luyện (quy trình này có thể mất 1-2 phút)...")
                horizon_results = train_multi_horizon_models(st.session_state.df_features, feature_cols, days=7)
                
                multi_models = horizon_results['models']
                multi_scalers = horizon_results['scalers']
                
                # Save models
                save_model(multi_models, L1_MULTI_MODELS_PATH)
                save_model(multi_scalers, L1_MULTI_SCALERS_PATH)
                
                # Update session state
                st.session_state.l1_multi_models = multi_models
                st.session_state.l1_multi_scalers = multi_scalers
                st.session_state.feature_cols = feature_cols
                
                status.update(label="Đã huấn luyện xong bộ 7 mô hình!", state="complete", expanded=False)
            except Exception as e:
                status.update(label=f"Lỗi khi huấn luyện: {e}", state="error")
                return

    with st.spinner("Đang tính toán dự đoán cho 7 ngày tới..."):
        try:
            # Prepare df for history
            df = st.session_state.df_features
            
            # Predict using the 7 individual models
            forecast_df = predict_multi_step_layer1(
                st.session_state.l1_multi_models,
                st.session_state.l1_multi_scalers,
                df,
                st.session_state.feature_cols,
                create_advanced_features,
                days=7
            )
            
            # Store in session state
            st.session_state.prediction_7days = forecast_df
            st.toast("Đã hoàn thành dự đoán xu hướng 7 ngày!", icon="📈")
            st.rerun() # Buộc Streamlit chạy lại để hiển thị kết quả ngay lập tức
            
        except Exception as e:
            st.error(f"Lỗi khi dự đoán 7 ngày: {e}")
            import traceback
            st.error(traceback.format_exc())


def update_csv_with_prediction(prediction_val, col_name='RF_Pred_Tomorrow', target_path=None):
    """Update the latest row in CSV with the prediction value"""
    if target_path is None:
        target_path = st.session_state.get('active_data_path', DISPLAY_DATA_PATH)
    try:
        df_csv = pd.read_csv(target_path)
        # Nếu cột chưa có thì tạo mới
        if col_name not in df_csv.columns:
            df_csv[col_name] = pd.NA
            
        # Assuming Date is unique and sorted -> update last row
        # Xác định vị trí cột cẩn thận
        col_idx = df_csv.columns.get_loc(col_name)
        df_csv.iloc[-1, col_idx] = prediction_val
        
        df_csv.to_csv(target_path, index=False)
        return True
    except Exception as e:
        st.error(f"Lỗi khi cập nhật CSV: {e}")
        return False


def display_market_status_card(df, df_display):
    """Hiển thị thẻ trạng thái thị trường (Chỉ phần Card OHLV)"""
    # Latest data section - Only show latest date and single row
    st.markdown('<div class="section-header">DỮ LIỆU THỊ TRƯỜNG MỚI NHẤT</div>', unsafe_allow_html=True)
    
    latest = get_latest_row(df)
    
    # --- Custom Market Dashboard Card ---
    with st.container():
        # Tính toán change percent
        change_val = latest.get('Return_1d', 0)
        trend_class = "trend-up" if change_val >= 0 else "trend-down"
        trend_arrow = "▲" if change_val >= 0 else "▼"
        
        # HTML Custom Layout
        col_main, col_details = st.columns([1.5, 3])
        
        with col_main:
            st.markdown(f"""
<div style="padding: 10px;">
<div class="market-label">Ngày giao dịch</div>
<div style="font-size: 1.1rem; font-weight: 500; color: #374151; margin-bottom: 15px;">{latest['Date'].strftime('%d/%m/%Y')}</div>
<div class="market-label">Giá Đóng Cửa (Close)</div>
<div class="market-value-lg">${format_number(latest['Price'])}</div>
<div class="{trend_class}" style="margin-top: 5px; font-size: 1rem;">
{trend_arrow} {format_number(abs(change_val), 2)}%
</div>
</div>
""", unsafe_allow_html=True)
            
        with col_details:
            # Dùng HTML Grid thay vì st.columns để tránh lỗi lồng cột (Nested Columns)
            data_points = [
                ("Mở cửa (Open)", f"${format_number(latest['Open'])}"),
                ("Cao nhất (High)", f"${format_number(latest['High'])}"),
                ("Thấp nhất (Low)", f"${format_number(latest['Low'])}"),
                ("Volume", f"{int(latest['Vol']):,}")
            ]
            
            grid_html = '<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">'
            for label, fmt_val in data_points:
                grid_html += f'<div style="background: #F3F4F6; padding: 10px; border-radius: 8px; text-align: center;">'
                grid_html += f'<div class="market-label" style="font-size: 0.7rem;">{label}</div>'
                grid_html += f'<div class="market-value-sm" style="font-size: 0.9rem;">{fmt_val}</div>'
                grid_html += '</div>'
            grid_html += '</div>'
            
            st.markdown(grid_html, unsafe_allow_html=True)
            st.caption("Cập nhật từ file nguồn.")

    # Show only the latest row in a clean table
    with st.expander("Xem chi tiết dòng dữ liệu thô (Dòng cuối cùng)", expanded=False):
        # Determine which columns to show as requested by user
        base_cols = ['Date', 'Price', 'Open', 'High', 'Low', 'Vol']
        latest_row_df = df[base_cols].tail(1).copy()
    
        # Add prediction columns from DISPLAY_DATA_PATH if available
        if df_display is not None and not df_display.empty:
            last_display = df_display.iloc[-1]
            for c in ['RF_Pred_Tomorrow', 'RF_Pred_Today', 'SVR_Pred_Tomorrow', 'SVR_Pred_Today']:
                if c in df_display.columns:
                    latest_row_df[c] = last_display[c]
    
        latest_row_df['Date'] = latest_row_df['Date'].dt.strftime('%d/%m/%Y')
        for col in latest_row_df.columns:
            if col != 'Date' and col != 'Vol':
                latest_row_df[col] = latest_row_df[col].apply(lambda x: f"${x:.4f}" if pd.notna(x) else "N/A")
            elif col == 'Vol':
                latest_row_df[col] = latest_row_df[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
        
        st.dataframe(latest_row_df, use_container_width=True, hide_index=True)


def display_prediction_results_and_charts(df, df_display):
    """Hiển thị kết quả dự báo 7 ngày và biểu đồ phân tích (Phần dưới cùng)"""
    # 1-day prediction results have been moved to the Dashboard area
    
    # Display 7-day prediction if available
    if 'prediction_7days' in st.session_state:
        display_7day_prediction_inline()
        st.markdown("---")
    
    # Charts section
    st.header("Phân tích giá")
    tab1, tab2, tab3, tab4 = st.tabs(["Lịch sử Giá", "Biểu đồ Nến", "Khối lượng", "Chỉ báo Kỹ thuật"])
    
    with tab1:
        st.plotly_chart(plot_price_history(df, n_days=100), use_container_width=True)
    with tab2:
        st.plotly_chart(plot_candlestick(df, n_days=60), use_container_width=True)
    with tab3:
        st.plotly_chart(plot_volume(df, n_days=60), use_container_width=True)
    with tab4:
        st.plotly_chart(plot_technical_indicators(df, n_days=60), use_container_width=True)
    

def display_prediction_inline():
    """Display prediction results inline with comparison"""
    if 'prediction' not in st.session_state:
        return
    
    pred = st.session_state.prediction
    results = pred['results']
    
    st.markdown('<div class="section-header">KẾT QUẢ DỰ ĐOÁN</div>', unsafe_allow_html=True)
    
    # Display cards for each model
    cols = st.columns(len(results))
    
    for i, (m_type, data) in enumerate(results.items()):
        with cols[i]:
            title = "RandomForest" if m_type == "RF" else "SVR (Support Vector Regression)"
            
            change = data['price'] - pred['current_price']
            change_pct = (change / pred['current_price']) * 100
            
            # Professional Financial Colors
            color = "#00b894" if change >= 0 else "#ff7675" # Green/Red flat colors
            arrow = "▲" if change >= 0 else "▼"

            st.markdown(f"""
<div class="prediction-card">
<div class="pred-title">{title}</div>
<div class="pred-sub">Mục tiêu: {pred['date'].strftime('%d/%m/%Y')}</div>
<div class="pred-price">${format_number(data['price'])}</div>
<div style="margin-bottom: 1rem;">
<span style="color: {color}; font-weight: 700; font-size: 1.2rem; background: {color}15; padding: 4px 12px; border-radius: 20px;">
{arrow} {format_number(abs(change_pct), 2)}%
</span>
</div>
<div class="confidence-box">
<span style="display: block; font-size: 0.75rem; text-transform: uppercase; color: #6B7280; margin-bottom: 4px;">Khoảng tin cậy (95%)</span>
<span style="font-weight: 600; color: #374151;">${format_number(data['lower'])} - ${format_number(data['upper'])}</span>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- New Chart: 30d History + Prediction ---
    if st.session_state.df_features is not None:
        fig_context = plot_prediction_30d(
            st.session_state.df_features, 
            results, 
            pred['date']
        )
        st.plotly_chart(fig_context, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    # Save action for RF
    with col1:
        if 'RF' in results:
            if st.button("Lưu dự đoán RF vào CSV", use_container_width=True, type="primary"):
                save_prediction_to_csv(model_type='RF')

    # Save action for SVR
    with col2:
        if 'SVR' in results:
            if st.button("Lưu dự đoán SVR vào CSV", use_container_width=True, type="primary"):
                save_prediction_to_csv(model_type='SVR')


def display_7day_prediction_inline():
    """Display 7-day forecast results with table and chart"""
    st.header("Dự đoán giá 7 ngày")
    
    forecast_df = st.session_state.prediction_7days
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Bảng dự đoán")
        display_df = forecast_df.copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%d/%m/%Y')
        display_df['Predicted_Price'] = display_df['Predicted_Price'].apply(lambda x: f"${x:.4f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Biểu đồ Xu hướng")
        
        # Thêm giá hiện tại vào biểu đồ để thấy sự kết nối
        df_hist = st.session_state.df_features.tail(5)
        
        fig = go.Figure()
        
        # Đường giá lịch sử ngắn
        fig.add_trace(go.Scatter(
            x=df_hist['Date'], y=df_hist['Price'],
            mode='lines+markers', name='Thực tế',
            line=dict(color='blue')
        ))
        
        # Đường dự đoán
        # Kết nối điểm cuối thực tế với điểm đầu dự đoán
        x_pred = [df_hist['Date'].iloc[-1]] + forecast_df['Date'].tolist()
        y_pred = [df_hist['Price'].iloc[-1]] + forecast_df['Predicted_Price'].tolist()
        
        fig.add_trace(go.Scatter(
            x=x_pred, y=y_pred,
            mode='lines+markers', name='Dự đoán (7 ngày)',
            line=dict(color='orange', dash='dash')
        ))
        
        fig.update_layout(
            template='plotly_white',
            margin=dict(l=20, r=20, t=20, b=20),
            height=400,
            xaxis_title="Ngày",
            yaxis_title="Giá XRP ($)"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def save_prediction_to_csv(model_type='RF'):
    """Save prediction to CSV file"""
    target_path = st.session_state.get('active_data_path', DISPLAY_DATA_PATH)
    
    if 'prediction' not in st.session_state:
        st.warning("Không có dự đoán để lưu!")
        return
    
    pred = st.session_state.prediction
    results = pred['results']
    
    # Xác định giá trị và tên cột cần lưu dựa trên model_type
    if model_type == 'RF':
        if 'RF' not in results: return
        pred_price = results['RF']['price']
        target_col = 'RF_Pred_Tomorrow'
    else:  # SVR
        if 'SVR' not in results: return
        pred_price = results['SVR']['price']
        target_col = 'SVR_Pred_Tomorrow'

    is_new_prediction = pred.get('is_new_prediction', True)
    
    if is_new_prediction:
        # Check if we should update an existing row (where RF_Pred_Tomorrow was NaN)
        # or append a completely new row.
        # If the prediction date matches the "tomorrow" of the last row in df
        df = st.session_state.df_features
        latest_date = df.iloc[-1]['Date']
        
        # If the prediction is indeed for the 'tomorrow' of the last existing row
        # we update that row's RF_Pred_Tomorrow column
        success = update_csv_with_prediction(pred_price, col_name=target_col, target_path=target_path)
        
        if success:
            st.success(f"Đã cập nhật dự đoán {model_type} cho ngày {pred['date'].strftime('%d/%m/%Y')} vào dữ liệu hiện có!")
            load_and_process_data(target_path=target_path) # Reload từ file vừa lưu
            st.rerun() # Làm mới giao diện ngay lập tức
        else:
            # Fallback to append if update fails or logic dictates
            # Note: Chỉ append dòng mới nếu là RF (chính), SVR chỉ update
            if model_type == 'RF':
                prediction_data = {
                    'Date': pred['date'],
                    'Price': pred_price,
                    'Open': pred_price,
                    'High': results['RF']['upper'],
                    'Low': results['RF']['lower'],
                    'Vol': 0
                }
                if append_prediction_to_csv(target_path, prediction_data):
                    st.success("Đã thêm dòng dự đoán mới vào CSV!")
                load_and_process_data(target_path=target_path)
                st.rerun()
            else:
                st.error("Lưu dự đoán thất bại")
    else:
        st.info("Dự đoán này đã tồn tại trong tệp dữ liệu.")



def display_manual_input_form():
    """Hiển thị form nhập dữ liệu thực tế cho ngày tiếp theo"""
    df = st.session_state.df_features
    latest_date = df.iloc[-1]['Date']
    next_date = get_next_trading_date(latest_date)
    
    target_path = st.session_state.get('active_data_path', DISPLAY_DATA_PATH)
    target_filename = os.path.basename(target_path)
    
    st.markdown(f'<div class="section-header">CẬP NHẬT DỮ LIỆU THỰC TẾ: {next_date.strftime("%d/%m/%Y")}</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.caption(f"📁 Tệp đang cập nhật: **{target_filename}**")
        st.write("Vui lòng nhập thông tin thị trường chốt phiên để cập nhật hệ thống:")
        
        with st.form("manual_input_form", clear_on_submit=True):
            price = st.number_input("Giá Đóng (Close)", value=None, format="%.4f", placeholder="0.0000")
            vol = st.number_input("Khối lượng (Volume)", value=None, step=1000, placeholder="Nhập volume...")
            
            c_ohl1, c_ohl2 = st.columns(2)
            with c_ohl1:
                open_p = st.number_input("Mở (Open)", value=None, format="%.4f", placeholder="0.0000")
                high = st.number_input("Cao (High)", value=None, format="%.4f", placeholder="0.0000")
            with c_ohl2:
                low = st.number_input("Thấp (Low)", value=None, format="%.4f", placeholder="0.0000")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("XÁC NHẬN CẬP NHẬT", use_container_width=True, type="primary")
            if submit:
                if any(v is None for v in [price, vol, open_p, high, low]):
                    st.error("Vui lòng nhập đầy đủ tất cả các trường dữ liệu!")
                else:
                    handle_manual_input_submission(next_date, price, open_p, high, low, vol)
    
    # Hiển thị kết quả vừa dự đoán nếu có
    if 'last_manual_result' in st.session_state:
        st.success("Dữ liệu đã được cập nhật thành công!")
        st.markdown("#### Kết quả dự đoán cho dòng dữ liệu vừa nhập:")
        st.dataframe(st.session_state.last_manual_result, use_container_width=True, hide_index=True)


def handle_manual_input_submission(date, price, open_p, high, low, vol):
    """Xử lý lưu dữ liệu thực tế và TẤT CẢ các chỉ số kỹ thuật vào CSV"""
    target_path = st.session_state.get('active_data_path', DISPLAY_DATA_PATH)
    try:
        # 1. Load dữ liệu hiện tại chỉ lấy các cột gốc để tránh bị lặp cột features cũ
        df_raw = load_data(target_path)
        base_cols = ['Date', 'Price', 'Open', 'High', 'Low', 'Vol']
        df_base = df_raw[base_cols].copy()
        
        # 2. Thêm dòng mới vào base data
        new_row = pd.DataFrame([{
            'Date': date,
            'Price': price,
            'Open': open_p,
            'High': high,
            'Low': low,
            'Vol': vol
        }])
        df_base = pd.concat([df_base, new_row], ignore_index=True)
        
        # 3. Tính toán lại TOÀN BỘ features trên dữ liệu đã nối
        df_all_features = create_advanced_features(df_base)
        
        # Đảm bảo RF_Pred_Today được tính từ RF_Pred_Tomorrow của ngày trước đó (nếu có)
        if 'RF_Pred_Tomorrow' in df_raw.columns:
            # Copy cột dự báo cũ sang để không bị mất dữ liệu lịch sử
            df_all_features['RF_Pred_Tomorrow'] = df_raw['RF_Pred_Tomorrow']
            df_all_features.loc[df_all_features.index[-1], 'RF_Pred_Tomorrow'] = np.nan
        
        # 4. Thực hiện dự báo RF_Pred_Tomorrow cho dòng vừa thêm
        if st.session_state.model is not None and st.session_state.scaler is not None:
            feature_cols = get_feature_columns()
            # Xử lý NaN và Infinity cho features trước khi dự báo
            df_for_pred = df_all_features[feature_cols].copy().replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            latest_features = df_for_pred.iloc[-1:].values
            
            # Dự báo giá cho ngày tiếp theo
            pred_val = predict_next_day_layer1(st.session_state.model, st.session_state.scaler, latest_features)
            df_all_features.loc[df_all_features.index[-1], 'RF_Pred_Tomorrow'] = pred_val
            
        # 5. Cập nhật RF_Pred_Today (Lấy dự báo của ngày trước đó gán cho hôm nay)
        if 'RF_Pred_Tomorrow' in df_all_features.columns:
            df_all_features['RF_Pred_Today'] = df_all_features['RF_Pred_Tomorrow'].shift(1)
            
        # 6. Lưu dữ liệu thô (Chỉ Input và Date) vào CSV
        # User yêu cầu chỉ lưu input và date vào file đã chọn
        base_cols = ['Date', 'Price', 'Open', 'High', 'Low', 'Vol']
        # Đảm bảo các cột tồn tại trong df_all_features
        cols_to_save = [c for c in base_cols if c in df_all_features.columns]
        df_save = df_all_features[cols_to_save].copy()
        df_save['Date'] = df_save['Date'].dt.strftime('%Y-%m-%d')
        df_save.to_csv(target_path, index=False)
        
        # 7. Cập nhật giao diện
        st.session_state.df_features = df_all_features
        
        # Lưu dòng kết quả để hiển thị ngay dưới form
        result_display = df_all_features.tail(1).copy()
        result_display['Date'] = result_display['Date'].dt.strftime('%d/%m/%Y')
        for col in result_display.columns:
            if col != 'Date' and col != 'Vol':
                result_display[col] = result_display[col].apply(lambda x: f"${x:.4f}" if pd.notna(x) else "N/A")
        
        st.session_state.last_manual_result = result_display
        st.success(f"Đã cập nhật toàn bộ chỉ số và dự báo vào file CSV!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Lỗi khi xử lý dữ liệu: {e}")
        import traceback
        st.error(traceback.format_exc())


def display_layer2_content():
    """Display Layer 2 (Within-day prediction) content"""
    # --- HÀNG 1: NHẬP LIỆU & HUẤN LUYỆN LAYER 2 ---
    col_top_left, col_top_right = st.columns([1, 1.8])
    
    with col_top_left:
        st.markdown('<div class="section-header">1. NHẬP DỮ LIỆU</div>', unsafe_allow_html=True)
        with st.container(border=True):
            uploaded_file_l2 = st.file_uploader("CSV/Excel file (L2)", type=['csv', 'xlsx'], key="file_l2", label_visibility="collapsed")
            if st.button("Xử lý", use_container_width=True, type="primary", key="btn_process_l2", disabled=(uploaded_file_l2 is None)):
                load_and_process_data(uploaded_file_l2)
                
    with col_top_right:
        st.markdown('<div class="section-header">2. HUẤN LUYỆN LAYER 2</div>', unsafe_allow_html=True)
        with st.container(border=True):
            ridge_status = "Đã train" if st.session_state.get('l2_ridge_model_trained', False) else "Chưa train"
            st.caption(f"Trạng thái Layer 2: Ridge Stacking [{ridge_status}]")
            
            col_l2_btn1, col_l2_btn2 = st.columns(2)
            with col_l2_btn1:
                if st.button("Train Layer 2", use_container_width=True, type="primary", key="btn_train_l2", disabled=not st.session_state.model_trained):
                    train_layer2_logic()
            with col_l2_btn2:
                if st.button("Load L2 Models", use_container_width=True, key="btn_load_l2"):
                    load_l2_model()

    st.markdown("---")
    
    if st.session_state.df_features is None:
        st.info("Vui lòng tải dữ liệu ở Mục 1 hoặc Sidebar để bắt đầu.")
        return

    # Prediction Section
    st.subheader("Dự đoán giá chốt phiên trực tuyến")
    
    # 1. Get Base Predictions from Layer 1 for the TARGET day
    latest_row = st.session_state.df_features.iloc[-1]
    last_date = latest_row['Date']
    target_date = get_next_trading_date(last_date)
    
    # Check if we have fresh predictions
    l1_rf_target = None
    l1_svr_target = None
    
    # Try getting from row first
    if 'RF_Pred_Tomorrow' in latest_row and pd.notna(latest_row['RF_Pred_Tomorrow']):
        l1_rf_target = latest_row['RF_Pred_Tomorrow']
    if 'SVR_Pred_Tomorrow' in latest_row and pd.notna(latest_row['SVR_Pred_Tomorrow']):
        l1_svr_target = latest_row['SVR_Pred_Tomorrow']
        
    # Overwrite/fill from session if user just clicked predict
    if 'prediction' in st.session_state:
        if st.session_state.prediction['date'].date() == target_date.date():
            results = st.session_state.prediction['results']
            if 'RF' in results: l1_rf_target = results['RF']['price']
            if 'SVR' in results: l1_svr_target = results['SVR']['price']

    if l1_rf_target is None or l1_svr_target is None:
        st.warning(f"Chưa có đủ dự đoán Layer 1 (RF & SVR) cho ngày {target_date.strftime('%d/%m/%Y')}. Vui lòng qua Tab Layer 1 huấn luyện và dự đoán cả 2 mô hình trước.")
        return

    # --- Layer 1 Summary Dashboard ---
    st.markdown(f"""
    <div style="background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 12px; padding: 15px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <div style="width: 4px; height: 18px; background-color: #007BFF; margin-right: 10px; border-radius: 2px;"></div>
            <div style="font-weight: 700; color: #1F2937; font-size: 0.9rem; text-transform: uppercase;">Thông tin từ Layer 1 (Tính năng đầu vào)</div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
            <div style="background: white; padding: 10px; border-radius: 8px; border: 1px solid #F3F4F6; text-align: center;">
                <div style="font-size: 0.7rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;">Ngày mục tiêu</div>
                <div style="font-size: 1rem; font-weight: 600; color: #111827;">{target_date.strftime('%d/%m/%Y')}</div>
            </div>
            <div style="background: white; padding: 10px; border-radius: 8px; border: 1px solid #F3F4F6; text-align: center;">
                <div style="font-size: 0.7rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;">RF_Pred_Today</div>
                <div style="font-size: 1rem; font-weight: 600; color: #00b894;">${format_number(l1_rf_target)}</div>
            </div>
            <div style="background: white; padding: 10px; border-radius: 8px; border: 1px solid #F3F4F6; text-align: center;">
                <div style="font-size: 0.7rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;">SVR_Pred_Today</div>
                <div style="font-size: 1rem; font-weight: 600; color: #00b894;">${format_number(l1_svr_target)}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. User Input
    with st.form("layer2_form"):
        st.write(f"Nhập dữ liệu thị trường thực tế của ngày {target_date.strftime('%d/%m/%Y')}:")
        col1, col2 = st.columns(2)
        with col1:
            open_price = st.number_input("Giá mở cửa (Open)", value=None, placeholder="Nhập giá mở cửa...", format="%.4f")
            high_price = st.number_input("Giá cao nhất (High)", value=None, placeholder="Nhập giá cao nhất...", format="%.4f")
        with col2:
            low_price = st.number_input("Giá thấp nhất (Low)", value=None, placeholder="Nhập giá thấp nhất...", format="%.4f")
            current_vol = st.number_input("Khối lượng dự kiến (Volume)", value=None, placeholder="Nhập khối lượng dự kiến...", format="%.0f")
        
        submit = st.form_submit_button("Tính toán giá chốt phiên (Layer 2)")

    if submit:
        if any(v is None for v in [open_price, high_price, low_price, current_vol]):
            st.error("Vui lòng nhập đầy đủ giá Open, High, Low và Volume của ngày hôm nay!")
        elif not st.session_state.l2_ridge_model_trained:
            st.error("Vui lòng train Layer 2 tại Tab này trước!")
        else:
            try:
                # Combine Features: [Open, High, Low, Vol, RF_Pred_Today, SVR_Pred_Today]
                l2_input = np.array([[open_price, high_price, low_price, current_vol, l1_rf_target, l1_svr_target]])
                
                pred_close = predict_layer2(st.session_state.l2_ridge_model, st.session_state.l2_ridge_scaler, l2_input)
                
                st.markdown(f"""
                <div class="prediction-card" style="max-width: 500px; margin: 0 auto;">
                    <div class="pred-title">L2: Ridge Stacking Result</div>
                    <div class="pred-sub">Giá chốt phiên hội tụ (Dựa trên O-H-L-V & L1 Hybrid)</div>
                    <div class="pred-price" style="color: #007BFF; font-size: 2.5rem;">${format_number(pred_close)}</div>
                    <div class="confidence-box" style="background-color: #ebf5ff;">
                        <span style="font-size: 0.9rem; color: #007BFF; font-weight: 600;">Kết hợp tối ưu từ Random Forest và Support Vector Regressor</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Lỗi dự đoán L2: {e}")

def train_layer2_logic():
    """Train Layer 2 (Single Ridge Stacking)"""
    with st.spinner("Đang chuẩn bị dữ liệu và huấn luyện Layer 2..."):
        try:
            if st.session_state.df_features is None:
                st.error("Vui lòng tải dữ liệu trước!")
                return

            # Check L1 models
            if not st.session_state.model_trained or not st.session_state.svr_model_trained:
                st.error("Cần huấn luyện cả RandomForest và SVR (Layer 1) trước khi train Layer 2!")
                return

            df = st.session_state.df_features.copy()
            feature_cols = st.session_state.feature_cols
            
            # --- Generate L1 Projections for History ---
            df_to_pred = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            X_all = df_to_pred.values
            
            # 1. RF Predictions
            rf_scaler = st.session_state.scaler
            rf_model = st.session_state.model
            X_scaled_rf = rf_scaler.transform(X_all)
            df['RF_Pred_Tomorrow'] = rf_model.predict(X_scaled_rf)
            df['RF_Pred_Today'] = df['RF_Pred_Tomorrow'].shift(1)
            
            # 2. SVR Predictions
            svr_scaler = st.session_state.svr_scaler
            svr_model = st.session_state.svr_model
            X_scaled_svr = svr_scaler.transform(X_all)
            df['SVR_Pred_Tomorrow'] = svr_model.predict(X_scaled_svr)
            df['SVR_Pred_Today'] = df['SVR_Pred_Tomorrow'].shift(1)
            
            # Update session state with L1 history
            st.session_state.df_features = df
            
            # --- Prepare L2 Stacking Data ---
            # Inputs: Open, High, Low, Vol, RF_Pred_Today, SVR_Pred_Today
            l2_features = ['Open', 'High', 'Low', 'Vol', 'RF_Pred_Today', 'SVR_Pred_Today']
            target = 'Price'
            
            # Drop NaNs from shifting
            df_l2 = df.dropna(subset=l2_features + [target])
            
            if len(df_l2) < 50:
                st.error("Không đủ dữ liệu sạch để train Layer 2 (thiếu lịch sử dự báo).")
                return
                
            X = df_l2[l2_features]
            y = df_l2[target]
            
            # Split (80/20)
            split_idx = int(len(X) * 0.8)
            X_train, y_train = X[:split_idx], y[:split_idx]
            
            # Train Single Ridge Model
            ridge_model, ridge_scaler = train_layer2_model(X_train, y_train)
            
            # Save
            save_model(ridge_model, L2_RIDGE_MODEL_PATH)
            save_model(ridge_scaler, L2_RIDGE_SCALER_PATH)
            
            # Update Session State
            st.session_state.l2_ridge_model = ridge_model
            st.session_state.l2_ridge_scaler = ridge_scaler
            st.session_state.l2_ridge_model_trained = True
            
            st.toast("Đã train Layer 2 (Single Ridge Stacking) thành công!")
            
        except Exception as e:
            st.error(f"Lỗi khi train L2: {e}")
            import traceback
            st.error(traceback.format_exc())

def load_l2_model():
    """Load Layer 2 model (Single Ridge Stacking)"""
    with st.spinner("Đang tải mô hình Layer 2..."):
        try:
            ridge_model = load_model(L2_RIDGE_MODEL_PATH)
            ridge_scaler = load_model(L2_RIDGE_SCALER_PATH)
            if ridge_model and ridge_scaler:
                st.session_state.l2_ridge_model = ridge_model
                st.session_state.l2_ridge_scaler = ridge_scaler
                st.session_state.l2_ridge_model_trained = True
                st.toast("Đã tải mô hình Layer 2 Ridge Stacking thành công!")
            else:
                st.warning("Không tìm thấy mô hình Layer 2 đã lưu.")
        except Exception as e:
            st.error(f"Lỗi khi load L2 model: {e}")


def display_keras_lstm_impl():
    """Hiển thị nội dung cho Layer 3 (LSTM) - Keras Implementation"""
    st.subheader("Layer 3: Dự báo chuỗi thời gian bằng Deep Learning (LSTM)")
    
    # Cho phép chọn file CSV riêng cho Layer 3
    st.markdown("### Chọn dữ liệu cho Layer 3")
    l3_file = st.file_uploader("Tải lên file CSV (Ví dụ: ETHUSDT.csv)", type=['csv'])
    
    if l3_file is not None:
        if st.button("Sử dụng file đã tải lên cho Layer 3"):
            load_l3_custom_data(l3_file)
            
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Xử lý dữ liệu L3", use_container_width=True):
            prepare_l3_features()
    with col2:
        if st.button("Train mô hình L3", use_container_width=True):
            train_l3_model()
    with col3:
        if st.button("Load L3 model", use_container_width=True):
            load_l3_model()
    with col4:
        if st.button("Dự báo LSTM (7 ngày)", use_container_width=True, disabled=not st.session_state.l3_model_trained):
            make_l3_prediction()

    st.markdown("---")
    
    if 'l3_prediction' in st.session_state:
        display_l3_prediction_results()
    else:
        st.info("Sử dụng LSTM để dự báo biến động giá trong 7 ngày tới dựa trên 30 ngày lịch sử.")
        
    if st.checkbox("Hiển thị kiến trúc mô hình LSTM"):
        st.code("""
        Model: Sequential
        Layer 1: LSTM (64 units, return_sequences=True)
        Layer 2: Dropout (0.2)
        Layer 3: LSTM (32 units)
        Layer 4: Dropout (0.2)
        Layer 5: Dense (7 units - forecast window)
        Optimizer: Adam (lr=0.001)
        Loss: MSE
        """)

def load_l3_custom_data(uploaded_file):
    """Load dữ liệu từ file upload cho Layer 3"""
    try:
        df = pd.read_csv(uploaded_file)
        # Standardize columns
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Mapping common column names if needed
        col_map = {
            'Close': 'Price',
            'Volume': 'Vol'
        }
        df = df.rename(columns=col_map)
        
        # Basic requirements
        required = ['Date', 'Price', 'Open', 'High', 'Low', 'Vol']
        if all(col in df.columns for col in required):
            st.session_state.df_l3_raw = df[required]
            st.toast(f"Đã tải dữ liệu cho Layer 3!", icon="📥")
            st.dataframe(df.head())
        else:
            st.error(f"File CSV thiếu các cột bắt buộc: {required}")
    except Exception as e:
        st.error(f"Lỗi khi xử lý file: {e}")

def prepare_l3_features():
    """Tạo features cho LSTM"""
    # Ưu tiên sử dụng dữ liệu riêng của L3 nếu có, nếu không lấy từ main df
    if 'df_l3_raw' in st.session_state:
        df = st.session_state.df_l3_raw.copy()
    elif st.session_state.df_features is not None:
        df = st.session_state.df_features[['Date', 'Price', 'Open', 'High', 'Low', 'Vol']]
    else:
        st.error("Vui lòng tải dữ liệu hoặc chọn file CSV!")
        return
        
    with st.spinner("Đang tính toán technical indicators cho LSTM..."):
        try:
            df_l3 = create_lstm_features(df)
            st.session_state.df_l3 = df_l3
            st.toast("✓ Đã chuẩn bị features cho LSTM!", icon="⚙️")
            st.dataframe(df_l3.tail(5))
        except Exception as e:
            st.error(f"Lỗi: {e}")

def train_l3_model():
    """Train LSTM model"""
    if 'df_l3' not in st.session_state:
        prepare_l3_features()
        
    df = st.session_state.df_l3.copy()
    feature_cols = ['Open', 'High', 'Low', 'Price', 'Vol', 'VVR', 'VWAP', 
                    'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'Lag_7', 
                    'Price_Change', 'Volatility', 'MA5', 'MA10']
    
    df_clean = df.dropna(subset=feature_cols)
    
    with st.spinner("Đang huấn luyện mô hình LSTM (Deep Learning)..."):
        try:
            X, y, scaler, target_scaler = prepare_lstm_data(df_clean, feature_cols)
            
            # Split
            n = len(X)
            split = int(n * 0.9)
            X_train, y_train = X[:split], y[:split]
            
            model = train_lstm_model(X_train, y_train)
            
            # Save
            model.save(L3_MODEL_PATH)
            save_model(scaler, L3_SCALER_PATH)
            save_model(target_scaler, L3_TARGET_SCALER_PATH)
            
            st.session_state.l3_model = model
            st.session_state.l3_scaler = scaler
            st.session_state.l3_target_scaler = target_scaler
            st.session_state.l3_model_trained = True
            st.session_state.l3_feature_cols = feature_cols
            
            st.toast("✓ Huấn luyện Layer 3 thành công!", icon="🚀")
        except Exception as e:
            st.error(f"Lỗi khi train LSTM: {e}")

def load_l3_model():
    """Load pre-trained LSTM model"""
    with st.spinner("Đang load mô hình LSTM..."):
        try:
            from tensorflow.keras.models import load_model as load_keras_model
            if os.path.exists(L3_MODEL_PATH):
                st.session_state.l3_model = load_keras_model(L3_MODEL_PATH)
                st.session_state.l3_scaler = load_model(L3_SCALER_PATH)
                st.session_state.l3_target_scaler = load_model(L3_TARGET_SCALER_PATH)
                st.session_state.l3_model_trained = True
                st.session_state.l3_feature_cols = ['Open', 'High', 'Low', 'Price', 'Vol', 'VVR', 'VWAP', 
                                                'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'Lag_7', 
                                                'Price_Change', 'Volatility', 'MA5', 'MA10']
                st.toast("✓ Đã load Layer 3 thành công!", icon="💾")
            else:
                st.warning("Không tìm thấy tệp mô hình Layer 3.")
        except Exception as e:
            st.error(f"Lỗi: {e}")

def make_l3_prediction():
    """Dự báo 7 ngày tới bằng LSTM"""
    if not st.session_state.l3_model_trained:
        st.error("Mô hình chưa được huấn luyện hoặc load!")
        return
        
    try:
        df = st.session_state.df_l3.copy()
        feature_cols = st.session_state.l3_feature_cols
        
        # Lấy 30 ngày cuối để làm sequence đầu vào
        last_30_days = df.dropna(subset=feature_cols).tail(30)
        scaled_sequence = st.session_state.l3_scaler.transform(last_30_days[feature_cols])
        
        # Predict
        pred_scaled = predict_lstm(st.session_state.l3_model, scaled_sequence)
        
        # Inverse transform
        pred_prices = st.session_state.l3_target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        
        # Dates
        last_date = df['Date'].max()
        pred_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        
        pred_df = pd.DataFrame({
            'Date': pred_dates,
            'Predicted_Price': pred_prices
        })
        
        st.session_state.l3_prediction = pred_df
        st.toast("✓ Đã hoàn thành dự báo LSTM!", icon="🔮")
        
    except Exception as e:
        st.error(f"Lỗi khi dự báo LSTM: {e}")

def display_l3_prediction_results():
    """Hiển thị kết quả dự báo của LSTM"""
    pred_df = st.session_state.l3_prediction
    
    # Xác định nguồn dữ liệu để hiển thị lịch sử
    if 'df_l3' in st.session_state:
        df_source = st.session_state.df_l3
    elif 'df_l3_raw' in st.session_state:
        df_source = st.session_state.df_l3_raw
    elif st.session_state.df_features is not None:
        df_source = st.session_state.df_features
    else:
        st.warning("Không tìm thấy dữ liệu lịch sử để hiển thị biểu đồ.")
        return

    df_hist = df_source.tail(20)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Bảng dự đoán 7 ngày")
        fmt_df = pred_df.copy()
        fmt_df['Date'] = fmt_df['Date'].dt.strftime('%d/%m/%Y')
        fmt_df['Predicted_Price'] = fmt_df['Predicted_Price'].map('${:,.4f}'.format)
        st.table(fmt_df)
        
    with col2:
        st.subheader("Biểu đồ dự báo Deep Learning")
        fig = go.Figure()
        
        # Lịch sử
        fig.add_trace(go.Scatter(
            x=df_hist['Date'], y=df_hist['Price'],
            mode='lines+markers', name='Lịch sử (20 ngày)',
            line=dict(color='white')
        ))
        
        # Dự báo
        # Nối điểm cuối lịch sử với điểm đầu dự báo
        connect_date = [df_hist['Date'].iloc[-1]] + pred_df['Date'].tolist()
        connect_price = [df_hist['Price'].iloc[-1]] + pred_df['Predicted_Price'].tolist()
        
        fig.add_trace(go.Scatter(
            x=connect_date, y=connect_price,
            mode='lines+markers', name='LSTM Forecast',
            line=dict(color='#00D9FF', dash='dash', width=3)
        ))
        
        fig.update_layout(
            template='plotly_dark',
            margin=dict(l=10, r=10, t=30, b=10),
            height=450,
            xaxis_title="Ngày",
            yaxis_title="Giá XRP ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)


# Layer 3 Enhanced Content - Append this to app.py before if __name__ == "__main__"

# =============================================================================
# LAYER 3: ENHANCED WITH REGIME LSTM \u0026 ML ENSEMBLE
# =============================================================================

def display_layer3_content():
    """
    Enhanced Layer 3 với 3 phương pháp học chuỗi:
    - Tab 1: LSTM (Keras) - Hiện tại
    - Tab 2: Regime LSTM - Custom implementation với regime detection
    - Tab 3: ML Ensemble - RandomForest + GradientBoosting + Ridge
    """
    
    # st.markdown('\u003cdiv class="section-header"\u003eLỚP 3: HỌC SÂU \u0026 SEQUENCE LEARNING\u003c/div\u003e', unsafe_allow_html=True)
    
    # Removed dependency check to allow custom file upload in sub-tabs
    
    # Create 3 tabs
    # Create 2 tabs (Hidden Keras LSTM)
    tab_regime, tab_ensemble = st.tabs([
        "🟠 Regime LSTM", 
        "🟢 ML Ensemble"
    ])
    
    # ===================
    # TAB 1: Keras LSTM (Hidden as requested)
    # ===================
    # with tab_keras:
    #     display_keras_lstm_tab()
    
    # ===================
    # TAB 2: Regime LSTM (NEW)
    # ===================
    with tab_regime:
        display_regime_lstm_tab()
    
    # ===================
    # TAB 3: ML Ensemble (NEW)
    # ===================
    with tab_ensemble:
        display_ml_ensemble_tab()


def display_keras_lstm_tab():
    """Tab cũ - LSTM Keras hiện tại"""
    # Gọi hàm thực thi logic cũ
    display_keras_lstm_impl()


def display_regime_lstm_tab():
    """Tab mới - Regime-aware LSTM theo đúng luồng LSTMCustom.ipynb"""
    st.subheader("Regime LSTM - Nhận diện chế độ volatility")
    
    # --- HÀNG 1: NHẬP LIỆU & HUẤN LUYỆN ---
    col_top_left, col_top_right = st.columns([1, 1.8])
    
    with col_top_left:
        st.markdown('<div class="section-header">1. NHẬP DỮ LIỆU</div>', unsafe_allow_html=True)
        with st.container(border=True):
            regime_file = st.file_uploader("Tải lên file CSV", type=['csv'], key="regime_uploader", label_visibility="collapsed")
            
            if regime_file is not None:
                try:
                    df_r = pd.read_csv(regime_file)
                    if 'Date' in df_r.columns:
                        df_r['Date'] = pd.to_datetime(df_r['Date'])
                        df_r = df_r.sort_values('Date').reset_index(drop=True)
                    
                    col_map = {'Close': 'Price', 'Volume': 'Vol', 'high': 'High', 'low': 'Low', 'open': 'Open'}
                    for old_col, new_col in col_map.items():
                        if old_col in df_r.columns and new_col not in df_r.columns:
                            df_r.rename(columns={old_col: new_col}, inplace=True)
                    
                    required_cols = ['Price', 'Vol', 'High', 'Low']
                    missing = [c for c in required_cols if c not in df_r.columns]
                    if missing:
                        st.error(f"Thiếu các cột bắt buộc: {', '.join(missing)}")
                    else:
                        st.session_state.df_regime_raw = df_r
                        st.toast(f"✓ Đã tải {len(df_r)} dòng dữ liệu gốc.")
                except Exception as e:
                    st.error(f"Lỗi tải file: {e}")
            
            col_reg_btn1, col_reg_btn2 = st.columns(2)
            with col_reg_btn1:
                if st.button("Xử lý Features", type="primary", use_container_width=True, key="btn_reg_process", disabled='df_regime_raw' not in st.session_state):
                    with st.spinner("Đang tính toán đặc trưng..."):
                        df_features = create_regime_features(st.session_state.df_regime_raw)
                        st.session_state.df_regime = df_features
                        st.toast("✓ Đã tạo xong 18 features!")
            with col_reg_btn2:
                if st.button("Xem Data", use_container_width=True, key="btn_reg_view", disabled='df_regime' not in st.session_state):
                    st.session_state.show_regime_data = not st.session_state.get('show_regime_data', False)

    with col_top_right:
        st.markdown('<div class="section-header">2. HUẤN LUYỆN</div>', unsafe_allow_html=True)
        with st.container(border=True):
            status_text = "Đã train" if st.session_state.get('regime_lstm_trained', False) else "Chưa train"
            st.caption(f"Trạng thái mô hình: {status_text}")
            
            col_train_l, col_train_r = st.columns(2)
            with col_train_l:
                with st.expander("Tham số training", expanded=False):
                    reg_epochs = st.slider("Epochs", 20, 150, 70, 10, key="reg_ep")
                    reg_lr = st.number_input("Learning rate", 0.0001, 0.01, 0.001, format="%.4f", key="reg_lr")
                if st.button("Bắt đầu Train", type="primary", use_container_width=True, key="btn_reg_train", disabled='df_regime' not in st.session_state):
                    train_regime_lstm_model(reg_epochs, reg_lr)
            
            with col_train_r:
                st.write("") # Spacer
                if st.button("Tải model từ disk", use_container_width=True, key="btn_reg_load"):
                    load_regime_lstm_model_from_disk()

    if st.session_state.get('show_regime_data', False) and 'df_regime' in st.session_state:
        st.dataframe(st.session_state.df_regime.tail(100), use_container_width=True, height=250)

    st.markdown("---")
    
    # --- PHẦN KẾT QUẢ DỰ BÁO ---
    if st.session_state.regime_lstm_trained:
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.markdown('<div class="section-header">3. DỰ BÁO T+7</div>', unsafe_allow_html=True)
            if st.button("Thực hiện Dự báo", type="primary", use_container_width=True, key="btn_reg_pred"):
                make_regime_lstm_prediction()
        
        if st.session_state.regime_lstm_metrics is not None:
            display_regime_lstm_results()
    else:
        st.info("Vui lòng xử lý dữ liệu và huấn luyện mô hình để dự báo.")


def display_ml_ensemble_tab():
    st.subheader("ML Ensemble - Stacking Approach")
    
    # --- HÀNG 1: NHẬP LIỆU & HUẤN LUYỆN ---
    col_top_left, col_top_right = st.columns([1, 1.8])
    
    with col_top_left:
        st.markdown('<div class="section-header">1. NHẬP DỮ LIỆU</div>', unsafe_allow_html=True)
        with st.container(border=True):
            ensemble_file = st.file_uploader("Tải lên file CSV", type=['csv'], key="ensemble_uploader", label_visibility="collapsed")
            
            if ensemble_file is not None:
                try:
                    df_e = pd.read_csv(ensemble_file)
                    if 'Date' in df_e.columns:
                        df_e['Date'] = pd.to_datetime(df_e['Date'])
                        df_e = df_e.sort_values('Date').reset_index(drop=True)
                    if 'Close' in df_e.columns and 'Price' not in df_e.columns:
                        df_e.rename(columns={'Close': 'Price'}, inplace=True)
                    if 'Volume' in df_e.columns and 'Vol' not in df_e.columns:
                        df_e.rename(columns={'Volume': 'Vol'}, inplace=True)
                    st.session_state.df_ensemble_raw = df_e
                    st.toast("✓ Đã nhận file CSV cho Ensemble.")
                except Exception as e:
                    st.error(f"Lỗi tải file: {e}")
            
            col_ens_btn1, col_ens_btn2 = st.columns(2)
            with col_ens_btn1:
                # Resolve data source (Independent only)
                data_ready = 'df_ensemble_raw' in st.session_state

                if st.button("Xử lý Features", type="primary", use_container_width=True, key="btn_ens_process", disabled=not data_ready):
                    with st.spinner("Đang tạo features..."):
                        source_df = st.session_state.df_ensemble_raw
                        df_processed = create_ml_features(source_df)
                        st.session_state.df_ensemble = df_processed
                        st.toast("✓ Đã chuẩn bị features cho Ensemble!")
            with col_ens_btn2:
                if st.button("Xem Data", use_container_width=True, key="btn_ens_view", disabled='df_ensemble' not in st.session_state):
                    st.session_state.show_ensemble_data = not st.session_state.get('show_ensemble_data', False)

    with col_top_right:
        st.markdown('<div class="section-header">2. HUẤN LUYỆN</div>', unsafe_allow_html=True)
        with st.container(border=True):
            status_text = "Đã train" if st.session_state.get('ml_ensemble_trained', False) else "Chưa train"
            st.caption(f"Trạng thái mô hình: {status_text}")
            
            col_e_train1, col_e_train2 = st.columns(2)
            with col_e_train1:
                if st.button("Train Ensemble", type="primary", use_container_width=True, key="btn_ens_train", disabled='df_ensemble' not in st.session_state):
                    train_ml_ensemble_model()
            with col_e_train2:
                if st.button("Tải model từ disk", use_container_width=True, key="btn_ens_load"):
                    load_ml_ensemble_model_from_disk()

    if st.session_state.get('show_ensemble_data', False) and 'df_ensemble' in st.session_state:
        st.dataframe(st.session_state.df_ensemble.tail(100), use_container_width=True, height=250)

    st.markdown("---")
    
    # --- PHẦN KẾT QUẢ DỰ BÁO ---
    if st.session_state.ml_ensemble_trained:
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.markdown('<div class="section-header">3. DỰ BÁO T+7</div>', unsafe_allow_html=True)
            if st.button("Thực hiện Dự báo", type="primary", use_container_width=True, key="btn_ens_pred"):
                make_ml_ensemble_prediction()
        
        if st.session_state.ml_ensemble_metrics is not None:
             display_ml_ensemble_results()
    else:
        st.info("Vui lòng tải file CSV riêng trong Tab này để huấn luyện mô hình Ensemble.")
    
    st.divider()
    
    # Display results
    if st.session_state.ml_ensemble_metrics is not None:
        display_ml_ensemble_results()


# ======================
# TRAINING FUNCTIONS
# ======================

def train_regime_lstm_model(epochs=60, lr=0.001):
    """Train Regime LSTM model"""
    with st.spinner(f"Đang training Regime LSTM ({epochs} epochs)..."):
        try:
            # Lấy dữ liệu riêng của Regime Tab
            if 'df_regime' in st.session_state and st.session_state.df_regime is not None:
                df = st.session_state.df_regime.copy()
            else:
                st.error("Không tìm thấy dữ liệu cho Regime LSTM! Vui lòng tải file CSV ở Bước 1 (trong Tab này).")
                return
            
            # Ensure required columns
            if 'Price' not in df.columns:
                st.error("Dữ liệu thiếu cột 'Price'!")
                return
            
            # Đảm bảo dữ liệu được sắp xếp theo thời gian tăng dần
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
            
            # Validate that features have been created (Regime specific)
            if 'vol_z' not in df.columns:
                st.warning("⚠️ Dữ liệu chưa có regime features. Đang tự động xử lý...")
                from utils.regime_lstm import create_regime_features
                df = create_regime_features(df)
                st.success("✓ Đã xử lý regime features tự động")
            
            # Train model
            model, scalers, metrics = train_regime_lstm(
                df, 
                epochs=epochs, 
                lr=lr, 
                lookback=30, 
                horizon=7,
                test_size=0.2,
                verbose=True
            )
            
            # Save to session
            st.session_state.regime_lstm_model = model
            st.session_state.regime_lstm_scalers = scalers
            st.session_state.regime_lstm_metrics = metrics
            st.session_state.regime_lstm_trained = True
            
            # Save to disk
            save_model(model, REGIME_LSTM_MODEL_PATH)
            save_model(scalers, REGIME_LSTM_SCALERS_PATH)
            
            st.toast(f"✓ Training hoàn tất! MAE: {metrics['mae']:.4f}")
            
        except Exception as e:
            st.error(f"Lỗi khi training: {e}")
            import traceback
            st.error(traceback.format_exc())


def train_ml_ensemble_model():
    """Train ML Ensemble model"""
    with st.spinner("Đang training ML Ensemble..."):
        try:
            # Lấy dữ liệu riêng của Ensemble Tab
            if 'df_ensemble' in st.session_state and st.session_state.df_ensemble is not None:
                df = st.session_state.df_ensemble.copy()
            else:
                st.error("Không tìm thấy dữ liệu cho ML Ensemble! Vui lòng tải file CSV ở Bước 1 (trong Tab này).")
                return

            df.to_csv("df_features_export.csv", index=False)
            
            # Đảm bảo dữ liệu được sắp xếp theo thời gian tăng dần
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
            
            # Validate that features have been created
            if 'RSI_14' not in df.columns:
                st.warning("Dữ liệu chưa được xử lý features. Đang tự động xử lý...")
                from utils.ml_ensemble import create_ml_features
                df = create_ml_features(df)
                st.success("✓ Đã xử lý features tự động")
            
            # Train model
            ensemble, metrics = train_ml_ensemble(
                df,
                lookback=30,
                horizon=7,
                train_ratio=0.8,
                verbose=False
            )
            
            # Save to session
            st.session_state.ml_ensemble_model = ensemble
            st.session_state.ml_ensemble_metrics = metrics
            st.session_state.ml_ensemble_trained = True
            
            # Save to disk
            save_model(ensemble, ML_ENSEMBLE_MODEL_PATH)
            
            mae = metrics['test']['mae']
            st.toast(f"✓ Training hoàn tất! MAE: {mae:.4f}", icon="🚀")
            
        except Exception as e:
            st.error(f"Lỗi khi training: {e}")
            st.error(traceback.format_exc())


def load_regime_lstm_model_from_disk():
    """Load Regime LSTM model from disk"""
    with st.spinner("Đang tải model Regime LSTM..."):
        try:
            model = load_model(REGIME_LSTM_MODEL_PATH)
            scalers = load_model(REGIME_LSTM_SCALERS_PATH)
            if model is not None and scalers is not None:
                st.session_state.regime_lstm_model = model
                st.session_state.regime_lstm_scalers = scalers
                st.session_state.regime_lstm_trained = True
                st.toast("Đã tải model Regime LSTM từ disk!")
            else:
                st.warning("Không tìm thấy file model để tải.")
        except Exception as e:
            st.error(f"Lỗi khi tải model: {e}")


def load_ml_ensemble_model_from_disk():
    """Load ML Ensemble model from disk"""
    with st.spinner("Đang tải model ML Ensemble..."):
        try:
            ensemble = load_model(ML_ENSEMBLE_MODEL_PATH)
            if ensemble is not None:
                st.session_state.ml_ensemble_model = ensemble
                st.session_state.ml_ensemble_trained = True
                st.toast("Đã tải model ML Ensemble từ disk!")
            else:
                st.warning("Không tìm thấy file model để tải.")
        except Exception as e:
            st.error(f"Lỗi khi tải model: {e}")


# ======================
# PREDICTION FUNCTIONS
# ======================

def make_regime_lstm_prediction():
    """Make prediction using Regime LSTM"""
    with st.spinner("Đang dự báo với Regime LSTM..."):
        try:
            # Ưu tiên dữ liệu riêng
            if 'df_regime' in st.session_state and st.session_state.df_regime is not None:
                df = st.session_state.df_regime.copy()
            else:
                df = st.session_state.df_features.copy()
                
            # Đảm bảo sắp xếp thời gian
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
            
            # Get latest 30+ rows for lookback
            df_latest = df.tail(35)  # Extra rows for feature engineering
            
            # Predict
            pred_price = predict_regime_lstm(
                st.session_state.regime_lstm_model,
                st.session_state.regime_lstm_scalers,
                df_latest,
                lookback=30
            )
            
            # Calculate predicted date (T+7)
            last_date = df.iloc[-1]['Date']
            pred_date = last_date + timedelta(days=7)
            
            # Store result
            if 'regime_lstm_prediction' not in st.session_state:
                st.session_state.regime_lstm_prediction = {}
            
            st.session_state.regime_lstm_prediction = {
                'date': pred_date,
                'price': pred_price,
                'current_price': df.iloc[-1]['Price']
            }
            
            st.success(f"Dự báo cho {pred_date.strftime('%d/%m/%Y')}: **${pred_price:.4f}**")
            
        except Exception as e:
            st.error(f"Lỗi khi dự báo: {e}")
            import traceback
            st.error(traceback.format_exc())


def make_ml_ensemble_prediction():
    """Make prediction using ML Ensemble"""
    with st.spinner("Đang dự báo với ML Ensemble..."):
        try:
            # Ưu tiên dữ liệu riêng
            if 'df_ensemble' in st.session_state and st.session_state.df_ensemble is not None:
                df = st.session_state.df_ensemble.copy()
            else:
                df = st.session_state.df_features.copy()
            
            # Đảm bảo sắp xếp thời gian
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
            
            # Get latest 30+ rows
            df_latest = df.tail(60)  # Extra for MA calculations
            
            # Predict
            pred_price = st.session_state.ml_ensemble_model.predict_next(df_latest)
            
            # Calculate predicted date (T+7)
            last_date = df.iloc[-1]['Date']
            pred_date = last_date + timedelta(days=7)
            
            # Store result
            if 'ml_ensemble_prediction' not in st.session_state:
                st.session_state.ml_ensemble_prediction = {}
            
            st.session_state.ml_ensemble_prediction = {
                'date': pred_date,
                'price': pred_price,
                'current_price': df.iloc[-1].get('Close', df.iloc[-1].get('Price', 0))
            }
            
            st.success(f"Dự báo cho {pred_date.strftime('%d/%m/%Y')}: **${pred_price:.4f}**")
            
        except Exception as e:
            st.error(f"Lỗi khi dự báo: {e}")
            import traceback
            st.error(traceback.format_exc())


# ======================
# RESULTS DISPLAY
# ======================

def display_regime_lstm_results():
    """Display Regime LSTM training \u0026 prediction results"""
    st.subheader("Kết quả Regime LSTM")
    
    metrics = st.session_state.regime_lstm_metrics
    
    # Metrics
    col1, col2 = st.columns(2)
    col1.metric("MAE (Test)", f"{metrics['mae']:.4f}")
    col2.metric("MSE (Test)", f"{metrics['mse']:.6f}")
    
    # Training loss curve
    if 'train_losses' in metrics:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=metrics['train_losses'],
            mode='lines',
            name='Training Loss',
            line=dict(color='#FF6B6B')
        ))
        fig.update_layout(
            title="Training Loss Curve",
            xaxis_title="Epoch",
            yaxis_title="Loss (MSE)",
            template='plotly_white',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Actual vs Predicted plot (from notebook logic)
    if 'y_true' in metrics and 'y_pred' in metrics:
        import plotly.graph_objects as go
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            y=metrics['y_true'],
            mode='lines',
            name='Actual (Test)',
            line=dict(color='gray', width=1)
        ))
        fig_pred.add_trace(go.Scatter(
            y=metrics['y_pred'],
            mode='lines',
            name='Predicted (Test)',
            line=dict(color='orange', width=2)
        ))
        fig_pred.update_layout(
            title="Actual vs Predicted (Test Set)",
            xaxis_title="Time Index",
            yaxis_title="Price",
            template='plotly_white',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pred, use_container_width=True)
    
    # Prediction display
    if 'regime_lstm_prediction' in st.session_state:
        pred = st.session_state.regime_lstm_prediction
        
        st.markdown(f"""
        \u003cdiv class="prediction-card"\u003e
            \u003ch3 class="pred-title"\u003eDự báo T+7\u003c/h3\u003e
            \u003ch1 class="pred-price"\u003e${pred['price']:.4f}\u003c/h1\u003e
            \u003cp class="pred-sub"\u003e{pred['date'].strftime('%d/%m/%Y')}\u003c/p\u003e
        \u003c/div\u003e
        """, unsafe_allow_html=True)


def display_ml_ensemble_results():
    """Display ML Ensemble training \u0026 prediction results"""
    st.subheader("Kết quả ML Ensemble")
    
    metrics = st.session_state.ml_ensemble_metrics
    test_metrics = metrics['test']
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (Test)", f"{test_metrics['mae']:.4f}")
    col2.metric("RMSE (Test)", f"{test_metrics['rmse']:.4f}")
    
    # Individual model R² scores
    if 'train' in metrics:
        st.write("**Individual Model Scores:**")
        train_metrics = metrics['train']
        
        cols = st.columns(3)
        for i, (name, m) in enumerate(train_metrics.items()):
            cols[i].metric(f"{name} R²", f"{m['train_r2']:.4f}")
    
    # Prediction display
    if 'ml_ensemble_prediction' in st.session_state:
        pred = st.session_state.ml_ensemble_prediction
        
        st.markdown(f"""
        \u003cdiv class="prediction-card"\u003e
            \u003ch3 class="pred-title"\u003eDự báo T+7\u003c/h3\u003e
            \u003ch1 class="pred-price"\u003e${pred['price']:.4f}\u003c/h1\u003e
            \u003cp class="pred-sub"\u003e{pred['date'].strftime('%d/%m/%Y')}\u003c/p\u003e
        \u003c/div\u003e
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
