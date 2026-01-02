import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go

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
    get_next_trading_date, format_number, calculate_change_percent,
    append_prediction_to_csv, validate_data
)

# Page config
st.set_page_config(
    page_title="Hệ thống Dự báo Giá XRP Đa tầng",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header */
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #f0f2f6;
        text-align: left;
        padding: 1rem 0;
        border-bottom: 1px solid #31333F;
        margin-bottom: 2rem;
    }
    
    /* Metrics Cards */
    .metric-card {
        background-color: #262730;
        border: 1px solid #31333F;
        padding: 1.5rem;
        border-radius: 8px;
        color: white;
    }
    
    /* Prediction Box (Professional Financial Card) */
    .prediction-box {
        background-color: #0e1117;
        border: 1px solid #41444C;
        padding: 2rem;
        border-radius: 6px;
        text-align: center;
        color: white;
    }
    
    /* Buttons - Minimalist Professional */
    .stButton>button {
        background-color: #262730; /* Dark gray */
        color: #ffffff;
        border: 1px solid #41444C;
        border-radius: 6px; /* Small radius */
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        border-color: #00D9FF; /* Highlight accent on hover */
        background-color: #31333F;
        color: #00D9FF;
        transform: none; /* No scaling */
        box-shadow: none;
    }
    
    /* DataFrame Tables */
    .stDataFrame {
        border: 1px solid #31333F;
        border-radius: 6px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0;
        color: #9da3af;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #00D9FF;
        border-bottom: 2px solid #00D9FF;
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
# Layer 2 paths
L2_RIDGE_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'layer2_ridge_model.pkl')
L2_RIDGE_SCALER_PATH = os.path.join(BASE_DIR, 'models', 'layer2_ridge_scaler.pkl')
L2_SVR_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'layer2_svr_model.pkl')
L2_SVR_SCALER_PATH = os.path.join(BASE_DIR, 'models', 'layer2_svr_scaler.pkl')
# Layer 3 paths
L3_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'layer3_lstm_model.keras')
L3_SCALER_PATH = os.path.join(BASE_DIR, 'models', 'layer3_scaler.pkl')
L3_TARGET_SCALER_PATH = os.path.join(BASE_DIR, 'models', 'layer3_target_scaler.pkl')

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
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'svr_metrics' not in st.session_state:
    st.session_state.svr_metrics = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None

# Layer 2 Session States
if 'l2_ridge_model_trained' not in st.session_state:
    st.session_state.l2_ridge_model_trained = False
if 'l2_ridge_model' not in st.session_state:
    st.session_state.l2_ridge_model = None
if 'l2_ridge_scaler' not in st.session_state:
    st.session_state.l2_ridge_scaler = None

if 'l2_svr_model_trained' not in st.session_state:
    st.session_state.l2_svr_model_trained = False
if 'l2_svr_model' not in st.session_state:
    st.session_state.l2_svr_model = None
if 'l2_svr_scaler' not in st.session_state:
    st.session_state.l2_svr_scaler = None

# Layer 3 Session States
if 'l3_model_trained' not in st.session_state:
    st.session_state.l3_model_trained = False
if 'l3_model' not in st.session_state:
    st.session_state.l3_model = None
if 'l3_scaler' not in st.session_state:
    st.session_state.l3_scaler = None
if 'l3_target_scaler' not in st.session_state:
    st.session_state.l3_target_scaler = None

def main():

    # Header
    st.markdown('<h1 class="main-header">HỆ THỐNG DỰ BÁO GIÁ XRP/USDT</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Bảng Điều Khiển")
        st.markdown("""
        **Kiến trúc Hệ thống:**
        1. **Lớp 1 (Máy học)**: Xác định Xu hướng (RandomForest, SVR)
        2. **Lớp 2 (Thống kê)**: Tinh chỉnh trong ngày (Ridge)
        3. **Lớp 3 (Học sâu)**: Dự báo chuỗi thời gian (LSTM)
        """)
        
        if st.button("Tải & Xử lý dữ liệu thô"):
            load_and_process_data()
        
        if st.session_state.df_features is not None:
            st.success("Dữ liệu đã sẵn sàng!")
            st.write(f"Tổng số dòng: {len(st.session_state.df_features)}")
    
    # Tabs for different Layers
    tab1, tab2, tab3 = st.tabs(["Lớp 1: Xu Hướng", "Lớp 2: Trong Ngày", "Lớp 3: Học Sâu"])
    
    with tab1:
        display_layer1_content()
    
    with tab2:
        display_layer2_content()
        
    with tab3:
        display_layer3_content()


def display_layer1_content():
    
    # Control buttons at top
    # Control buttons at top
    st.subheader("Điều khiển Mô hình Lớp 1")
    
    col_up, col_cmd = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader("Tải lên Dữ liệu CSV", type=['csv'], label_visibility="collapsed")
        if uploaded_file is not None:
            if st.button("Sử dụng Tệp đã Kiểm tra", use_container_width=True):
                load_and_process_data(uploaded_file)
    
    with col_cmd:
        if st.button("Xem Dữ liệu đã Xử lý", use_container_width=True, help="Hiển thị bảng đầy đủ các tính năng"):
            st.session_state.show_processed_data = not st.session_state.get('show_processed_data', False)
    
    # Hiển thị bảng dữ liệu đầy đủ nếu được yêu cầu
    if st.session_state.get('show_processed_data', False):
        if st.session_state.df_features is not None:
            st.markdown("### Toàn bộ dữ liệu đã xử lý (Features)")
            st.write("Bảng dưới đây bao gồm tất cả các chỉ số kỹ thuật đã được tính toán:")
            st.dataframe(st.session_state.df_features, use_container_width=True)
            if st.button("Đóng bảng dữ liệu"):
                st.session_state.show_processed_data = False
                st.rerun()
        else:
            st.warning("Chưa có dữ liệu để hiển thị!")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Huấn luyện Mô hình", expanded=True):
        train_col1, train_col2 = st.columns(2)
        with train_col1:
            if st.button("Huấn luyện RandomForest", use_container_width=True, disabled=st.session_state.df_features is None):
                train_model(model_type="RF")
        with train_col2:
            if st.button("Huấn luyện SVR", use_container_width=True, disabled=st.session_state.df_features is None):
                train_model(model_type="SVR")

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Hàng nút load/delete
    col_l1, col_l2, col_del = st.columns(3)
    with col_l1:
        if st.button("Tải Mô hình Chính (RF/SVR)", use_container_width=True):
            load_saved_model()
    with col_l2:
        if st.button("Tải Mô hình 7 Ngày", use_container_width=True):
            load_saved_7day_models()
    with col_del:
        if st.button("Xóa Mô hình Cũ", use_container_width=True):
            delete_old_models()
            
    st.markdown("<br>", unsafe_allow_html=True)
            
    # Hàng nút dự báo
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("Dự báo Ngày mai", use_container_width=True, 
                     disabled=not (st.session_state.model_trained or st.session_state.svr_model_trained)):
            make_prediction()
    with col_p2:
        if st.button("Dự báo 7 Ngày", use_container_width=True, 
                     disabled=not (st.session_state.model_trained or st.session_state.svr_model_trained)):
            make_7day_prediction()
    
    st.markdown("---")
    
    # Display dashboard if data is loaded
    if st.session_state.df_features is not None:
        display_dashboard()
    else:
        st.info("Vui lòng nhấn **Tải dữ liệu** để bắt đầu")


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
            st.success(f"Đã tải {len(df)} dòng dữ liệu từ {target_path if target_path else 'file nguồn'} thành công!")
            
        except Exception as e:
            st.error(f"Lỗi khi xử lý dữ liệu: {e}")


#### train model
def train_model(model_type="RF"):
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
            
            st.info(f"🛠️ Đang chuẩn bị dữ liệu với {len(feature_cols)} features...")
            
            # Prepare data
            X_train, X_test, y_train, y_test, _ = prepare_data_for_training(
                st.session_state.df_features,
                feature_columns=feature_cols,
                target_column='Target_Price',
                test_size=0.5
            )
            
            # Lưu danh sách features vào session state để dùng khi dự đoán
            st.session_state.feature_cols = feature_cols
            
            if model_type == "RF":
                # Train RF
                model, scaler = train_layer1_model(X_train, y_train)
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
                model, scaler = train_svr_model(X_train, y_train)
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
            st.success(f"Huấn luyện mô hình {model_name} thành công!")
            
            # Store metrics specifically
            if model_type == "RF":
                st.session_state.metrics = metrics
            else:
                st.session_state.svr_metrics = metrics
                
        except Exception as e:
            st.error(f"Lỗi khi huấn luyện mô hình {model_name}: {e}")
            import traceback
            st.error(traceback.format_exc())


def load_saved_model():
    """Load pre-trained models Layer 1 (RF & SVR)"""
    with st.spinner("Đang tải các mô hình Layer 1 đã lưu..."):
        try:
            # Feature columns are shared - lấy 1 lần để dùng chung
            if st.session_state.feature_cols is None:
                st.session_state.feature_cols = get_feature_columns()
            
            loaded_any = False
            
            # Load RF
            try:
                rf_model = load_model(L1_MODEL_PATH)
                rf_scaler = load_model(L1_SCALER_PATH)
                multi_models = load_model(L1_MULTI_MODELS_PATH)
                multi_scalers = load_model(L1_MULTI_SCALERS_PATH)
                
                if rf_model and rf_scaler:
                    if hasattr(rf_scaler, 'n_features_in_') and rf_scaler.n_features_in_ != len(st.session_state.feature_cols):
                        st.error(f"⚠️ Scaler RF cũ ({rf_scaler.n_features_in_} cột) k khớp với code mới {len(st.session_state.feature_cols)}. Bỏ qua tải RF.")
                        # Xóa file lỗi để tránh lặp lại
                        if os.path.exists(L1_SCALER_PATH): os.remove(L1_SCALER_PATH)
                    else:
                        st.session_state.model = rf_model
                        st.session_state.scaler = rf_scaler
                        st.session_state.l1_multi_models = multi_models
                        st.session_state.l1_multi_scalers = multi_scalers
                        st.session_state.model_trained = True
                        loaded_any = True
                        st.info("✅ Đã tải mô hình RandomForest")
            except Exception as e:
                st.warning(f"Không thể tải RF: {e}")

            # Load SVR
            try:
                svr_model = load_model(L1_SVR_MODEL_PATH)
                svr_scaler = load_model(L1_SVR_SCALER_PATH)
                
                if svr_model and svr_scaler:
                    if hasattr(svr_scaler, 'n_features_in_') and svr_scaler.n_features_in_ != len(st.session_state.feature_cols):
                        st.error(f"⚠️ Scaler SVR cũ ({svr_scaler.n_features_in_} cột) k khớp code mới {len(st.session_state.feature_cols)}. Bỏ qua tải SVR.")
                        if os.path.exists(L1_SVR_SCALER_PATH): os.remove(L1_SVR_SCALER_PATH)
                    else:
                        st.session_state.svr_model = svr_model
                        st.session_state.svr_scaler = svr_scaler
                        st.session_state.svr_model_trained = True
                        loaded_any = True
                        st.info("✅ Đã tải mô hình SVR")
            except Exception as e:
                st.warning(f"Không thể tải SVR: {e}")
            
            if not loaded_any:
                st.warning("Không tìm thấy bất kỳ mô hình Layer 1 nào đã lưu.")
            else:
                st.success("Quá trình tải mô hình hoàn tất!")
            
        except Exception as e:
            st.error(f"Lỗi khi load mô hình L1: {e}")


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
                st.success("✅ Đã tải thành công bộ mô hình dự báo 7 ngày!")
            else:
                st.warning("⚠️ Không tìm thấy file mô hình 7 ngày đã lưu.")
        except Exception as e:
            st.error(f"Lỗi khi tải mô hình 7 ngày: {e}")


def delete_old_models():
    """Xóa tất cả các file model đã lưu trong thư mục models"""
    models_dir = './models/'
    try:
        if os.path.exists(models_dir):
            files = os.listdir(models_dir)
            if not files:
                st.info("Không có model nào để xóa.")
                return
                
            for file in files:
                file_path = os.path.join(models_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            # Reset session state
            st.session_state.model = None
            st.session_state.scaler = None
            st.session_state.l1_multi_models = None
            st.session_state.l1_multi_scalers = None
            st.session_state.svr_model = None
            st.session_state.svr_scaler = None
            st.session_state.svr_model_trained = False
            st.session_state.model_trained = False
            if 'metrics' in st.session_state:
                del st.session_state.metrics
            if 'svr_metrics' in st.session_state:
                del st.session_state.svr_metrics
            if 'prediction' in st.session_state:
                del st.session_state.prediction
            if 'prediction_7days' in st.session_state:
                del st.session_state.prediction_7days
                
            st.success("Đã xóa tất cả model cũ thành công!")
        else:
            st.info("Thư mục model không tồn tại.")
    except Exception as e:
        st.error(f"Lỗi khi xóa model: {e}")


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
            print(f"✅ Đã xuất dữ liệu df_features ra file: debug_df_features.csv")
            
            latest_row = df.iloc[-1]
            print("\n" + "🚀 " + "="*60)
            print("🔍 DEBUG: CHI TIẾT DÒNG DỮ LIỆU CUỐI CÙNG (LATEST ROW)")
            print("-" * 64)
            print(latest_row.to_string())
            print("-" * 64)
            print("🚀 " + "="*60 + "\n")
            
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
                    st.warning("⚠️ Mô hình SVR hiện tại không tương thích với dữ liệu mới. Hệ thống sẽ bỏ qua SVR trong lần này. Vui lòng nhấn nút 'Train SVR' để huấn luyện lại!")
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
            
            st.success("Đã cập nhật dự đoán so sánh!")
            
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
            st.success("Đã hoàn thành dự đoán xu hướng 7 ngày!")
            
        except Exception as e:
            st.error(f"Lỗi khi dự đoán 7 ngày: {e}")
            import traceback
            st.error(traceback.format_exc())


def update_csv_with_prediction(prediction_val, col_name='RF_Pred_Tomorrow'):
    """Update the latest row in CSV with the prediction value"""
    try:
        df_csv = pd.read_csv(DISPLAY_DATA_PATH)
        # Nếu cột chưa có thì tạo mới
        if col_name not in df_csv.columns:
            df_csv[col_name] = pd.NA
            
        # Assuming Date is unique and sorted -> update last row
        # Xác định vị trí cột cẩn thận
        col_idx = df_csv.columns.get_loc(col_name)
        df_csv.iloc[-1, col_idx] = prediction_val
        
        df_csv.to_csv(DISPLAY_DATA_PATH, index=False)
        return True
    except Exception as e:
        st.error(f"Lỗi khi cập nhật CSV: {e}")
        return False


def display_dashboard():
    """Display main dashboard merging source data and saved predictions"""
    df = st.session_state.df_features
    
    # Load display data for predictions
    df_display = None
    if os.path.exists(DISPLAY_DATA_PATH):
        try:
            df_display = pd.read_csv(DISPLAY_DATA_PATH)
        except:
            pass
            
    # Latest data section - Only show latest date and single row
    st.header("Dữ liệu Thị trường Mới nhất")
    
    latest = get_latest_row(df)
    
    # Display latest date prominently
    st.subheader(f"Ngày: {latest['Date'].strftime('%d/%m/%Y')}")
    
    # Metrics in one row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Đóng",
            f"${format_number(latest['Price'])}",
            f"{format_number(latest['Return_1d'] if 'Return_1d' in latest else 0, 2)}%"
        )
    
    with col2:
        st.metric("Mở", f"${format_number(latest['Open'])}")
    
    with col3:
        st.metric("Cao", f"${format_number(latest['High'])}")
    
    with col4:
        st.metric("Thấp", f"${format_number(latest['Low'])}")
    
    with col5:
        st.metric("Khối lượng", f"{int(latest['Vol']):,}")
    
    # Show only the latest row in a clean table
    st.subheader("Chi tiết Dữ liệu Mới nhất")
    
    # Determine which columns to show as requested by user
    base_cols = ['Date', 'Price', 'Open', 'High', 'Low', 'Vol']
    latest_row_df = df[base_cols].tail(1).copy()

    # Add prediction columns from DISPLAY_DATA_PATH if available
    if df_display is not None and not df_display.empty:
        last_display = df_display.iloc[-1]
        if 'RF_Pred_Tomorrow' in df_display.columns:
            latest_row_df['RF_Pred_Tomorrow'] = last_display['RF_Pred_Tomorrow']
        if 'RF_Pred_Today' in df_display.columns:
            latest_row_df['RF_Pred_Today'] = last_display['RF_Pred_Today']
            
        # Thêm hiển thị cho SVR
        if 'SVR_Pred_Tomorrow' in df_display.columns:
            latest_row_df['SVR_Pred_Tomorrow'] = last_display['SVR_Pred_Tomorrow']
        if 'SVR_Pred_Today' in df_display.columns:
            latest_row_df['SVR_Pred_Today'] = last_display['SVR_Pred_Today']

    latest_row_df['Date'] = latest_row_df['Date'].dt.strftime('%d/%m/%Y')
    
    # Format numeric columns
    price_cols = ['Price', 'Open', 'High', 'Low', 'RF_Pred_Tomorrow', 'RF_Pred_Today', 'SVR_Pred_Tomorrow', 'SVR_Pred_Today']
    for col in price_cols:
        if col in latest_row_df.columns:
            latest_row_df[col] = latest_row_df[col].apply(lambda x: f"${x:.4f}" if pd.notna(x) else "N/A")
    
    if 'Vol' in latest_row_df.columns:
        latest_row_df['Vol'] = latest_row_df['Vol'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
    
    st.dataframe(latest_row_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")

    # Logic tự động hiển thị form nhập liệu:
    # Nếu dữ liệu mới nhất đã có dự báo RF (Tomorrow & Today) -> Hiển thị form để nhập thực tế cho ngày tiếp theo
    should_show_input = st.session_state.show_manual_input
    
    if df_display is not None and not df_display.empty:
        last_display = df_display.iloc[-1]
        has_rf_tomorrow = 'RF_Pred_Tomorrow' in last_display and pd.notna(last_display['RF_Pred_Tomorrow'])
        has_rf_today = 'RF_Pred_Today' in last_display and pd.notna(last_display['RF_Pred_Today'])
        
        if has_rf_tomorrow and has_rf_today:
            should_show_input = True
            st.success("✅ Đã có dự báo đầy đủ. Mời nhập dữ liệu thực tế cho ngày tiếp theo bên dưới 👇")

    # Hiển thị form nhập dữ liệu thủ công
    if should_show_input:
        display_manual_input_form()
        st.markdown("---")
    
    # Display prediction 1-day if available
    if 'prediction' in st.session_state:
        display_prediction_inline()
        st.markdown("---")
    
    # Display 7-day prediction if available
    if 'prediction_7days' in st.session_state:
        display_7day_prediction_inline()
        st.markdown("---")
    
    # Charts section
    st.header("Phân tích Giá")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Lịch sử Giá", "Biểu đồ Nến", "Khối lượng", "Chỉ báo Kỹ thuật"])
    
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
    # Model performance comparison
    if (st.session_state.model_trained and 'metrics' in st.session_state) or \
       (st.session_state.svr_model_trained and 'svr_metrics' in st.session_state):
        
        st.markdown("---")
        st.header("📊 So sánh hiệu suất mô hình")
        
        m_tabs = []
        if st.session_state.model_trained: m_tabs.append("🌲 RandomForest")
        if st.session_state.svr_model_trained: m_tabs.append("📈 SVR")
        
        if m_tabs:
            tabs = st.tabs(m_tabs)
            
            tab_idx = 0
            if st.session_state.model_trained:
                with tabs[tab_idx]:
                    if 'metrics' in st.session_state and st.session_state.metrics is not None:
                        metrics = st.session_state.metrics
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("MAE", f"{metrics['MAE']:.6f}")
                        c2.metric("RMSE", f"{metrics['RMSE']:.6f}")
                        c3.metric("R² Score", f"{metrics['R2']:.4f}")
                        c4.metric("Hướng", f"{metrics['Direction_Accuracy']:.2f}%")
                    else:
                        st.info("Chưa có thông tin đánh giá mô hình RF. Vui lòng huấn luyện lại để xem chi tiết.")
                    
                    if st.checkbox("Feature Importance (RF)", key="show_fi_rf"):
                        feature_imp = get_feature_importance(st.session_state.model, st.session_state.feature_cols, top_n=15)
                        st.plotly_chart(plot_feature_importance(feature_imp), use_container_width=True)
                tab_idx += 1
                
            if st.session_state.svr_model_trained:
                with tabs[tab_idx]:
                    if 'svr_metrics' in st.session_state and st.session_state.svr_metrics is not None:
                        metrics = st.session_state.svr_metrics
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("MAE", f"{metrics['MAE']:.6f}")
                        c2.metric("RMSE", f"{metrics['RMSE']:.6f}")
                        c3.metric("R² Score", f"{metrics['R2']:.4f}")
                        c4.metric("Hướng", f"{metrics['Direction_Accuracy']:.2f}%")
                    else:
                        st.info("Chưa có thông tin đánh giá mô hình SVR. Vui lòng huấn luyện lại để xem chi tiết.")
                    st.info("💡 SVR không hỗ trợ tính toán trực tiếp Feature Importance như RandomForest.")


def display_prediction_inline():
    """Display prediction results inline with comparison"""
    if 'prediction' not in st.session_state:
        return
    
    pred = st.session_state.prediction
    results = pred['results']
    
    st.header("Kết quả Dự báo So sánh")
    
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
            <div class="prediction-box">
                <h3 style="color: #dfe6e9; margin-bottom: 0.5rem; font-weight: 500;">{title}</h3>
                <p style="color: #b2bec3; font-size: 0.85rem;">Mục tiêu: {pred['date'].strftime('%d/%m/%Y')}</p>
                <h1 style="font-size: 2.5rem; margin: 1rem 0; font-weight: 700;">${format_number(data['price'])}</h1>
                <div style="margin-bottom: 1rem;">
                    <span style="color: {color}; font-weight: 600; font-size: 1.2rem;">{arrow} {format_number(abs(change_pct), 2)}%</span>
                </div>
                <div style="border-top: 1px solid #2d3436; padding-top: 0.8rem; margin-top: 0.8rem;">
                    <p style="color: #636e72; font-size: 0.8rem; margin: 0;">Khoảng tin cậy</p>
                    <p style="color: #b2bec3; font-weight: 500;">${format_number(data['lower'])} - ${format_number(data['upper'])}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    # Save action for RF
    with col1:
        if 'RF' in results:
            if st.button("Lưu Dự báo RF vào CSV", use_container_width=True):
                save_prediction_to_csv(model_type='RF')

    # Save action for SVR
    with col2:
        if 'SVR' in results:
            if st.button("Lưu Dự báo SVR vào CSV", use_container_width=True):
                save_prediction_to_csv(model_type='SVR')


def display_7day_prediction_inline():
    """Display 7-day forecast results with table and chart"""
    st.header("Dự báo Giá 7 Ngày")
    
    forecast_df = st.session_state.prediction_7days
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Bảng Dự báo")
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
        success = update_csv_with_prediction(pred_price, col_name=target_col)
        
        if success:
            st.success(f"Đã cập nhật dự đoán {model_type} cho ngày {pred['date'].strftime('%d/%m/%Y')} vào dữ liệu hiện có!")
            load_and_process_data(target_path=DISPLAY_DATA_PATH) # Reload từ file vừa lưu
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
                if append_prediction_to_csv(DISPLAY_DATA_PATH, prediction_data):
                    st.success("Đã thêm dòng dự đoán mới vào CSV!")
                load_and_process_data(target_path=DISPLAY_DATA_PATH)
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
    
    st.subheader(f"Nhập dữ liệu thực tế cho ngày: {next_date.strftime('%d/%m/%Y')}")
    
    with st.form("manual_input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            price = st.number_input("Price (Giá đóng cửa)", value=float(df.iloc[-1]['Price']), format="%.4f")
            open_p = st.number_input("Open (Giá mở cửa)", value=float(df.iloc[-1]['Price']), format="%.4f")
        with col2:
            high = st.number_input("High (Giá cao nhất)", value=float(df.iloc[-1]['Price']), format="%.4f")
            low = st.number_input("Low (Giá thấp nhất)", value=float(df.iloc[-1]['Price']), format="%.4f")
        with col3:
            vol = st.number_input("Volume (Khối lượng)", value=int(df.iloc[-1]['Vol']), step=1000)
            
        submit = st.form_submit_button("Dự đoán cho ngày tiếp theo")
        
        if submit:
            handle_manual_input_submission(next_date, price, open_p, high, low, vol)
    
    # Hiển thị kết quả vừa dự đoán nếu có
    if 'last_manual_result' in st.session_state:
        st.markdown("#### Kết quả dự đoán cho dòng dữ liệu vừa nhập:")
        st.dataframe(st.session_state.last_manual_result, use_container_width=True, hide_index=True)


def handle_manual_input_submission(date, price, open_p, high, low, vol):
    """Xử lý lưu dữ liệu thực tế và TẤT CẢ các chỉ số kỹ thuật vào CSV"""
    try:
        # 1. Load dữ liệu hiện tại chỉ lấy các cột gốc để tránh bị lặp cột features cũ
        df_raw = load_data(DISPLAY_DATA_PATH)
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
            # Xử lý NaN cho features trước khi dự báo
            df_for_pred = df_all_features[feature_cols].copy().ffill().fillna(0)
            latest_features = df_for_pred.iloc[-1:].values
            
            # Dự báo giá cho ngày tiếp theo
            pred_val = predict_next_day_layer1(st.session_state.model, st.session_state.scaler, latest_features)
            df_all_features.loc[df_all_features.index[-1], 'RF_Pred_Tomorrow'] = pred_val
            
        # 5. Cập nhật RF_Pred_Today (Lấy dự báo của ngày trước đó gán cho hôm nay)
        if 'RF_Pred_Tomorrow' in df_all_features.columns:
            df_all_features['RF_Pred_Today'] = df_all_features['RF_Pred_Tomorrow'].shift(1)
            
        # 6. Lưu TOÀN BỘ dataframe với hàng trăm cột vào CSV
        # Chuyển Date sang string YYYY-MM-DD trước khi lưu
        df_save = df_all_features.copy()
        df_save['Date'] = df_save['Date'].dt.strftime('%Y-%m-%d')
        df_save.to_csv(DISPLAY_DATA_PATH, index=False)
        
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
    st.header("🎯 Dự đoán giá trong ngày (Layer 2)")
    
    if st.session_state.df_features is None:
        st.info("Vui lòng tải dữ liệu ở Sidebar trước.")
        return

    # Train/Load buttons for L2
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train mô hình Layer 2", use_container_width=True, disabled=not st.session_state.model_trained):
            train_layer2_logic()
    with col2:
        if st.button("Load Layer 2 model", use_container_width=True):
            load_l2_model()

    st.markdown("---")

    # Prediction Section
    st.subheader("🔮 Dự đoán giá chốt phiên trực tuyến")
    
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
        st.warning(f"⚠️ Chưa có đủ dự đoán Layer 1 (RF & SVR) cho ngày {target_date.strftime('%d/%m/%Y')}. Vui lòng qua Tab Layer 1 huấn luyện và dự đoán cả 2 mô hình trước.")
        return

    st.success(f"📅 Mục tiêu: Dự đoán giá Đóng cửa cho ngày **{target_date.strftime('%d/%m/%Y')}**")
    col_l1a, col_l1b = st.columns(2)
    col_l1a.info(f"💡 RF L1: **${l1_rf_target:.4f}**")
    col_l1b.info(f"💡 SVR L1: **${l1_svr_target:.4f}**")

    # 2. User Input
    with st.form("layer2_form"):
        st.write(f"Nhập dữ liệu thị trường thực tế của ngày {target_date.strftime('%d/%m/%Y')}:")
        col1, col2 = st.columns(2)
        with col1:
            open_price = st.number_input("Giá mở cửa (Open)", value=None, placeholder="Nhập giá mở cửa...", format="%.4f")
            high_price = st.number_input("Giá cao nhất (High)", value=None, placeholder="Nhập giá cao nhất...", format="%.4f")
        with col2:
            current_vol = st.number_input("Khối lượng dự kiến (Volume)", value=None, placeholder="Nhập khối lượng dự kiến...", format="%.0f")
            low_price = st.number_input("Giá thấp nhất (Low)", value=None, placeholder="Nhập giá thấp nhất...", format="%.4f")
        
        submit = st.form_submit_button("🔥 Tính toán giá chốt phiên (Layer 2)")

    if submit:
        if any(v is None for v in [open_price, high_price, low_price, current_vol]):
            st.error("Vui lòng nhập đầy đủ giá Open, High, Low và Volume của ngày hôm nay!")
        elif not (st.session_state.l2_ridge_model_trained or st.session_state.l2_svr_model_trained):
            st.error("Vui lòng train Layer 2 tại Tab này trước!")
        else:
            try:
                # Prepare L2 input: [Open, High, Low, Vol, RF_Pred_Today, SVR_Pred_Today]
                l2_input = np.array([[open_price, high_price, low_price, current_vol, l1_rf_target, l1_svr_target]])
                
                res_col1, res_col2 = st.columns(2)
                
                if st.session_state.l2_ridge_model_trained:
                    pred_ridge = predict_layer2(st.session_state.l2_ridge_model, st.session_state.l2_ridge_scaler, l2_input)
                    with res_col1:
                        st.markdown(f"""
                        <div class="prediction-box" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                            <h3 style="color: white; margin-bottom: 0px;">L2: Ridge (Thống kê)</h3>
                            <h1 style="color: white; font-size: 3rem; margin-top: 10px;">${pred_ridge:.4f}</h1>
                            <p style="color: white; font-size: 0.9rem;">(Dựa trên O-H-L-V & L1 Hybrid)</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                if st.session_state.l2_svr_model_trained:
                    pred_l2_svr = predict_layer2(st.session_state.l2_svr_model, st.session_state.l2_svr_scaler, l2_input)
                    with res_col2:
                        st.markdown(f"""
                        <div class="prediction-box" style="background: linear-gradient(135deg, #FF512F 0%, #DD2476 100%);">
                            <h3 style="color: white; margin-bottom: 0px;">L2: SVR (Máy học)</h3>
                            <h1 style="color: white; font-size: 3rem; margin-top: 10px;">${pred_l2_svr:.4f}</h1>
                            <p style="color: white; font-size: 0.9rem;">(Dựa trên O-H-L-V & L1 Hybrid)</p>
                        </div>
                        """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Lỗi dự đoán L2: {e}")

def train_layer2_logic():
    """Train Layer 2 (Ridge & SVR) using both L1 predictions"""
    with st.spinner("Đang chuẩn bị dữ liệu và huấn luyện Layer 2..."):
        try:
            if st.session_state.df_features is None:
                st.error("Vui lòng tải dữ liệu trước!")
                return

            # Check L1 models
            if not st.session_state.model_trained or not st.session_state.svr_model_trained:
                st.error("⚠️ Cần huấn luyện cả RandomForest và SVR (Layer 1) trước khi train Layer 2!")
                return

            df = st.session_state.df_features.copy()
            feature_cols = st.session_state.feature_cols
            
            # --- Generate L1 Projections for History if missing ---
            # We need RF_Pred_Today and SVR_Pred_Today for the entire dataset to train L2
            
            # Prepare X_all for batch prediction
            # Note: We must handle NaNs like we did in training
            df_to_pred = df[feature_cols].copy().ffill().fillna(0)
            X_all = df_to_pred.values
            
            # 1. Generate RF Predictions
            rf_scaler = st.session_state.scaler
            rf_model = st.session_state.model
            X_all_rf_scaled = rf_scaler.transform(X_all)
            df['RF_Pred_Tomorrow'] = rf_model.predict(X_all_rf_scaled)
            df['RF_Pred_Today'] = df['RF_Pred_Tomorrow'].shift(1)
            
            # 2. Generate SVR Predictions
            svr_scaler = st.session_state.svr_scaler
            svr_model = st.session_state.svr_model
            
            # Check SVR consistency
            if hasattr(svr_scaler, 'n_features_in_') and svr_scaler.n_features_in_ != X_all.shape[1]:
                 st.error("⚠️ Model SVR cũ không khớp số lượng features hiện tại. Vui lòng Train lại SVR!")
                 return

            X_all_svr_scaled = svr_scaler.transform(X_all)
            df['SVR_Pred_Tomorrow'] = svr_model.predict(X_all_svr_scaled)
            df['SVR_Pred_Today'] = df['SVR_Pred_Tomorrow'].shift(1)
            
            # Update session state with new columns
            st.session_state.df_features = df
            
            # --- Prepare L2 Data ---
            # Features are: Open, High, Low, Vol, RF_Pred_Today, SVR_Pred_Today
            l2_features = ['Open', 'High', 'Low', 'Vol', 'RF_Pred_Today', 'SVR_Pred_Today']
            target = 'Price'
            
            # Drop NaNs created by shifting
            df_l2 = df.dropna(subset=l2_features + [target])
            
            if len(df_l2) < 50:
                st.error("Không đủ dữ liệu sạch để train Layer 2 (do thiếu history prediction).")
                return
                
            X = df_l2[l2_features]
            y = df_l2[target]
            
            # Split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train Ridge
            ridge_model, ridge_scaler = train_layer2_model(X_train, y_train)
            save_model(ridge_model, L2_RIDGE_MODEL_PATH)
            save_model(ridge_scaler, L2_RIDGE_SCALER_PATH)
            
            # Train SVR for L2
            svr_model_l2, svr_scaler_l2 = train_svr_model(X_train, y_train)
            save_model(svr_model_l2, L2_SVR_MODEL_PATH)
            save_model(svr_scaler_l2, L2_SVR_SCALER_PATH)
            
            st.session_state.l2_ridge_model = ridge_model
            st.session_state.l2_ridge_scaler = ridge_scaler
            st.session_state.l2_ridge_model_trained = True
            
            st.session_state.l2_svr_model = svr_model_l2
            st.session_state.l2_svr_scaler = svr_scaler_l2
            st.session_state.l2_svr_model_trained = True
            
            st.success(f"Đã train Layer 2 (Ridge & SVR) thành công!")
            
        except Exception as e:
            st.error(f"Lỗi khi train L2: {e}")
            import traceback
            st.error(traceback.format_exc())


def load_l2_model():
    """Load Layer 2 models"""
    with st.spinner("Đang tải các mô hình Layer 2..."):
        try:
            # Load Ridge
            ridge_model = load_model(L2_RIDGE_MODEL_PATH)
            ridge_scaler = load_model(L2_RIDGE_SCALER_PATH)
            if ridge_model and ridge_scaler:
                st.session_state.l2_ridge_model = ridge_model
                st.session_state.l2_ridge_scaler = ridge_scaler
                st.session_state.l2_ridge_model_trained = True
                st.info("✅ Đã tải mô hình L2 Ridge")
                
            # Load SVR
            svr_model = load_model(L2_SVR_MODEL_PATH)
            svr_scaler = load_model(L2_SVR_SCALER_PATH)
            if svr_model and svr_scaler:
                st.session_state.l2_svr_model = svr_model
                st.session_state.l2_svr_scaler = svr_scaler
                st.session_state.l2_svr_model_trained = True
                st.info("✅ Đã tải mô hình L2 SVR")
                
            st.success("Tải mô hình Layer 2 hoàn tất!")
        except Exception as e:
            st.error(f"Lỗi khi load L2 models: {e}")


def display_layer3_content():
    """Hiển thị nội dung cho Layer 3 (LSTM)"""
    st.subheader("Layer 3: Dự báo chuỗi thời gian bằng Deep Learning (LSTM)")
    
    # Cho phép chọn file CSV riêng cho Layer 3
    st.markdown("### 📁 Chọn dữ liệu cho Layer 3")
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
            st.success(f"Đã tải dữ liệu từ {uploaded_file.name} cho Layer 3!")
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
            st.success(f"Đã chuẩn bị {len(df_l3.columns)} features cho LSTM!")
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
            
            st.success("Huấn luyện Layer 3 (LSTM) thành công!")
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
                st.success("Đã load Layer 3 thành công!")
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
        st.success("Đã hoàn thành dự báo LSTM cho 7 ngày tới!")
        
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
        st.subheader("Bảng dự báo 7 ngày")
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


if __name__ == "__main__":
    main()
