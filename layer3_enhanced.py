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
    
    st.markdown('\u003cdiv class="section-header"\u003eLỚP 3: HỌC SÂU \u0026 SEQUENCE LEARNING\u003c/div\u003e', unsafe_allow_html=True)
    
    if st.session_state.df_features is None:
        st.info("Vui lòng tải dữ liệu ở Tab Layer 1 trước.")
        return
    
    # Create 3 tabs
    tab_keras, tab_regime, tab_ensemble = st.tabs([
        "🔵 LSTM (Keras)",
        "🟠 Regime LSTM", 
        "🟢 ML Ensemble"
    ])
    
    # ===================
    # TAB 1: Keras LSTM (Keep existing implementation)
    # ===================
    with tab_keras:
        display_keras_lstm_tab()
    
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
    st.subheader("Dự báo chuỗi thời gian với LSTM (Keras)")
    st.info("Phần này giữ nguyên implementation LSTM hiện tại. Vui lòng xem code cũ.")
    # TODO: Move existing L3 code here


def display_regime_lstm_tab():
    """Tab mới - Regime-aware LSTM"""
    st.subheader("🟠 Regime LSTM - Nhận diện chế độ volatility")
    
    st.info("""
    **Đặc điểm:**
    - Custom LSTM implementation (NumPy) với khả năng nhận diện regime
    - Sử dụng volatility z-score để phát hiện spike
    - Lookback: 30 days | Forecast: T+7
    """)
    
    # Training controls
    col_train, col_pred = st.columns([1, 1])
    
    with col_train:
        st.markdown("**Huấn luyện Mô hình**")
        
        with st.form("regime_lstm_form"):
            epochs = st.slider("Số epochs", 20, 100, 60, 10)
            lr = st.number_input("Learning rate", 0.0001, 0.01, 0.001, format="%.4f")
            
            train_btn = st.form_submit_button("Train Regime LSTM", type="primary", use_container_width=True)
        
        if train_btn:
            train_regime_lstm_model(epochs, lr)
    
    with col_pred:
        st.markdown("**Dự báo**")
        st.write("")  # Spacer
        st.write("")
        
        disabled = not st.session_state.regime_lstm_trained
        if st.button("Dự báo T+7", disabled=disabled, use_container_width=True, key="regime_predict"):
            make_regime_lstm_prediction()
        
        if st.session_state.regime_lstm_trained:
            st.success("✓ Model đã train")
        else:
            st.warning("Chưa train model")
    
    st.divider()
    
    # Display results
    if st.session_state.regime_lstm_metrics is not None:
        display_regime_lstm_results()


def display_ml_ensemble_tab():
    """Tab mới - ML Ensemble"""
    st.subheader("🟢 ML Ensemble - Stacking Approach")
    
    st.info("""
    **Đặc điểm:**
    - Ensemble: 0.5 RF + 0.3 GB + 0.2 Ridge
    - Flattened sequences (30 days × features)
    - Fast training với sklearn optimizations
    - Lookback: 30 days | Forecast: T+7
    """)
    
    # Training controls
    col_train, col_pred = st.columns([1, 1])
    
    with col_train:
        st.markdown("**Huấn luyện Mô hình**")
        st.write("")
        
        if st.button("Train ML Ensemble", type="primary", use_container_width=True):
            train_ml_ensemble_model()
    
    with col_pred:
        st.markdown("**Dự báo**")
        st.write("")
        
        disabled = not st.session_state.ml_ensemble_trained
        if st.button("Dự báo T+7", disabled=disabled, use_container_width=True, key="ensemble_predict"):
            make_ml_ensemble_prediction()
        
        if st.session_state.ml_ensemble_trained:
            st.success("✓ Model đã train")
        else:
            st.warning("Chưa train model")
    
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
            df = st.session_state.df_features.copy()
            
            # Ensure required columns
            if 'Price' not in df.columns:
                st.error("Dữ liệu thiếu cột 'Price'!")
                return
            
            # Train model
            model, scalers, metrics = train_regime_lstm(
                df, 
                epochs=epochs, 
                lr=lr, 
                lookback=30, 
                horizon=7,
                test_size=0.4,
                verbose=False
            )
            
            # Save to session
            st.session_state.regime_lstm_model = model
            st.session_state.regime_lstm_scalers = scalers
            st.session_state.regime_lstm_metrics = metrics
            st.session_state.regime_lstm_trained = True
            
            st.success(f"✓ Training hoàn tất! MAE: {metrics['mae']:.4f}")
            
        except Exception as e:
            st.error(f"Lỗi khi training: {e}")
            import traceback
            st.error(traceback.format_exc())


def train_ml_ensemble_model():
    """Train ML Ensemble model"""
    with st.spinner("Đang training ML Ensemble..."):
        try:
            df = st.session_state.df_features.copy()
            
            # Ensure required columns (rename if needed)
            if 'Price' in df.columns and 'Close' not in df.columns:
                df.rename(columns={'Price': 'Close'}, inplace=True)
            if 'Vol' in df.columns and 'Volume' not in df.columns:
                df.rename(columns={'Vol': 'Volume'}, inplace=True)
            
            # Train model
            ensemble, metrics = train_ml_ensemble(
                df,
                lookback=30,
                horizon=7,
                train_ratio=0.4,
                verbose=False
            )
            
            # Save to session
            st.session_state.ml_ensemble_model = ensemble
            st.session_state.ml_ensemble_metrics = metrics
            st.session_state.ml_ensemble_trained = True
            
            mae = metrics['test']['mae']
            st.success(f"✓ Training hoàn tất! MAE: {mae:.4f}")
            
        except Exception as e:
            st.error(f"Lỗi khi training: {e}")
            import traceback
            st.error(traceback.format_exc())


# ======================
# PREDICTION FUNCTIONS
# ======================

def make_regime_lstm_prediction():
    """Make prediction using Regime LSTM"""
    with st.spinner("Đang dự báo với Regime LSTM..."):
        try:
            df = st.session_state.df_features.copy()
            
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
            df = st.session_state.df_features.copy()
            
            # Rename columns if needed
            if 'Price' in df.columns and 'Close' not in df.columns:
                df.rename(columns={'Price': 'Close'}, inplace=True)
            if 'Vol' in df.columns and 'Volume' not in df.columns:
                df.rename(columns={'Vol': 'Volume'}, inplace=True)
            
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
