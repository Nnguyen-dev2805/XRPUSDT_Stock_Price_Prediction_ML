import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

EPS = 1e-8

# ==============================================
# LSTM ACTIVATIONS & DERIVATIVES
# ==============================================

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def dsigmoid(y):
    return y * (1 - y)

def dtanh(y):
    return 1 - y**2

# ==============================================
# CUSTOM LSTM CLASS (MATCHING NOTEBOOK)
# ==============================================

class RegimeLSTM:
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Xavier initialization style (0.1 scale as in notebook)
        self.Wf = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.1
        self.Wi = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.1
        self.Wc = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.1
        self.Wo = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.1
        self.Wy = np.random.randn(hidden_dim, 1) * 0.1
    
    def forward(self, X):
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        self.cache = []
        
        for x in X:
            z = np.concatenate([h, x])
            
            f = sigmoid(z @ self.Wf)
            i = sigmoid(z @ self.Wi)
            c_hat = np.tanh(z @ self.Wc)
            o = sigmoid(z @ self.Wo)
            
            c_prev = c
            c = f * c_prev + i * c_hat
            h = o * np.tanh(c)
            
            # Store everything needed for BPTT
            self.cache.append((z, f, i, c_hat, o, c, h, c_prev))
        
        y = h @ self.Wy
        return y.item(), h
    
    def backward(self, y_true, y_pred, lr):
        # Initial gradient from loss (MSE)
        dy = 2 * (y_pred - y_true)
        
        # Gradient for output weights
        h_last = self.cache[-1][6]
        self.Wy -= lr * np.outer(h_last, dy)
        
        # Backpropagation through time
        dh_next = self.Wy.flatten() * dy
        dc_next = np.zeros(self.hidden_dim)
        
        for t in reversed(range(len(self.cache))):
            z, f, i, c_hat, o, c, h, c_prev = self.cache[t]
            
            do = dh_next * np.tanh(c) * dsigmoid(o)
            dc = dh_next * o * dtanh(np.tanh(c)) + dc_next
            dc_hat = dc * i * dtanh(c_hat)
            di = dc * c_hat * dsigmoid(i)
            df = dc * c_prev * dsigmoid(f)
            
            self.Wf -= lr * np.outer(z, df)
            self.Wi -= lr * np.outer(z, di)
            self.Wc -= lr * np.outer(z, dc_hat)
            self.Wo -= lr * np.outer(z, do)
            
            # Gradient for next timestep (previous in time)
            dz = df @ self.Wf.T + di @ self.Wi.T + dc_hat @ self.Wc.T + do @ self.Wo.T
            dh_next = dz[:self.hidden_dim]
            dc_next = f * dc


# ==============================================
# FEATURE ENGINEERING (MATCHING 18 FEATURES)
# ==============================================

def create_regime_features(df):
    df = df.copy()
    
    # 1. VVR
    df['Volatility'] = df['High'] - df['Low']
    df['VVR'] = df['Vol'] / (df['Volatility'] + EPS)
    
    # 2. VWAP
    df['VWAP'] = (
        ((df['High'] + df['Low'] + df['Price']) / 3 * df['Vol']).cumsum()
        / (df['Vol'].cumsum() + EPS)
    )
    
    # 3-7. Lags
    for lag in [1, 2, 3, 5, 7]:
        df[f'Lag_{lag}'] = df['Price'].shift(lag)
    
    # 8. Change
    df['Change'] = df['Price'].pct_change()
    
    # 9-11. MAs
    df['MA5'] = df['Price'].rolling(5).mean()
    df['MA15'] = df['Price'].rolling(15).mean()
    df['MA30'] = df['Price'].rolling(30).mean()
    
    # 12-13. Regime Feature (vol_z & spike)
    df['vol_z'] = (
        df['Volatility'] - df['Volatility'].rolling(30).mean()
    ) / (df['Volatility'].rolling(30).std() + EPS)
    df['is_spike'] = (df['vol_z'] > 2).astype(int)
    
    # Ensure all required features for 18 total (Price omitted from features)
    # The notebook has 18 features in X_seq. Let's re-verify the list from notebook:
    # feature_cols = [c for c in df.columns if c not in ['Date', 'Price']]
    # Looking at notebook source: VVR, VWAP, Lag_1,2,3,5,7, Change, Volatility, MA5, MA15, MA30, vol_z, is_spike
    # High, Low, Vol are also in the dataframe.
    # Total cols in df after FE: Date, Price, High, Low, Vol, VVR, VWAP, Lag_1,2,3,5,7, Change, Volatility, MA5, MA15, MA30, vol_z, is_spike.
    # Total excluding Date and Price: 18 features. PERFECT.
    
    df = df.dropna().reset_index(drop=True)
    return df


# ==============================================
# DATA PREPARATION (MIN-MAX SCALING)
# ==============================================

def prepare_regime_data(df, lookback=30, horizon=7, test_size=0.4):
    feature_cols = [c for c in df.columns if c not in ['Date', 'Price']]
    X_raw = df[feature_cols].values
    y_raw = df['Price'].values.reshape(-1, 1)
    dates = df['Date'].values
    
    # Notebook uses MinMaxScaler(0, 1)
    sx = MinMaxScaler(feature_range=(0, 1))
    sy = MinMaxScaler(feature_range=(0, 1))
    
    X_scaled = sx.fit_transform(X_raw)
    y_scaled = sy.fit_transform(y_raw).ravel()
    
    X_seq, y_seq, y_dates = [], [], []
    
    # Sequence building logic: i -> i+lookback is X, i+lookback+horizon-1 is y
    # This precisely matches notebook: y_seq.append(y_scaled[i + LOOKBACK + HORIZON - 1])
    for i in range(len(X_scaled) - lookback - horizon):
        X_seq.append(X_scaled[i:i+lookback])
        y_seq.append(y_scaled[i+lookback+horizon-1])
        y_dates.append(dates[i+lookback+horizon-1])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    y_dates = np.array(y_dates)
    
    # Split train/test (Note: notebook uses 0.4 split, but split = int(len * 0.4) for TRAIN)
    # Then Test is the rest (60% test)
    split = int(len(X_seq) * (1 - test_size)) if test_size < 1 else int(len(X_seq) * 0.4)
    # To strictly match notebook where TRAIN = 40%:
    split = int(len(X_seq) * 0.4)
    
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    date_test = y_dates[split:]
    
    print(f"--- Data Prepared ---")
    print(f"X_train shape: {X_train.shape} (Sequences, Lookback, Features)")
    print(f"X_test shape: {X_test.shape}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"---------------------")
    
    scalers = {
        'X_scaler': sx,
        'y_scaler': sy,
        'feature_cols': feature_cols,
        'dates': date_test
    }
    
    return X_train, X_test, y_train, y_test, scalers


# ==============================================
# TRAINING
# ==============================================

def train_regime_lstm(df, epochs=70, lr=0.001, lookback=30, horizon=7, 
                     hidden_dim=32, test_size=0.4, verbose=True):
    """
    Train Regime LSTM model
    
    Args:
        df: Input DataFrame
        epochs: Number of training epochs
        lr: Learning rate
        lookback: Sequence lookback length
        horizon: Forecast horizon (T+horizon)
        hidden_dim: LSTM hidden units
        test_size: Train split ratio
        verbose: Print training progress
        
    Returns:
        model: Trained RegimeLSTM
        scalers: Dictionary with scalers and metadata
        metrics: Training metrics
    """
    # Create regime features only if not already present
    if 'vol_z' not in df.columns:
        df_regime = create_regime_features(df)
    else:
        df_regime = df.copy()
    
    # Prepare data
    X_train, X_test, y_train, y_test, scalers = prepare_regime_data(
        df_regime, lookback=lookback, horizon=horizon, test_size=test_size
    )
    
    # Initialize model
    input_dim = X_train.shape[2]
    model = RegimeLSTM(input_dim, hidden_dim)
    
    # Training loop
    train_losses = []
    for ep in range(epochs):
        losses = []
        for X, y in zip(X_train, y_train):
            yp, _ = model.forward(X)
            losses.append((yp - y)**2)
            model.backward(y, yp, lr)
        
        epoch_loss = np.mean(losses)
        train_losses.append(epoch_loss)
        
        if verbose:
            print(f"Epoch {ep+1}/{epochs} | Loss {epoch_loss:.6f}")
    
    # Evaluate on test set
    preds = np.array([model.forward(X)[0] for X in X_test])
    
    # Inverse transform
    y_true_inv = scalers['y_scaler'].inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred_inv = scalers['y_scaler'].inverse_transform(preds.reshape(-1, 1)).ravel()
    
    # Metrics
    mse = mean_squared_error(y_true_inv, y_pred_inv)
    mae = np.mean(np.abs(y_true_inv - y_pred_inv))
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'train_losses': train_losses,
        'y_true': y_true_inv,
        'y_pred': y_pred_inv
    }
    
    if verbose:
        print(f"\nFinal Test MSE: {mse:.6f}")
        print(f"Final Test MAE: {mae:.6f}")
    
    return model, scalers, metrics


# ==============================================
# PREDICTION
# ==============================================

def predict_regime_lstm(model, scalers, latest_data, lookback=30):
    """
    Make prediction using trained Regime LSTM
    
    Args:
        model: Trained RegimeLSTM
        scalers: Dictionary with scalers
        latest_data: DataFrame with latest N rows (>=lookback)
        lookback: Sequence length
        
    Returns:
        prediction: Predicted price
    """
    # Create regime features only if not already present
    if 'vol_z' not in latest_data.columns:
        df_regime = create_regime_features(latest_data)
    else:
        df_regime = latest_data.copy()
    
    # Get feature columns
    feature_cols = scalers['feature_cols']
    
    # Extract last lookback rows
    X_raw = df_regime[feature_cols].iloc[-lookback:].values
    
    # Scale
    X_scaled = scalers['X_scaler'].transform(X_raw)
    
    # Predict
    y_scaled, _ = model.forward(X_scaled)
    
    # Inverse transform
    y_pred = scalers['y_scaler'].inverse_transform([[y_scaled]])[0][0]
    
    return y_pred
