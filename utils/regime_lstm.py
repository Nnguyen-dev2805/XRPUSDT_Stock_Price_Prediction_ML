"""
Regime-aware LSTM implementation based on NotBack.ipynb
Custom LSTM với khả năng nhận diện chế độ volatility của thị trường
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

EPS = 1e-8

# ==============================================
# LSTM IMPLEMENTATION
# ==============================================

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def dsigmoid(y):
    """Derivative of sigmoid"""
    return y * (1 - y)

def dtanh(y):
    """Derivative of tanh"""
    return 1 - y**2


class RegimeLSTM:
    """
    Custom LSTM implementation with regime detection capability
    Based on NotBack.ipynb approach
    """
    
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize LSTM weights
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
        """
        self.hidden_dim = hidden_dim
        
        # Weight matrices
        self.Wf = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.1
        self.Wi = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.1
        self.Wc = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.1
        self.Wo = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.1
        self.Wy = np.random.randn(hidden_dim, 1) * 0.1
        self.b = np.zeros(hidden_dim)
    
    def forward(self, X):
        """
        Forward pass through LSTM
        
        Args:
            X: Input sequence (timesteps, features)
            
        Returns:
            prediction: Scalar output
            last_hidden: Final hidden state
        """
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        self.cache = []
        
        for x in X:
            # Concatenate hidden state and input
            z = np.concatenate([h, x])
            
            # Gates
            f = sigmoid(z @ self.Wf)  # Forget gate
            i = sigmoid(z @ self.Wi)  # Input gate
            c_hat = np.tanh(z @ self.Wc)  # Candidate cell state
            o = sigmoid(z @ self.Wo)  # Output gate
            
            # Update cell and hidden state
            c = f * c + i * c_hat
            h = o * np.tanh(c)
            
            # Cache for backward pass
            self.cache.append((z, f, i, c_hat, o, c, h))
        
        # Output prediction
        y = h @ self.Wy
        return y.item(), h
    
    def backward(self, y_true, y_pred, lr):
        """
        Simplified backward pass (only updates output weights)
        
        Args:
            y_true: Ground truth value
            y_pred: Predicted value
            lr: Learning rate
        """
        dy = 2 * (y_pred - y_true)
        h_last = self.cache[-1][6]
        self.Wy -= lr * np.outer(h_last, dy)


# ==============================================
# FEATURE ENGINEERING
# ==============================================

def create_regime_features(df):
    """
    Create regime-aware features for LSTM
    Based on NotBack.ipynb feature engineering
    
    Args:
        df: DataFrame with columns [Date, Open, High, Low, Price, Vol]
        
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # Basic volatility features
    df['Volatility'] = df['High'] - df['Low']
    df['VVR'] = df['Vol'] / (df['Volatility'] + EPS)
    
    # Volume Weighted Average Price
    df['VWAP'] = (
        (df['High'] + df['Low'] + df['Price']) / 3 * df['Vol']
    ).cumsum() / df['Vol'].cumsum()
    
    # Lagged prices
    for lag in [1, 2, 3, 5, 7]:
        df[f'Lag_{lag}'] = df['Price'].shift(lag)
    
    # Price changes and moving averages
    df['Change'] = df['Price'].pct_change()
    df['MA5'] = df['Price'].rolling(5).mean()
    df['MA10'] = df['Price'].rolling(10).mean()
    
    # ===== REGIME FEATURE =====
    # Volatility z-score (30-day rolling window)
    df['vol_z'] = (
        df['Volatility'] - df['Volatility'].rolling(30).mean()
    ) / (df['Volatility'].rolling(30).std() + EPS)
    
    # Spike indicator (regime detection)
    df['is_spike'] = (df['vol_z'] > 2).astype(int)
    
    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)
    
    return df


# ==============================================
# DATA PREPARATION
# ==============================================

def prepare_regime_data(df, lookback=30, horizon=7, test_size=0.4):
    """
    Prepare data for Regime LSTM training/testing
    
    Args:
        df: DataFrame with regime features
        lookback: Number of past days to look back
        horizon: Number of days to forecast ahead (T+horizon)
        test_size: Proportion for training (0.4 = 40% train, 60% test)
        
    Returns:
        X_train, X_test, y_train, y_test, scalers
    """
    # Select feature columns (exclude Date)
    feature_cols = [c for c in df.columns if c not in ['Date', 'Price']]
    X_raw = df[feature_cols].values
    y_raw = df['Price'].values.reshape(-1, 1)
    dates = df['Date'].values
    
    # Scaling
    sx = StandardScaler()
    sy = StandardScaler()
    
    X_scaled = sx.fit_transform(X_raw)
    y_scaled = sy.fit_transform(y_raw).ravel()
    
    # Build sequences
    X_seq, y_seq, y_dates = [], [], []
    
    for i in range(len(X_scaled) - lookback - horizon):
        X_seq.append(X_scaled[i:i+lookback])
        y_seq.append(y_scaled[i+lookback+horizon-1])
        y_dates.append(dates[i+lookback+horizon-1])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    y_dates = np.array(y_dates)
    
    # Split train/test
    split = int(len(X_seq) * test_size)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    date_test = y_dates[split:]
    
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

def train_regime_lstm(df, epochs=60, lr=0.001, lookback=30, horizon=7, 
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
    # Create regime features
    df_regime = create_regime_features(df)
    
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
        
        if verbose and ((ep + 1) % 10 == 0 or ep == 0):
            print(f"Epoch {ep+1}/{epochs} | Loss {epoch_loss:.5f}")
    
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
    # Create regime features
    df_regime = create_regime_features(latest_data)
    
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
