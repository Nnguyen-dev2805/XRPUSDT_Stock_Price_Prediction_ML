"""
ML Ensemble Forecaster implementation based on StackingMLTraditional.ipynb
Ensemble của RandomForest + GradientBoosting + Ridge với cấu trúc sequence
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

EPS = 1e-8
RANDOM_STATE = 42


# ==============================================
# FEATURE ENGINEERING
# ==============================================

def create_ml_features(df):
    """
    Create features for ML Ensemble (lighter than full advanced features)
    Based on StackingMLTraditional.ipynb
    
    Args:
        df: DataFrame with columns [Date, Open, High, Low, Price/Close, Vol/Volume]
        
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # Standardize column names
    if 'Price' in df.columns:
        df.rename(columns={'Price': 'Close'}, inplace=True)
    if 'Vol' in df.columns:
        df.rename(columns={'Vol': 'Volume'}, inplace=True)
    
    # Return and ranges
    df['Return_1d'] = df['Close'].pct_change()
    df['HL_Range'] = df['High'] - df['Low']
    df['OC_Range'] = abs(df['Open'] - df['Close'])
    
    # Simple Moving Averages
    for p in [5, 10, 20]:
        df[f'SMA_{p}'] = df['Close'].rolling(p).mean()
        df[f'Price_SMA_{p}'] = df['Close'] / (df[f'SMA_{p}'] + EPS) - 1
    
    # RSI (14-period)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + EPS)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df['Volatility_20'] = df['Return_1d'].rolling(20).std()
    
    # Drop NaN
    df = df.dropna().reset_index(drop=True)
    
    return df


# ==============================================
# ML ENSEMBLE CLASS
# ==============================================

class MLEnsembleForecaster:
    """
    ML-based sequence forecaster using ensemble of traditional models
    Mimics LSTM structure but uses flattened sequences
    """
    
    def __init__(self, lookback=30, horizon=7, train_ratio=0.4):
        """
        Initialize ML Ensemble
        
        Args:
            lookback: Number of past days in sequence
            horizon: Forecast horizon (T+horizon)
            train_ratio: Ratio for training split
        """
        self.lookback = lookback
        self.horizon = horizon
        self.train_ratio = train_ratio
        
        # Models
        self.models = {
            'RF': RandomForestRegressor(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=RANDOM_STATE
            ),
            'GB': GradientBoostingRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                random_state=RANDOM_STATE
            ),
            'Ridge': Ridge(alpha=1.0)
        }
        
        # Ensemble weights
        self.weights = {
            'RF': 0.5,
            'GB': 0.3,
            'Ridge': 0.2
        }
        
        # Scalers
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        # Metadata
        self.feature_cols = None
        self.is_trained = False
    
    def prepare_data(self, df):
        """
        Prepare data with sequence structure similar to LSTM
        
        Args:
            df: Input DataFrame
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Create ML features
        df_features = create_ml_features(df)
        
        # Get feature columns (exclude Date)
        self.feature_cols = [c for c in df_features.columns if c not in ['Date']]
        target_col = 'Close'
        target_idx = self.feature_cols.index(target_col)
        
        # Scale data
        scaled = self.scaler.fit_transform(df_features[self.feature_cols].values)
        
        # Fit target scaler
        self.target_scaler.fit(df_features[[target_col]].values)
        
        # Build sequences
        X, y = [], []
        
        for i in range(self.lookback, len(scaled) - self.horizon + 1):
            X.append(scaled[i - self.lookback:i])
            y.append(scaled[i + self.horizon - 1, target_idx])
        
        X = np.array(X)
        y = np.array(y)
        
        # Flatten sequences (key difference from LSTM)
        X_flat = X.reshape(X.shape[0], -1)
        
        # Split train/test
        split_idx = int(len(X_flat) * self.train_ratio)
        
        X_train = X_flat[:split_idx]
        X_test = X_flat[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        """
        Train all ensemble models
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            metrics: Dictionary with training metrics
        """
        metrics = {}
        
        print("\nTraining ML Ensemble models...")
        for name, model in self.models.items():
            print(f"Training {name}...", end=" ")
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            metrics[name] = {
                'train_r2': train_score
            }
            print(f"✓ (R² = {train_score:.4f})")
        
        self.is_trained = True
        return metrics
    
    def predict(self, X_test):
        """
        Make ensemble predictions
        
        Args:
            X_test: Test features
            
        Returns:
            predictions: Weighted ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train() first.")
        
        # Get individual predictions
        preds = {}
        for name, model in self.models.items():
            preds[name] = model.predict(X_test)
        
        # Weighted ensemble
        y_pred_ensemble = (
            self.weights['RF'] * preds['RF'] +
            self.weights['GB'] * preds['GB'] +
            self.weights['Ridge'] * preds['Ridge']
        )
        
        return y_pred_ensemble
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate ensemble on test set
        
        Args:
            X_test: Test features
            y_test: Test targets (scaled)
            
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        # Predict
        y_pred_scaled = self.predict(X_test)
        
        # Inverse transform
        y_test_inv = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_inv = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        # Metrics
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'y_true': y_test_inv,
            'y_pred': y_pred_inv
        }
        
        return metrics
    
    def predict_next(self, df_latest):
        """
        Predict next value using latest data
        
        Args:
            df_latest: DataFrame with at least lookback rows
            
        Returns:
            prediction: Predicted price
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet.")
        
        # Create features
        df_features = create_ml_features(df_latest)
        
        # Get last lookback rows
        X_raw = df_features[self.feature_cols].iloc[-self.lookback:].values
        
        # Scale
        X_scaled = self.scaler.transform(X_raw)
        
        # Flatten
        X_flat = X_scaled.reshape(1, -1)
        
        # Predict
        y_scaled = self.predict(X_flat)[0]
        
        # Inverse transform
        y_pred = self.target_scaler.inverse_transform([[y_scaled]])[0][0]
        
        return y_pred


# ==============================================
# TRAINING FUNCTION
# ==============================================

def train_ml_ensemble(df, lookback=30, horizon=7, train_ratio=0.4, verbose=True):
    """
    Train ML Ensemble forecaster
    
    Args:
        df: Input DataFrame
        lookback: Sequence length
        horizon: Forecast horizon
        train_ratio: Train split ratio
        verbose: Print progress
        
    Returns:
        ensemble: Trained MLEnsembleForecaster
        metrics: Evaluation metrics
    """
    # Initialize
    ensemble = MLEnsembleForecaster(
        lookback=lookback,
        horizon=horizon,
        train_ratio=train_ratio
    )
    
    # Prepare data
    if verbose:
        print("Preparing data...")
    
    X_train, X_test, y_train, y_test = ensemble.prepare_data(df)
    
    if verbose:
        print(f"Training data: {X_train.shape}")
        print(f"Test data: {X_test.shape}")
    
    # Train
    train_metrics = ensemble.train(X_train, y_train)
    
    # Evaluate
    if verbose:
        print("\nEvaluating on test set...")
    
    eval_metrics = ensemble.evaluate(X_test, y_test)
    
    if verbose:
        print(f"MAE:  {eval_metrics['mae']:.4f}")
        print(f"RMSE: {eval_metrics['rmse']:.4f}")
    
    # Combine metrics
    metrics = {
        'train': train_metrics,
        'test': eval_metrics
    }
    
    return ensemble, metrics
