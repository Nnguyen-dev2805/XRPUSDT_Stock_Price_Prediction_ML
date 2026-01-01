"""
Model Utilities Module - Layer 1 (RandomForest)
Training và prediction cho Layer 1
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os


def train_layer1_model(X_train, y_train, params=None):
    """
    Train RandomForest model cho Layer 1 (đúng như notebook)
    
    Args:
        X_train: Training features
        y_train: Training target (giá ngày mai)
        params: Dictionary chứa hyperparameters
    
    Returns:
        tuple: (model, scaler)
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Remove NaN rows
    mask = ~np.isnan(X_train_scaled).any(axis=1)
    X_train_clean = X_train_scaled[mask]
    y_train_clean = y_train.iloc[mask] if isinstance(y_train, pd.Series) else y_train[mask]
    
    if params is None:
        # Exact parameters from notebook
        params = {
            'n_estimators': 500,
            'max_depth': 8,
            'min_samples_leaf': 20,
            'random_state': 42,
            'n_jobs': -1
        }
    
    model = RandomForestRegressor(**params)
    model.fit(X_train_clean, y_train_clean)
    
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test):
    """
    Đánh giá performance của model
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dictionary chứa các metrics
    """
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    
    # Remove NaN rows
    mask = ~np.isnan(X_test_scaled).any(axis=1)
    X_test_clean = X_test_scaled[mask]
    y_test_clean = y_test.iloc[mask] if isinstance(y_test, pd.Series) else y_test[mask]
    
    y_pred = model.predict(X_test_clean)
    
    mae = mean_absolute_error(y_test_clean, y_pred)
    mse = mean_squared_error(y_test_clean, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_clean, y_pred)
    
    # Direction accuracy
    actual_direction = np.sign(np.diff(y_test_clean))
    pred_direction = np.sign(np.diff(y_pred))
    direction_acc = np.mean(actual_direction == pred_direction) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Direction_Accuracy': direction_acc
    }


def predict_next_day_layer1(model, scaler, latest_features):
    """
    Dự đoán giá ngày tiếp theo sử dụng Layer 1
    
    Args:
        model: Trained RandomForest model
        scaler: Fitted scaler
        latest_features: DataFrame hoặc array chứa features của ngày hiện tại
    
    Returns:
        float: Giá dự đoán cho ngày mai
    """
    if isinstance(latest_features, pd.DataFrame):
        latest_features = latest_features.values
    
    if len(latest_features.shape) == 1:
        latest_features = latest_features.reshape(1, -1)
    
    # Scale features
    latest_features_scaled = scaler.transform(latest_features)
    
    prediction = model.predict(latest_features_scaled)[0]
    
    return prediction


def save_model(model, filepath):
    """
    Lưu model vào file
    
    Args:
        model: Model cần lưu
        filepath: Đường dẫn file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load model từ file
    
    Args:
        filepath: Đường dẫn file
    
    Returns:
        Loaded model
    """
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    return model


def get_feature_importance(model, feature_names, top_n=20):
    """
    Lấy feature importance từ RandomForest
    
    Args:
        model: Trained RandomForest model
        feature_names: List tên các features
        top_n: Số lượng top features cần lấy
    
    Returns:
        DataFrame chứa feature importance
    """
    importances = model.feature_importances_
    
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    feature_imp = feature_imp.sort_values('Importance', ascending=False)
    
    return feature_imp.head(top_n)


def prepare_data_for_training(df_features, feature_columns, target_column='Target_Price', test_size=0.2):
    """
    Chuẩn bị dữ liệu cho training
    
    Args:
        df_features: DataFrame chứa tất cả features
        feature_columns: List các columns dùng làm features
        target_column: Tên column target
        test_size: Tỷ lệ test set
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Remove rows with NaN
    df_clean = df_features.dropna(subset=feature_columns + [target_column])
    
    # Split features and target
    X = df_clean[feature_columns]
    y = df_clean[target_column]
    
    # Time series split (không shuffle)
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test


def create_prediction_with_confidence(model, scaler, features, n_estimators=None):
    """
    Tạo dự đoán với confidence interval từ RandomForest
    
    Args:
        model: Trained RandomForest model
        scaler: Fitted scaler
        features: Features để predict
        n_estimators: Số lượng trees (None = all trees)
    
    Returns:
        dict: {
            'prediction': giá dự đoán,
            'std': độ lệch chuẩn,
            'lower_bound': giới hạn dưới (95% CI),
            'upper_bound': giới hạn trên (95% CI)
        }
    """
    if isinstance(features, pd.DataFrame):
        features = features.values
    
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Get predictions from all trees
    tree_predictions = np.array([tree.predict(features_scaled)[0] for tree in model.estimators_])
    
    # Calculate statistics
    prediction = tree_predictions.mean()
    std = tree_predictions.std()
    
    # 95% confidence interval (1.96 * std)
    lower_bound = prediction - 1.96 * std
    upper_bound = prediction + 1.96 * std
    
    return {
        'prediction': prediction,
        'std': std,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


def train_layer2_model(X_train, y_train, alpha=1.0):
    """
    Train Ridge model cho Layer 2 (Stacking)
    
    Args:
        X_train: Training features (Open, Vol, RF_Pred_Today)
        y_train: Training target (Giá đóng cửa thực tế hôm nay)
        alpha: Regularization strength
        
    Returns:
        tuple: (model, scaler)
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Remove NaN rows
    mask = ~np.isnan(X_train_scaled).any(axis=1)
    X_train_clean = X_train_scaled[mask]
    y_train_clean = y_train.iloc[mask] if isinstance(y_train, pd.Series) else y_train[mask]
    
    model = Ridge(alpha=alpha)
    model.fit(X_train_clean, y_train_clean)
    
    return model, scaler


def predict_layer2(model, scaler, layer2_features):
    """
    Dự đoán giá cuối ngày hôm nay sử dụng Layer 2 (giá trong ngày)
    
    Args:
        model: Trained Ridge model
        scaler: Fitted scaler for Layer 2
        layer2_features: Array-like [Open, Vol, RF_Pred_Today]
        
    Returns:
        float: Giá dự đoán cuối ngày
    """
    if isinstance(layer2_features, pd.DataFrame):
        layer2_features = layer2_features.values
    
    if len(np.array(layer2_features).shape) == 1:
        layer2_features = np.array(layer2_features).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(layer2_features)
    
    prediction = model.predict(features_scaled)[0]
    
    return prediction
