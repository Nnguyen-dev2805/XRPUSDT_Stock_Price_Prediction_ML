import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from datetime import timedelta


# train model
def train_layer1_model(X_train, y_train, params=None):

    # chuẩn hóa dữ liệu
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # loại bỏ dữ liệu NaN
    mask = ~np.isnan(X_train_scaled).any(axis=1)
    X_train_clean = X_train_scaled[mask]
    y_train_clean = y_train.iloc[mask] if isinstance(y_train, pd.Series) else y_train[mask]
    
    if params is None:
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

# đánh giá mô hình
def evaluate_model(model, scaler, X_test, y_test):
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

# dự đoán giá cho ngày tiếp theo
def predict_next_day_layer1(model, scaler, latest_features):
    if isinstance(latest_features, pd.DataFrame):
        latest_features = latest_features.values
    
    if len(latest_features.shape) == 1:
        latest_features = latest_features.reshape(1, -1)
    
    latest_features_scaled = scaler.transform(latest_features)
    
    prediction = model.predict(latest_features_scaled)[0]
    
    return prediction


# lưu model
def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

# load lại model
def load_model(filepath):
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    return model

# lấy những feature quan trọng nhất
def get_feature_importance(model, feature_names, top_n=20):
    importances = model.feature_importances_
    
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    feature_imp = feature_imp.sort_values('Importance', ascending=False)
    
    return feature_imp.head(top_n)


def prepare_data_for_training(df_features, feature_columns, target_column='Target_Price', test_size=0.2):

    df_clean = df_features.dropna(subset=feature_columns + [target_column])
    
    X = df_clean[feature_columns]
    y = df_clean[target_column]
    
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def create_prediction_with_confidence(model, scaler, features, n_estimators=None):
    """
    Tạo dự đoán với confidence interval từ RandomForest
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
    
    # 95% confidence interval
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
    Dự đoán giá cuối ngày hôm nay sử dụng Layer 2
    """
    if isinstance(layer2_features, pd.DataFrame):
        layer2_features = layer2_features.values
    
    if len(np.array(layer2_features).shape) == 1:
        layer2_features = np.array(layer2_features).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(layer2_features)
    
    prediction = model.predict(features_scaled)[0]
    
    return prediction

# dữ đoán RF Price
def predict_multi_step_layer1(model, scaler, df_initial, feature_cols, create_features_func, steps=7):
    forecast_results = []
    current_df = df_initial.tail(60).copy()
    
    if not pd.api.types.is_datetime64_any_dtype(current_df['Date']):
        current_df['Date'] = pd.to_datetime(current_df['Date'])
        
    last_date = current_df['Date'].max()
    last_vol = current_df['Vol'].iloc[-1]
    
    for i in range(1, steps + 1):
        df_with_features = create_features_func(current_df)
        df_clean_features = df_with_features[feature_cols].ffill().fillna(0)
        latest_features = df_clean_features.iloc[-1:].values
        
        latest_features_scaled = scaler.transform(latest_features)
        pred_price = model.predict(latest_features_scaled)[0]
        
        next_date = last_date + timedelta(days=i)
        forecast_results.append({
            'Date': next_date,
            'Price': pred_price
        })
        
        new_row = {
            'Date': next_date,
            'Price': pred_price,
            'Open': pred_price,
            'High': pred_price,
            'Low': pred_price,
            'Vol': last_vol
        }
        current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
        
    return pd.DataFrame(forecast_results)
