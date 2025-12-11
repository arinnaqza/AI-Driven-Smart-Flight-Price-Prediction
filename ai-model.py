import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import joblib

# Set model constants
RANDOM_STATE = 42
TARGET_COLUMN = 'price'

def load_data():
    """Loads the training and testing datasets using KRW filenames."""
    print("--- 1. LOADING TRAIN/TEST DATA (KRW) ---")
    try:
        # Loading the KRW-specific files
        X_train = pd.read_csv('X_train_krw.csv')
        X_test = pd.read_csv('X_test_krw.csv')
        y_train = pd.read_csv('y_train_krw.csv').squeeze()
        y_test = pd.read_csv('y_test_krw.csv').squeeze()
        print(f"Data loaded successfully. Training samples: {len(X_train)}, Testing samples: {len(X_test)}.")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print("ERROR: KRW Training/Testing CSV files not found. Ensure 'data-analysis.py' was run successfully.")
        return None, None, None, None

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluates a regression model and prints key metrics."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- {model_name} Evaluation (KRW) ---")
    print(f"Mean Absolute Error (MAE): {mae:,.2f} KRW")
    print(f"R-squared Score (R2): {r2:.4f}")
    return mae, r2

def train_linear_regression(X_train, X_test, y_train, y_test):
    """Trains and evaluates the Linear Regression baseline model."""
    print("\n--- 2. BASELINE MODEL (Linear Regression) ---")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    mae, r2 = evaluate_model(lr_model, X_test, y_test, "Linear Regression Baseline")
    return lr_model, mae, r2

def train_initial_xgboost(X_train, X_test, y_train, y_test):
    """Trains and evaluates the initial XGBoost model."""
    print("\n--- 3. INITIAL XGBOOST MODEL ---")
    xgb_params = {
        'objective': 'reg:squarederror', 
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'verbosity': 0, 
        'n_jobs': -1,
        'random_state': RANDOM_STATE
    }
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)
    mae, r2 = evaluate_model(xgb_model, X_test, y_test, "XGBoost Initial")
    return xgb_model, mae, r2

def train_initial_lightgbm(X_train, X_test, y_train, y_test):
    """Trains and evaluates the initial LightGBM model."""
    print("\n--- 4. INITIAL LIGHTGBM MODEL ---")
    lgbm_params = {
        'objective': 'regression_l1',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'verbose': -1, 
        'n_jobs': -1,
        'seed': RANDOM_STATE
    }
    lgbm_model = lgb.LGBMRegressor(**lgbm_params)
    lgbm_model.fit(X_train, y_train)
    mae, r2 = evaluate_model(lgbm_model, X_test, y_test, "LightGBM Initial")
    return lgbm_model, mae, r2

def train_final_champion(X_train, X_test, y_train, y_test):
    """Trains the final model using the optimized parameters (based on best initial model's type)."""
    print("\n--- 5. FINAL TUNED CHAMPION MODEL (XGBoost/LightGBM Optimized) ---")

    # Define the optimized parameters
    final_xgb_params = {
        'objective': 'reg:squarederror', 
        'n_estimators': 1500, 
        'learning_rate': 0.1, 
        'max_depth': 10, 
        'subsample': 0.6, 
        'colsample_bytree': 0.6, 
        'verbosity': 0, 
        'n_jobs': -1,
        'random_state': RANDOM_STATE
    }
    
    # Assuming XGboost perform slightly better than lightbgm
    print("Training FINAL XGBoost model with optimized settings.")
    
    final_xgb_model = xgb.XGBRegressor(**final_xgb_params)
    final_xgb_model.fit(X_train, y_train)
    
    # Analyze Feature Importance for 'Days_Left'
    if 'Days_Left' in X_train.columns:
        importances = final_xgb_model.feature_importances_
        feature_names = X_train.columns
        days_left_importance = pd.Series(importances, index=feature_names).loc['Days_Left']
        print(f"Feature 'Days_Left' Importance in FINAL Model: {days_left_importance:.4f}")

    # Evaluate the Tuned Model
    tuned_mae, tuned_r2 = evaluate_model(final_xgb_model, X_test, y_test, "FINAL Tuned XGBoost Champion")
    
    # Save the final model
    joblib.dump(final_xgb_model, 'xgboost_tuned_champion_model_krw.joblib')
    print("\nFinal Champion Model saved as 'xgboost_tuned_champion_model_krw.joblib'.")

    return final_xgb_model, tuned_mae, tuned_r2


# MAIN EXECUTION

X_train, X_test, y_train, y_test = load_data()

if X_train is not None:
    results = {}
    
    # 1. Baseline
    _, results['LR_MAE'], results['LR_R2'] = train_linear_regression(X_train, X_test, y_train, y_test)
    
    # 2. Initial XGBoost
    _, results['XGB_MAE'], results['XGB_R2'] = train_initial_xgboost(X_train, X_test, y_train, y_test)
    
    # 3. Initial LightGBM
    _, results['LGBM_MAE'], results['LGBM_R2'] = train_initial_lightgbm(X_train, X_test, y_train, y_test)
    
    # 4. Final Tuned Model
    _, final_mae, final_r2 = train_final_champion(X_train, X_test, y_train, y_test)

    # FINAL COMPARISON TABLE
    print("\n\n#####################################################")
    print("### FINAL MODEL COMPARISON (KRW) ####################")
    print("#####################################################")
    print(f"| Model | MAE (KRW) | R2 Score |")
    print("| :------ | :--------- | :-------- |")
    print(f"| Baseline (LR) | {results['LR_MAE']:,.2f} | {results['LR_R2']:.4f} |")
    print(f"| Initial XGBoost | {results['XGB_MAE']:,.2f} | {results['XGB_R2']:.4f} |")
    print(f"| Initial LightGBM | {results['LGBM_MAE']:,.2f} | {results['LGBM_R2']:.4f} |")
    print(f"| FINAL TUNED (XGBoost) | {final_mae:,.2f} | {final_r2:.4f} |")
    print("#####################################################")
    
    print("\nModeling pipeline complete. The final model is saved.")