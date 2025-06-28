import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import os
import joblib

# Define numerical columns
NUM_COLUMNS = [
    'amount', 'oldbalance_org', 'newbalance_orig',
    'oldbalance_dest', 'newbalance_dest',
    'diff_new_old_balance', 'diff_new_old_destiny'
]

def save_scaled_data(X_train, X_valid, X_test, scaler, output_dir='data/processed/scaling'):
    """Save scaled datasets and scaler"""
    try:
        # Create directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save scaled features
        X_train.to_csv(f'{output_dir}/X_train_scaled.csv', index=False)
        X_valid.to_csv(f'{output_dir}/X_valid_scaled.csv', index=False)
        X_test.to_csv(f'{output_dir}/X_test_scaled.csv', index=False)
        
        # Save scaler object
        joblib.dump(scaler, f'{output_dir}/minmax_scaler.joblib')
        
        print(f"✅ Scaled data and scaler saved to {output_dir}")
    except Exception as e:
        print(f"❌ Error saving scaled data: {e}")
        raise

def scale_numerical_features(X_train, X_valid, X_test):
    """Scale numerical features using MinMaxScaler"""
    try:
        # Initialize scaler
        mm = MinMaxScaler()
        
        # Validate numerical columns exist
        missing_cols = [col for col in NUM_COLUMNS if col not in X_train.columns]
        if missing_cols:
            raise ValueError(f"Missing numerical columns: {missing_cols}")
        
        print("\nScaling numerical features...")
        
        # Fit on train and transform all sets
        X_train[NUM_COLUMNS] = mm.fit_transform(X_train[NUM_COLUMNS])
        X_valid[NUM_COLUMNS] = mm.transform(X_valid[NUM_COLUMNS])
        X_test[NUM_COLUMNS] = mm.transform(X_test[NUM_COLUMNS])
        
        # Print scaling summary
        print(f"Scaled columns: {NUM_COLUMNS}")
        print(f"Train set range after scaling:")
        print(X_train[NUM_COLUMNS].agg(['min', 'max']))
        
        return X_train, X_valid, X_test, mm
        
    except Exception as e:
        print(f"❌ Error in feature scaling: {e}")
        raise

if __name__ == "__main__":
    try:
        # Load your data (adjust paths as needed)
        X_train = pd.read_csv('data/processed/ohe/X_train_ohe.csv')
        X_valid = pd.read_csv('data/processed/ohe/X_valid_ohe.csv')
        X_test = pd.read_csv('data/processed/ohe/X_test_ohe.csv')
        
        print("✔ Successfully loaded data for scaling")
        
        # Scale numerical features
        X_train_scaled, X_valid_scaled, X_test_scaled, scaler = scale_numerical_features(
            X_train.copy(), X_valid.copy(), X_test.copy()
        )
        
        # Save scaled data and scaler
        save_scaled_data(X_train_scaled, X_valid_scaled, X_test_scaled, scaler)
        
    except Exception as e:
        print(f"\n❌ Scaling failed: {e}")