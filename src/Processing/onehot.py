import pandas as pd
from category_encoders import OneHotEncoder
from pathlib import Path
import os
import joblib

def save_ohe_data(X_train, X_valid, X_test, encoder=None, output_dir='data/processed/ohe'):
    """Save one-hot encoded datasets and optionally the encoder"""
    try:
        # Create directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save encoded data
        X_train.to_csv(f'{output_dir}/X_train_ohe.csv', index=False)
        X_valid.to_csv(f'{output_dir}/X_valid_ohe.csv', index=False)
        X_test.to_csv(f'{output_dir}/X_test_ohe.csv', index=False)

        # Save encoder if provided
        if encoder:
            models_dir = 'parameters'
            Path(models_dir).mkdir(parents=True, exist_ok=True)
            joblib.dump(encoder, f'{models_dir}/one_hot_encoder.joblib')
            print(f"✅ One-hot encoder saved to {models_dir}")
        
        print(f"✅ One-hot encoded data saved to {output_dir}")
    except Exception as e:
        print(f"❌ Error saving one-hot encoded data: {e}")
        raise

def apply_onehot_encoding(X_train, X_valid, X_test):
    """Apply one-hot encoding to categorical features"""
    try:
        print("\nApplying one-hot encoding...")
        
        # Initialize encoder
        ohe = OneHotEncoder(cols=['type'], use_cat_names=True)
        
        # Fit and transform on training data
        X_train_ohe = ohe.fit_transform(X_train)
        
        # Transform validation and test data
        X_valid_ohe = ohe.transform(X_valid)
        X_test_ohe = ohe.transform(X_test)
        
        # Print feature information
        print(f"Original features: {X_train.columns.tolist()}")
        print(f"After one-hot encoding: {X_train_ohe.columns.tolist()}")
        print(f"Train shape: {X_train_ohe.shape}")
        
        return X_train_ohe, X_valid_ohe, X_test_ohe, ohe  # Now returning the encoder
        
    except Exception as e:
        print(f"❌ Error in one-hot encoding: {e}")
        raise

if __name__ == "__main__":
    try:
        # Load your split data
        X_train = pd.read_csv('data/processed/split/X_train.csv')
        X_valid = pd.read_csv('data/processed/split/X_val.csv')
        X_test = pd.read_csv('data/processed/split/X_test.csv')
        
        # Apply one-hot encoding (now returns encoder)
        X_train_ohe, X_valid_ohe, X_test_ohe, ohe = apply_onehot_encoding(X_train, X_valid, X_test)
        
        # Save the encoded data and encoder
        save_ohe_data(X_train_ohe, X_valid_ohe, X_test_ohe, encoder=ohe)
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")