import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def save_split_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir='data/processed/split'):
    """Save split datasets to files"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save features
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_val.to_csv(f'{output_dir}/X_val.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)

    # Save targets
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False, header=True)
    y_val.to_csv(f'{output_dir}/y_val.csv', index=False, header=True)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False, header=True)

def split_and_save_data(df_path='data/interim/featured_data.csv'):
    try:
        # 1. Load data
        df = pd.read_csv(df_path)
        print(f"✔ Data loaded successfully. Initial shape: {df.shape}")

        # 2. Check target variable values
        print("\nTarget value distribution:")
        print(df['is_fraud'].value_counts(dropna=False))

        # 3. Validate target values (must be 0 or 1)
        if not set(df['is_fraud'].unique()).issubset({0, 1}):
            raise ValueError("❌ 'is_fraud' column contains non-binary values.")

        # 4. Prepare features and target
        X = df.drop(columns=['is_fraud', 'is_flagged_fraud', 'name_orig', 'name_dest',
                             'step_weeks', 'step_days'], errors='ignore')
        y = df['is_fraud']

        if len(X) == 0:
            raise ValueError("❌ No valid data remaining after cleaning.")

        # 5. Split data
        print("\nSplitting data...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42
        )

        # 6. Summary
        print("\n✔ Data splitting completed:")
        print(f"Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%})")
        print(f"Val set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X):.1%})")
        print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.1%})")
        print(f"Fraud ratio - Train: {y_train.mean():.4f}, Val: {y_val.mean():.4f}, Test: {y_test.mean():.4f}")

        # 7. Save data
        save_split_data(X_train, X_val, X_test, y_train, y_val, y_test)
        print(f"\n✅ Data saved to 'data/processed/split'")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    split_and_save_data()
