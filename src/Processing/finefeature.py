import pandas as pd
import os

# Define final selected features
final_columns_selected = [
    'step', 'oldbalance_org', 
    'newbalance_orig', 'newbalance_dest', 
    'diff_new_old_balance', 'diff_new_old_destiny', 
    'type_TRANSFER'
]

X_train=pd.read_csv('data\processed\scaling\X_train_scaled.csv')
X_test=pd.read_csv('data\processed\scaling\X_test_scaled.csv')
X_valid=pd.read_csv('data\processed\scaling\X_valid_scaled.csv')

# Select features from each dataset
X_train_cs = X_train[final_columns_selected]
X_valid_cs = X_valid[final_columns_selected]
# X_temp_cs = X_temp[final_columns_selected]
X_test_cs = X_test[final_columns_selected]
# X_params_cs = X_params[final_columns_selected]

# Create directory if it doesn't exist
os.makedirs('data/processed/finefeatures', exist_ok=True)

# Save datasets to CSV
X_train_cs.to_csv('data/processed/finefeatures/X_train_finalfeatures.csv', index=False)
X_valid_cs.to_csv('data/processed/finefeatures/X_valid_finalfeatures.csv', index=False)
# X_temp_cs.to_csv('data/processed/finefeatures/X_temp_finalfeatures.csv', index=False)
X_test_cs.to_csv('data/processed/finefeatures/X_test_finalfeatures.csv', index=False)
# X_params_cs.to_csv('data/processed/finefeatures/X_params_finalfeatures.csv', index=False)

print("✅ Final features saved in 'data/processed/finefeatures'")