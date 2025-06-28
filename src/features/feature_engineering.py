import pandas as pd 
import numpy as np 
import inflection
import os

# Read the data safely
df = pd.read_csv('data/external/fraud_detection.csv')

# Changing column names to snake_case
cols_old = df.columns.tolist()
snakecase = lambda x: inflection.underscore(x)
cols_new = list(map(snakecase, cols_old))
df.columns = cols_new

# Add new features
df['step_days'] = df['step'] / 24
df['step_weeks'] = df['step'] / (24 * 7)
df['diff_new_old_balance'] = df['newbalance_orig'] - df['oldbalance_org']
df['diff_new_old_destiny'] = df['newbalance_dest'] - df['oldbalance_dest']
df['name_orig'] = df['name_orig'].apply(lambda i: i[0])
df['name_dest'] = df['name_dest'].apply(lambda i: i[0])

# Save processed data
os.makedirs('data/interim', exist_ok=True)
df.to_csv('data/interim/featured_data.csv', index=False)
