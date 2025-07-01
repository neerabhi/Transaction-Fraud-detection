import pandas as pd
import numpy as np
from sklearn.ensemble     import RandomForestClassifier
from pathlib import Path
import joblib  # better than pickle for sklearn models
from dvclive import live
# Load data - using raw strings or forward slashes for paths
data_dir = Path('./data/processed/')
X_train = pd.read_csv(data_dir / 'finefeatures/X_train_finalfeatures.csv')
y_train = pd.read_csv(data_dir / 'split/y_train.csv')

# Ensure y_train is 1D array
y_train = y_train.values.ravel()

# Train dummy model
rf = RandomForestClassifier(class_weight='balanced')
rf.fit(X_train, y_train)

# Save model - create models directory if it doesn't exist
model_dir = Path('./models')
model_dir.mkdir(exist_ok=True)

# Save using joblib
joblib.dump(rf, model_dir / 'rf_model.joblib')

# Alternatively using pickle (fixed variable name)
import pickle
# pickle.dump(dummy, open(model_dir / 'dummy_model.pkl', 'wb'))