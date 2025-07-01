import os
import json
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score

# Create output directory
os.makedirs("tuning", exist_ok=True)

# Define F1 scorer
f1 = make_scorer(f1_score, average='macro')

# Define parameter grid
params = {
    'booster': ['gbtree', 'gblinear', 'dart'],
    'eta': [0.3, 0.1, 0.01],
    'scale_pos_weight': [1, 774, 508, 99]
}

# Grid search setup
gs = GridSearchCV(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    param_grid=params,
    scoring=f1,
    cv=StratifiedKFold(n_splits=5),
    n_jobs=-1,
    verbose=1
)

# Load data
X_params_cs = pd.read_csv('data/processed/finefeatures/X_train_finalfeatures.csv')
y_temp = pd.read_csv('data/processed/split/y_train.csv').squeeze()

# # Fit the model
# gs.fit(X_params_cs, y_temp)

# Save best parameters
best_params = gs.best_params_
with open("tuning/xgb_best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)
print("✅ Best parameters saved to tuning/xgb_best_params.json")

# Save full CV results
cv_results = pd.DataFrame(gs.cv_results_)
cv_results.to_csv("tuning/xgb_gridsearch_results.csv", index=False)
print("📊 Full CV results saved to tuning/xgb_gridsearch_results.csv")
