import joblib
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, roc_auc_score, classification_report)
from xgboost  import XGBClassifier
import logging
from datetime import datetime
from dvclive import Live 
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = Path('models/xgb_model.joblib')
DATA_DIR = Path('./data/processed/')
METRICS_PATH = Path('metrics/xgb/evaluation_metrics.json')
CLASS_REPORT_PATH = Path('metrics/xgb/classification_report.txt')

def load_model(model_path: Path):
    """Load a saved model from file"""
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluate model performance and return metrics"""
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Generate classification report
        clf_report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model": str(model.__class__),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            "classification_report": clf_report
        }
        
        return metrics, classification_report(y_test, y_pred)  # Return both dict and string versions
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

def save_metrics(metrics: dict, clf_report: str, metrics_path: Path, report_path: Path):
    """Save evaluation metrics to JSON and classification report to text file"""
    try:
        # Save metrics to JSON
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save classification report to text file
        with open(report_path, 'w') as f:
            f.write(clf_report)
        logger.info(f"Classification report saved to {report_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def main():
    # Load model
    model = load_model(MODEL_PATH)
    
    # Load test data
    X_test = pd.read_csv(DATA_DIR / 'finefeatures/X_test_finalfeatures.csv')
    y_test = pd.read_csv(DATA_DIR / 'split/y_test.csv').squeeze()  # Convert to Series
    
    # Evaluate model
    metrics, clf_report = evaluate_model(model, X_test, y_test)
    
    # Save results
    save_metrics(metrics, clf_report, METRICS_PATH, CLASS_REPORT_PATH)
    
    # Print results to console
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        if k != 'classification_report':  # Don't print the full report dict
            print(f"{k:>20}: {v}")
    
    print("\nClassification Report:")
    print(clf_report)

if __name__ == "__main__":
    main()