
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "Data" / "heart2.csv"
TARGET_COLUMN = "HeartDisease"


# Train/Test Split
TEST_SIZE = 0.2
RANDOM_STATE = 42


# Data Preprocessing
REMOVE_DUPLICATES = True
FILL_NUMERIC_METHOD = "median"
FILL_CATEGORICAL_METHOD = "mode"
ENCODING_METHOD = "label"
USE_SMOTE = True

# Model
MODEL_PATH = BASE_DIR / "src" / "models" / "xgboost_model.json"

# Report
REPORT_FOLDER = BASE_DIR / "src" / "reports"

# XGBoost Parameters
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.3,
    "reg_alpha": 0.5,
    "reg_lambda": 2,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": RANDOM_STATE,
    "tree_method": "hist",
}