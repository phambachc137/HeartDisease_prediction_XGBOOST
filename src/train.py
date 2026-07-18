

import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier

from preprocessing import preprocess
from config import (
    DATA_PATH,
    TARGET_COLUMN,
    MODEL_PATH,
    XGB_PARAMS,
)


def train_model():
    print("=" * 60)
    print("Loading and preprocessing data...")
    print("=" * 60)

    X_train, X_test, y_train, y_test = preprocess(
        DATA_PATH,
        TARGET_COLUMN,
    )

    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    print("\nInitializing XGBoost model...")

    model = XGBClassifier(**XGB_PARAMS)

    print("Training model...\n")

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed!")


# Evaluate

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    print("\n========== Model Performance ==========")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")


# Save model


    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True,
    )

    model.save_model(MODEL_PATH)

    print("\nModel saved successfully!")
    print(f"Model path: {MODEL_PATH}")

    return (
        model,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    train_model()