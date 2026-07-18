import os

import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from preprocessing import preprocess
from config import (
    DATA_PATH,
    TARGET_COLUMN,
    MODEL_PATH,
    REPORT_FOLDER,
)


def evaluate():

    os.makedirs(REPORT_FOLDER, exist_ok=True)

    # Load dataset
    X_train, X_test, y_train, y_test = preprocess(
        DATA_PATH,
        TARGET_COLUMN
    )

    # Load model
    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]


    # Metrics


    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("=" * 50)
    print("Evaluation")
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")


    # Classification Report


    report = classification_report(y_test, y_pred)

    with open(
        os.path.join(REPORT_FOLDER, "classification_report.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(report)


    # Confusion Matrix


    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap="Blues"
    )

    plt.title("Confusion Matrix")

    plt.savefig(
        os.path.join(
            REPORT_FOLDER,
            "confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close()


    # ROC Curve


    RocCurveDisplay.from_predictions(
        y_test,
        y_prob
    )

    plt.title("ROC Curve")

    plt.savefig(
        os.path.join(
            REPORT_FOLDER,
            "roc_curve.png"
        ),
        dpi=300
    )

    plt.close()


    # Feature Importance


    importance = model.feature_importances_

    plt.figure(figsize=(10,6))

    plt.barh(
        X_train.columns,
        importance
    )

    plt.xlabel("Importance")

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            REPORT_FOLDER,
            "feature_importance.png"
        ),
        dpi=300
    )

    plt.close()


    # Metrics Graph


    metrics = [
        accuracy,
        precision,
        recall,
        f1
    ]

    names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ]

    plt.figure(figsize=(6,5))

    plt.bar(names, metrics)

    plt.ylim(0,1)

    for i, v in enumerate(metrics):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")

    plt.title("Model Performance")

    plt.savefig(
        os.path.join(
            REPORT_FOLDER,
            "metrics.png"
        ),
        dpi=300
    )

    plt.close()

    print("\nAll reports saved successfully!")


if __name__ == "__main__":
    evaluate()