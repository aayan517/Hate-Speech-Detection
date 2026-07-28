import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(predictions, test_labels):
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = np.array(test_labels)

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "Hate Speech",
            "Offensive Language",
            "Neither"
        ]
    )

    print(report)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Hate", "Offensive", "Neither"],
        yticklabels=["Hate", "Offensive", "Neither"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - BERT")

    os.makedirs("../results/confusion_matrices", exist_ok=True)
    plt.savefig(
        "../results/confusion_matrices/bert_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    results = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )

    metrics = pd.DataFrame(results).transpose()

    os.makedirs("../results/metrics", exist_ok=True)

    metrics.to_csv(
        "../results/metrics/bert_metrics.csv",
        index=True
    )

    prediction_df = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred
    })

    os.makedirs("../results/tables", exist_ok=True)

    prediction_df.to_csv(
        "../results/tables/bert_predictions.csv",
        index=False
    )

    return metrics