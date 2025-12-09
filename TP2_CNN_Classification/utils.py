import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def evaluate_and_visualize(
    model,
    loader,
    device,
    class_names,
    save_csv=False,
    csv_path="./predictions.csv",
    cm_path="./confusion_matrix.png"
):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n* Test Accuracy: {acc * 100:.2f}%\n")

    # classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True
    )

    print("* Classification Report:\n")

    print(f"{'Class':22s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}")
    for cls in class_names:
        cls_report = report[cls]
        print(f"{cls:22s} "
              f"{cls_report['precision']:10.2f} "
              f"{cls_report['recall']:10.2f} "
              f"{cls_report['f1-score']:10.2f} "
              f"{int(cls_report['support']):10d}")

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.show()


    # csv 저장
    if save_csv:
        df = pd.DataFrame({
            "true_label": all_labels,
            "pred_label": all_preds,
        })
        df.to_csv(csv_path, index=False)
