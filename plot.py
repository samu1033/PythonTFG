import math

import numpy as np
import matplotlib.pyplot as plt
from anomalib.visualization.image import visualize_image_item
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryConfusionMatrix,
)


def plot_metrics(model_name, scores, pred_labels, gt_labels):
    auroc_val     = BinaryAUROC()(scores, gt_labels).item()
    f1_val        = BinaryF1Score()(pred_labels, gt_labels).item()
    precision_val = BinaryPrecision()(pred_labels, gt_labels).item()
    recall_val    = BinaryRecall()(pred_labels, gt_labels).item()

    matrix = BinaryConfusionMatrix()
    matrix.update(pred_labels, gt_labels)
    tn, fp, fn, tp = matrix.compute().flatten().tolist()

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(f"Model: {model_name}", fontsize=14, fontweight="bold", y=1.01)

    ax = fig.add_axes([0.05, 0.05, 0.5, 0.9])
    data = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(data, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, data[i, j], ha="center", va="center",
                    color="white" if data[i, j] > data.max() / 2 else "black",
                    fontsize=14, fontweight="bold")

    labels_cm = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i + 0.3, labels_cm[i][j], ha="center", va="center",
                    color="white" if data[i, j] > data.max() / 2 else "black",
                    fontsize=10)

    plt.colorbar(im, ax=ax)

    ax_metrics = fig.add_axes([0.62, 0.15, 0.35, 0.7])
    ax_metrics.axis("off")
    ax_metrics.set_title("Metrics", fontsize=12, fontweight="bold", pad=10)

    for idx, (name, val) in enumerate([
        ("AUROC",     auroc_val),
        ("F1 Score",  f1_val),
        ("Precision", precision_val),
        ("Recall",    recall_val),
    ]):
        y_pos = 0.82 - idx * 0.22
        ax_metrics.text(0.5, y_pos,        name,              ha="center", va="center", fontsize=11, color="gray")
        ax_metrics.text(0.5, y_pos - 0.10, f"{val*100:.2f}%", ha="center", va="center", fontsize=14, fontweight="bold", color="#1f77b4")

    plt.show()


def plot_errors(model_name: str, items: list) -> None:
    false_negatives = [item for item in items if item.gt_label.item() == 1 and item.pred_label.item() == 0]
    false_positives = [item for item in items if item.gt_label.item() == 0 and item.pred_label.item() == 1]

    false_negatives.sort(key=lambda x: x.pred_score.item())
    false_positives.sort(key=lambda x: x.pred_score.item(), reverse=True)

    total_anomalies = sum(item.gt_label.item() == 1 for item in items)
    total_normals   = sum(item.gt_label.item() == 0 for item in items)

    print(f"Model: {model_name}")
    print(f"Falsos Negativos : {len(false_negatives)} / {total_anomalies} anomalias reales  (FNR: {len(false_negatives)/total_anomalies:.2%})")
    print(f"Falsos Positivos : {len(false_positives)} / {total_normals} imagenes normales (FPR: {len(false_positives)/total_normals:.2%})")

    def _visualize_group(group, title):
        n = len(group)
        if n == 0:
            print(f"No hay {title}")
            return
        cols = 4
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
        axes = axes.flatten()
        for i, item in enumerate(group):
            vis = visualize_image_item(
                item,
                fields=["image", "anomaly_map"],
                fields_config={"anomaly_map": {"normalize": False}},
            )
            axes[i].imshow(vis)
            axes[i].set_title(f"Score: {item.pred_score.item():.4f}", fontsize=8)
            axes[i].axis("off")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"{model_name} — {title}: {n}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    _visualize_group(false_negatives, "Falsos Negativos (gt=1, pred=0)")
    _visualize_group(false_positives, "Falsos Positivos (gt=0, pred=1)")
