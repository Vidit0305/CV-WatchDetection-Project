"""
Model Evaluation and Metrics Reporting Script.
Evaluates the trained MobileNetV2 watch classifier on the independent test dataset.
Computes test accuracy, precision, recall, F1-score, and saves confusion matrix plot and JSON.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

# Suppress TF debug logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from src.preprocessing import TARGET_IMAGE_SIZE


IMG_HEIGHT, IMG_WIDTH = TARGET_IMAGE_SIZE
CLASSES = ["analog", "digital", "smartwatch"]


def check_test_dataset(test_dir: str) -> Tuple[bool, int]:
    """Checks whether the test directory has images across classes."""
    if not os.path.isdir(test_dir):
        return False, 0
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    total_imgs = 0
    for cls in CLASSES:
        cls_path = os.path.join(test_dir, cls)
        if os.path.isdir(cls_path):
            total_imgs += len([
                f for f in os.listdir(cls_path)
                if os.path.splitext(f)[1].lower() in valid_exts
            ])
    return total_imgs > 0, total_imgs


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str = "models/confusion_matrix.png",
) -> None:
    """Plots and saves a high-resolution styled confusion matrix heatmap."""
    plt.figure(figsize=(7, 6))
    display_labels = [c.replace("_", " ").title() for c in class_names]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title("Watch Classification Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Category", fontsize=12, labelpad=10)
    plt.ylabel("Ground Truth Category", fontsize=12, labelpad=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[INFO] Confusion matrix plot saved to: {save_path}")


def evaluate_model(
    model_path: str = "models/watch_classifier_mobilenetv2.keras",
    labels_path: str = "models/class_labels.json",
    test_dir: str = "dataset/test",
    output_metrics_path: str = "models/test_metrics.json",
) -> Dict:
    """Runs test dataset evaluation and outputs comprehensive metrics."""
    print("=" * 70)
    print("Watch Type Classifier \u2014 Independent Test Set Evaluation")
    print("=" * 70)

    # 1. Model Availability Check
    if not os.path.isfile(model_path):
        alt_path = model_path.replace(".keras", ".h5")
        if os.path.isfile(alt_path):
            model_path = alt_path

    if not os.path.isfile(model_path):
        print(f"\n[ERROR] Trained model file not found at: {model_path}")
        print("Please train the model first by running:")
        print("    python train.py\n")
        sys.exit(1)

    # 2. Test Dataset Check
    has_test_data, total_test_samples = check_test_dataset(test_dir)
    if not has_test_data:
        print(f"\n[ERROR] No test images found in '{test_dir}'.")
        print("Organize test images into:")
        print("    dataset/test/analog/")
        print("    dataset/test/digital/")
        print("    dataset/test/smartwatch/")
        print("\nOr run 'python collect_data.py' / 'python download_samples.py' to acquire test images.\n")
        sys.exit(1)

    # 3. Load Labels
    class_names = CLASSES
    if os.path.isfile(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                class_names = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not load labels from {labels_path}: {e}")

    # 4. Load Model
    print(f"\n[INFO] Loading model from: {model_path} ...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("[INFO] Model loaded successfully.")

    # 5. Load Test Dataset
    print(f"[INFO] Loading test dataset from: {test_dir} ({total_test_samples} total images) ...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        batch_size=16,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=False,
    )

    # 6. Inference across Test Set
    y_true = []
    y_pred = []
    y_pred_probs = []

    for images, labels in test_ds:
        # Preprocessing matches training: MobileNetV2 preprocess
        preds = model(images, training=False).numpy()
        pred_labels = np.argmax(preds, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(pred_labels)
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_probs = np.array(y_pred_probs)

    # 7. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    # 8. Print Results Table
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total Test Samples: {len(y_true)}")
    print(f"Overall Accuracy:   {acc * 100:.2f}%\n")

    print("Detailed Classification Report:")
    print(report_str)

    print("Confusion Matrix:")
    print(f"{'True \\ Pred':<15} | " + " | ".join(f"{c:<10}" for c in class_names))
    print("-" * 55)
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<15} | " + " | ".join(f"{val:<10}" for val in row))
    print("=" * 70)

    # 9. Save Confusion Matrix Plot
    cm_plot_path = os.path.join(os.path.dirname(output_metrics_path), "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_plot_path)

    # 10. Save Metrics to JSON
    metrics_payload = {
        "total_test_samples": int(len(y_true)),
        "accuracy": float(acc),
        "macro_metrics": {
            "precision": float(precision_macro),
            "recall": float(recall_macro),
            "f1_score": float(f1_macro),
        },
        "weighted_metrics": {
            "precision": float(precision_weighted),
            "recall": float(recall_weighted),
            "f1_score": float(f1_weighted),
        },
        "confusion_matrix": cm.tolist(),
        "classes": class_names,
        "per_class_report": report_dict,
    }

    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"[INFO] Test metrics JSON saved to: {output_metrics_path}")
    return metrics_payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate watch classifier on test dataset.")
    parser.add_argument("--model", type=str, default="models/watch_classifier_mobilenetv2.keras")
    parser.add_argument("--labels", type=str, default="models/class_labels.json")
    parser.add_argument("--test_dir", type=str, default="dataset/test")
    parser.add_argument("--output_metrics", type=str, default="models/test_metrics.json")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        labels_path=args.labels,
        test_dir=args.test_dir,
        output_metrics_path=args.output_metrics,
    )


if __name__ == "__main__":
    main()
