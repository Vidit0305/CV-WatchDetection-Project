"""
MobileNetV2 Transfer Learning Training Pipeline for Watch Classification.
Includes data validation, data augmentation, class weighting, two-phase training,
early stopping, and export of models and label mappings.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Suppress debug logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.utils.class_weight import compute_class_weight

from src.preprocessing import TARGET_IMAGE_SIZE


BATCH_SIZE = 16
IMG_HEIGHT, IMG_WIDTH = TARGET_IMAGE_SIZE
CLASSES = ["analog", "digital", "smartwatch"]


def check_dataset_readiness(train_dir: str, val_dir: str) -> Tuple[bool, Dict[str, int]]:
    """
    Checks if train and validation directories contain images for all required classes.
    """
    counts = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_ready = True

    for split_name, split_path in [("train", train_dir), ("validation", val_dir)]:
        if not os.path.isdir(split_path):
            print(f"[ERROR] Directory does not exist: {split_path}")
            return False, {}
        for cls in CLASSES:
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path):
                print(f"[ERROR] Class folder missing: {cls_path}")
                all_ready = False
                continue
            num_imgs = len([
                f for f in os.listdir(cls_path)
                if os.path.splitext(f)[1].lower() in valid_exts
            ])
            counts[f"{split_name}_{cls}"] = num_imgs
            if num_imgs == 0:
                all_ready = False

    return all_ready, counts


def build_augmentation_layer() -> tf.keras.Sequential:
    """Builds a data augmentation pipeline inside the Keras graph."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.12),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomBrightness(0.15),
            layers.RandomContrast(0.15),
        ],
        name="data_augmentation",
    )


def build_mobilenetv2_model(num_classes: int = 3, l2_reg: float = 1e-4) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Constructs the MobileNetV2 transfer learning model with custom classification head.
    Returns: (full_model, base_model)
    """
    # Pre-trained MobileNetV2 base
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze during initial phase

    # Data augmentation block
    augmentation = build_augmentation_layer()

    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="input_image")
    x = augmentation(inputs)
    # MobileNetV2 preprocessing: scales [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="dense_128")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="WatchClassifier_MobileNetV2")
    return model, base_model


def calculate_class_weights(train_ds: tf.data.Dataset, class_names: List[str]) -> Dict[int, float]:
    """Computes balanced class weights from training dataset."""
    y_labels = []
    for _, labels in train_ds:
        y_labels.extend(labels.numpy())
    y_labels = np.array(y_labels)

    classes_unique = np.unique(y_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_unique,
        y=y_labels,
    )
    weight_dict = {int(c): float(w) for c, w in zip(classes_unique, weights)}
    print(f"[INFO] Computed Class Weights: {weight_dict}")
    return weight_dict


def plot_and_save_history(history: tf.keras.callbacks.History, save_path: str = "models/training_history.png") -> None:
    """Plots training and validation accuracy/loss curves and saves to disk."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, acc, label="Training Accuracy", color="#2563EB", lw=2)
    ax1.plot(epochs_range, val_acc, label="Validation Accuracy", color="#10B981", lw=2)
    ax1.set_title("Model Accuracy per Epoch", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0.0, 1.05])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    ax2.plot(epochs_range, loss, label="Training Loss", color="#EF4444", lw=2)
    ax2.plot(epochs_range, val_loss, label="Validation Loss", color="#F59E0B", lw=2)
    ax2.set_title("Model Loss per Epoch", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[INFO] Training curves saved to: {save_path}")


def train(
    dataset_dir: str = "dataset",
    output_model_path: str = "models/watch_classifier_mobilenetv2.keras",
    labels_path: str = "models/class_labels.json",
    epochs_head: int = 15,
    epochs_fine_tune: int = 10,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = 1e-3,
    fine_tune_at: int = 120,
) -> None:
    """Executes the complete transfer-learning training workflow."""
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "validation")

    print("=" * 70)
    print("Watch Type Classifier \u2014 MobileNetV2 Transfer Learning Pipeline")
    print("=" * 70)

    # 1. Dataset Readiness Check
    is_ready, counts = check_dataset_readiness(train_dir, val_dir)
    if not is_ready:
        print("\n[DATASET NOT READY] The training or validation folders are missing images.")
        print(f"Current Counts: {counts}")
        print("\nPlease follow these steps to prepare your dataset:")
        print("  1. Place watch images into dataset/train/, dataset/validation/, dataset/test/.")
        print("  2. Run 'python prepare_dataset.py' to validate integrity and prevent data leakage.")
        print("  3. Once your dataset is populated, rerun 'python train.py'.\n")
        sys.exit(1)

    print(f"\n[INFO] Dataset validated. Class counts:")
    for key, val in counts.items():
        print(f"  {key:<22}: {val} images")

    # 2. Load Datasets using image_dataset_from_directory
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=False,
    )

    class_names = train_ds.class_names
    print(f"[INFO] Classes detected: {class_names}")

    # Compute class weights to handle slight imbalances
    class_weights = calculate_class_weights(train_ds, class_names)

    # Performance optimization: prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # 3. Build Model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    model, base_model = build_mobilenetv2_model(num_classes=len(class_names))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    print("\n[INFO] Model Architecture Summary:")
    model.summary()

    # Callbacks
    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=output_model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    cb_early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    )
    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    # 4. Phase 1: Train Head (Transfer Learning)
    print("\n" + "=" * 70)
    print("PHASE 1: Training Classification Head (Base Model Frozen)")
    print("=" * 70)

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head,
        callbacks=[cb_checkpoint, cb_early_stop, cb_reduce_lr],
        class_weight=class_weights,
    )

    # 5. Phase 2: Fine-Tuning Top Layers
    if epochs_fine_tune > 0:
        print("\n" + "=" * 70)
        print("PHASE 2: Fine-Tuning Top Layers of MobileNetV2")
        print("=" * 70)

        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate * 0.05),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        initial_epoch = len(history_phase1.epoch)
        total_epochs = initial_epoch + epochs_fine_tune

        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epoch,
            epochs=total_epochs,
            callbacks=[cb_checkpoint, cb_early_stop, cb_reduce_lr],
            class_weight=class_weights,
        )
        final_history = history_phase2
    else:
        final_history = history_phase1

    # 6. Save Labels and Training Plot
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"\n[INFO] Saved class labels to: {labels_path}")

    plot_path = os.path.join(os.path.dirname(output_model_path), "training_history.png")
    plot_and_save_history(final_history, plot_path)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Model training complete! Saved to: {output_model_path}")
    print("Next step: Run 'python evaluate.py' to measure test accuracy and generate confusion matrix.")
    print("Then run: 'python main.py' to launch real-time webcam inference.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 watch classifier.")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Root dataset folder")
    parser.add_argument("--output_model", type=str, default="models/watch_classifier_mobilenetv2.keras")
    parser.add_argument("--labels", type=str, default="models/class_labels.json")
    parser.add_argument("--epochs_head", type=int, default=15, help="Epochs for classification head")
    parser.add_argument("--epochs_finetune", type=int, default=10, help="Epochs for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset_dir,
        output_model_path=args.output_model,
        labels_path=args.labels,
        epochs_head=args.epochs_head,
        epochs_fine_tune=args.epochs_finetune,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
