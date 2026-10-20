"""
Inference Engine and Temporal Smoothing for Watch Classification.
Loads trained MobileNetV2 model and performs smoothed, jitter-free predictions
with multi-stage uncertainty detection.
"""

import json
import os
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Suppress verbose TensorFlow logs during runtime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import tensorflow as tf
except ImportError:
    tf = None

from src.preprocessing import (
    preprocess_frame,
    crop_roi,
    compute_frame_texture_energy,
    TARGET_IMAGE_SIZE,
)
from src.utils import CLASS_DISPLAY_NAMES


class WatchClassifier:
    """
    Manages model loading, live frame classification, temporal smoothing,
    and uncertainty determination.
    """

    def __init__(
        self,
        model_path: str = "models/watch_classifier_mobilenetv2.keras",
        labels_path: str = "models/class_labels.json",
        confidence_threshold: float = 0.65,
        margin_threshold: float = 0.15,
        history_len: int = 10,
        min_texture_energy: float = 30.0,
    ):
        self.model_path = model_path
        self.labels_path = labels_path
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.history_len = history_len
        self.min_texture_energy = min_texture_energy

        self.model: Optional[Any] = None
        self.class_labels: List[str] = []
        self.prediction_history: deque = deque(maxlen=history_len)

        self._load_labels()
        self._load_model()

    def _load_labels(self) -> None:
        """Load class labels from JSON if present, else fallback to canonical order."""
        if os.path.isfile(self.labels_path):
            try:
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.class_labels = data
                    elif isinstance(data, dict) and "classes" in data:
                        self.class_labels = data["classes"]
            except Exception as e:
                print(f"[WARN] Error reading labels from {self.labels_path}: {e}")

        if not self.class_labels:
            # Canonical three classes
            self.class_labels = ["analog", "digital", "smartwatch"]

    def _load_model(self) -> None:
        """Loads trained Keras MobileNetV2 model weights."""
        if not os.path.isfile(self.model_path):
            # Check fallback to .h5 if .keras not found
            alt_path = self.model_path.replace(".keras", ".h5")
            if os.path.isfile(alt_path):
                self.model_path = alt_path

        if not os.path.isfile(self.model_path):
            print(
                f"\n[INFO] No trained model found at '{self.model_path}'.\n"
                f"Please run 'python train.py' to train the MobileNetV2 classifier on your dataset.\n"
            )
            self.model = None
            return

        if tf is None:
            raise RuntimeError(
                "TensorFlow is not installed in the active environment. "
                "Please run 'pip install -r requirements.txt'."
            )

        print(f"[INFO] Loading trained model from: {self.model_path} ...", flush=True)
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print("[INFO] Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    @property
    def is_model_ready(self) -> bool:
        return self.model is not None

    def reset_history(self) -> None:
        """Clears the temporal smoothing queue."""
        self.prediction_history.clear()

    def predict(
        self,
        frame: np.ndarray,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Runs inference on the provided frame or cropped ROI.
        
        Args:
            frame: Raw BGR frame from OpenCV webcam.
            roi_bbox: Optional (x, y, w, h) bounding box. If provided, classification
                      is performed specifically on this target patch.

        Returns:
            dict containing predicted class, confidence, uncertainty flag, and smoothed probs.
        """
        if not self.is_model_ready:
            return {
                "class": None,
                "display_name": "Model Not Loaded",
                "confidence": 0.0,
                "is_uncertain": True,
                "probabilities": {},
                "status_text": "Model not trained. Run python train.py first.",
                "reason": "missing_model",
                "texture_energy": 0.0,
            }

        # Crop ROI or use full frame
        target_patch = crop_roi(frame, roi_bbox) if roi_bbox is not None else frame

        # Check scene/target texture energy to prevent false positives on plain blank walls
        gray_patch = cv2.cvtColor(target_patch, cv2.COLOR_BGR2GRAY)
        texture_energy = compute_frame_texture_energy(gray_patch)

        # Preprocess frame matching MobileNetV2 training spec
        processed = preprocess_frame(target_patch, target_size=TARGET_IMAGE_SIZE, is_bgr=True)

        # Forward pass
        raw_preds = self.model(processed, training=False).numpy()[0]

        # Temporal smoothing: maintain rolling window of probabilities
        self.prediction_history.append(raw_preds)
        smoothed_preds = np.mean(self.prediction_history, axis=0)

        # Evaluate top predictions
        sorted_indices = np.argsort(smoothed_preds)[::-1]
        top_idx = sorted_indices[0]
        second_idx = sorted_indices[1] if len(sorted_indices) > 1 else None

        top_prob = float(smoothed_preds[top_idx])
        second_prob = float(smoothed_preds[second_idx]) if second_idx is not None else 0.0
        prob_margin = top_prob - second_prob

        predicted_class = self.class_labels[top_idx] if top_idx < len(self.class_labels) else "unknown"

        # Multi-factor uncertainty assessment:
        # 1. Texture energy below minimum -> camera looking at flat background / empty space
        # 2. Top probability below confidence threshold
        # 3. Margin between top-1 and top-2 too small (ambiguous classification)
        is_uncertain = False
        uncertain_reason = ""

        if texture_energy < self.min_texture_energy:
            is_uncertain = True
            uncertain_reason = "Empty / low contrast scene"
        elif top_prob < self.confidence_threshold:
            is_uncertain = True
            uncertain_reason = f"Low confidence ({top_prob * 100:.1f}% < {self.confidence_threshold * 100:.0f}%)"
        elif prob_margin < self.margin_threshold:
            is_uncertain = True
            uncertain_reason = f"Ambiguous categories (margin {prob_margin * 100:.1f}%)"

        # Format display text
        if is_uncertain:
            display_name = "Uncertain"
            status_text = "Uncertain \u2014 show the watch clearly"
        else:
            display_name = CLASS_DISPLAY_NAMES.get(predicted_class, predicted_class.title())
            status_text = f"{display_name} ({top_prob * 100:.1f}%)"

        probs_dict = {
            self.class_labels[i]: float(smoothed_preds[i])
            for i in range(len(self.class_labels))
        }

        return {
            "class": predicted_class if not is_uncertain else None,
            "raw_class": predicted_class,
            "display_name": display_name,
            "confidence": top_prob,
            "margin": prob_margin,
            "is_uncertain": is_uncertain,
            "uncertain_reason": uncertain_reason,
            "probabilities": probs_dict,
            "status_text": status_text,
            "texture_energy": texture_energy,
        }
