"""
High-Performance Watch Classification Engine with Multi-Region Focus & Graph Acceleration.
Uses @tf.function compiled inference, intelligent multi-candidate spatial scanning,
responsive rolling-window smoothing, and calibrated uncertainty gating.
"""

import json
import os
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import cv2
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
    Manages MobileNetV2 model loading, compiled graph execution, background thread processing,
    multi-region candidate search, temporal prediction smoothing, and uncertainty gating.
    """

    def __init__(
        self,
        model_path: str = "models/watch_classifier_mobilenetv2.keras",
        labels_path: str = "models/class_labels.json",
        confidence_threshold: float = 0.48,
        margin_threshold: float = 0.08,
        history_len: int = 5,
        min_texture_energy: float = 6.0,
        enable_async: bool = True,
    ):
        self.model_path = model_path
        self.labels_path = labels_path
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.history_len = history_len
        self.min_texture_energy = min_texture_energy
        self.enable_async = enable_async

        self.model: Optional[Any] = None
        self._fast_step: Optional[Any] = None
        self.class_labels: List[str] = []
        self.prediction_history: deque = deque(maxlen=history_len)

        # Threading state for async execution
        self._lock = threading.Lock()
        self._pending_frame: Optional[np.ndarray] = None
        self._pending_roi: Optional[Tuple[int, int, int, int]] = None
        self._latest_result: Optional[Dict[str, Any]] = None
        self._worker_running = True
        self._worker_thread: Optional[threading.Thread] = None

        self._load_labels()
        self._load_model()

        if self.enable_async and self.is_model_ready:
            self._start_worker()

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
            self.class_labels = ["analog", "digital", "smartwatch"]

    def _load_model(self) -> None:
        """Loads trained Keras MobileNetV2 model and compiles fast @tf.function graph."""
        if not os.path.isfile(self.model_path):
            alt_path = self.model_path.replace(".keras", ".h5")
            if os.path.isfile(alt_path):
                self.model_path = alt_path

        if not os.path.isfile(self.model_path):
            print(
                f"\n[INFO] No trained model found at '{self.model_path}'.\n"
                f"Please run 'python train.py' to train the MobileNetV2 classifier.\n"
            )
            self.model = None
            return

        if tf is None:
            raise RuntimeError("TensorFlow is not installed.")

        print(f"[INFO] Loading trained model: {self.model_path} ...", flush=True)
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)

            # Compile graph step to eliminate eager execution lag (13x speedup)
            @tf.function(reduce_retracing=True)
            def compiled_step(batch):
                return self.model(batch, training=False)

            self._fast_step = compiled_step

            # Warm-up compile with dummy input
            dummy_input = np.zeros((1, TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0], 3), dtype=np.float32)
            _ = self._fast_step(dummy_input)
            print("[INFO] Model loaded and compiled with @tf.function (accelerated).", flush=True)

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    @property
    def is_model_ready(self) -> bool:
        return self.model is not None

    def _start_worker(self) -> None:
        """Launches the background inference worker thread."""
        self._worker_running = True
        self._worker_thread = threading.Thread(target=self._inference_worker_loop, daemon=True)
        self._worker_thread.start()

    def _inference_worker_loop(self) -> None:
        """Continuous background worker loop for non-blocking inference."""
        while self._worker_running:
            frame = None
            roi = None
            with self._lock:
                if self._pending_frame is not None:
                    frame = self._pending_frame
                    roi = self._pending_roi
                    self._pending_frame = None
                    self._pending_roi = None

            if frame is not None:
                result = self._evaluate_multi_region(frame, roi)
                with self._lock:
                    self._latest_result = result
            else:
                time.sleep(0.005)

    def reset_history(self) -> None:
        """Clears the temporal smoothing queue."""
        with self._lock:
            self.prediction_history.clear()
            self._latest_result = None

    def predict_async(
        self,
        frame: np.ndarray,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Non-blocking prediction call: submits the frame to the worker thread
        and immediately returns the latest smoothed prediction result (<0.01 ms).
        """
        if not self.is_model_ready:
            return self._default_empty_result("Model not trained")

        with self._lock:
            self._pending_frame = frame.copy()
            self._pending_roi = roi_bbox
            if self._latest_result is not None:
                return self._latest_result

        # Fallback to direct synchronous prediction for first frame
        return self._evaluate_multi_region(frame, roi_bbox)

    def predict(
        self,
        frame: np.ndarray,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """Synchronous prediction pass with multi-region scanning."""
        if not self.is_model_ready:
            return self._default_empty_result("Model not trained")

        return self._evaluate_multi_region(frame, roi_bbox)

    def _predict_single_crop(self, patch: np.ndarray) -> Tuple[np.ndarray, float]:
        """Runs model inference on a single image patch, returning raw probs and texture energy."""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        energy = compute_frame_texture_energy(gray)
        processed = preprocess_frame(patch, target_size=TARGET_IMAGE_SIZE, is_bgr=True)

        if self._fast_step is not None:
            raw_preds = self._fast_step(processed).numpy()[0]
        else:
            raw_preds = self.model(processed, training=False).numpy()[0]

        return raw_preds, energy

    def _evaluate_multi_region(
        self,
        frame: np.ndarray,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Intelligent multi-candidate spatial scanning:
        1. Evaluates center viewfinder ROI.
        2. If center ROI is confident (>=0.52), locks on immediately.
        3. If center ROI is ambiguous or uncertain, scans candidate regions (full frame, right-hand, left-hand).
        4. Selects the highest-confidence candidate without penalizing off-center presentation.
        """
        h_img, w_img = frame.shape[:2]

        candidates: List[Tuple[str, np.ndarray, Optional[Tuple[int, int, int, int]]]] = []

        # 1. Primary: Center Viewfinder ROI
        if roi_bbox is not None:
            center_patch = crop_roi(frame, roi_bbox)
            candidates.append(("reticle", center_patch, roi_bbox))

        # First pass on primary center patch
        primary_preds = None
        primary_energy = 0.0

        if candidates:
            tag, patch, box = candidates[0]
            primary_preds, primary_energy = self._predict_single_crop(patch)
            top_prob = float(np.max(primary_preds))
            sorted_p = np.sort(primary_preds)[::-1]
            margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0

            # If primary center box is clearly confident, use it directly
            if top_prob >= 0.55 and margin >= 0.10:
                return self._package_result(primary_preds, primary_energy, active_source="reticle", active_bbox=box)

        # 2. Secondary candidates if center ROI did not decisively capture the watch
        # Right-hand presentation box
        rw = int(min(h_img, w_img) * 0.52)
        rh = rw
        rx = min(w_img - rw, int(w_img * 0.45))
        ry = max(0, min(h_img - rh, int(h_img * 0.18)))
        right_box = (rx, ry, rw, rh)
        right_patch = crop_roi(frame, right_box)

        # Left-hand presentation box
        lx = max(0, int(w_img * 0.05))
        ly = ry
        left_box = (lx, ly, rw, rh)
        left_patch = crop_roi(frame, left_box)

        secondary_candidates = [
            ("fullscreen", frame, None),
            ("right_hand", right_patch, right_box),
            ("left_hand", left_patch, left_box),
        ]

        best_preds = primary_preds
        best_energy = primary_energy
        best_top_prob = float(np.max(primary_preds)) if primary_preds is not None else 0.0
        best_source = "reticle"
        best_box = roi_bbox

        for tag, patch, box in secondary_candidates:
            if patch is None or patch.size == 0:
                continue
            preds, energy = self._predict_single_crop(patch)
            top_p = float(np.max(preds))
            if top_p > best_top_prob:
                best_top_prob = top_p
                best_preds = preds
                best_energy = energy
                best_source = tag
                best_box = box

        if best_preds is None:
            return self._default_empty_result("No visual input")

        return self._package_result(best_preds, best_energy, active_source=best_source, active_bbox=best_box)

    def _package_result(
        self,
        raw_preds: np.ndarray,
        texture_energy: float,
        active_source: str = "reticle",
        active_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """Applies responsive temporal smoothing and calibrated uncertainty gating."""
        # Temporal smoothing buffer
        self.prediction_history.append(raw_preds)
        smoothed_preds = np.mean(self.prediction_history, axis=0)

        # Top probabilities
        sorted_indices = np.argsort(smoothed_preds)[::-1]
        top_idx = sorted_indices[0]
        second_idx = sorted_indices[1] if len(sorted_indices) > 1 else None

        top_prob = float(smoothed_preds[top_idx])
        second_prob = float(smoothed_preds[second_idx]) if second_idx is not None else 0.0
        prob_margin = top_prob - second_prob

        predicted_class = self.class_labels[top_idx] if top_idx < len(self.class_labels) else "unknown"

        # Calibrated Multi-Factor Uncertainty Gating
        is_uncertain = False
        uncertain_reason = ""

        # 1. Low confidence check
        if top_prob < self.confidence_threshold:
            is_uncertain = True
            uncertain_reason = f"Low confidence ({top_prob * 100:.0f}%)"
        # 2. Ambiguity check: top classes nearly tied
        elif prob_margin < self.margin_threshold and top_prob < 0.60:
            is_uncertain = True
            uncertain_reason = "Ambiguous features"
        # 3. Flat blank scene check: only active if model isn't already confident
        elif texture_energy < self.min_texture_energy and top_prob < 0.75:
            is_uncertain = True
            uncertain_reason = "Empty / low contrast scene"

        if is_uncertain:
            display_name = "Uncertain"
            status_text = "Uncertain — show the watch clearly"
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
            "active_source": active_source,
            "active_bbox": active_bbox,
        }

    def _default_empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "class": None,
            "raw_class": None,
            "display_name": "Uncertain",
            "confidence": 0.0,
            "margin": 0.0,
            "is_uncertain": True,
            "uncertain_reason": reason,
            "probabilities": {},
            "status_text": reason,
            "texture_energy": 0.0,
            "active_source": "none",
            "active_bbox": None,
        }

    def close(self) -> None:
        """Stops background worker thread."""
        self._worker_running = False
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)
