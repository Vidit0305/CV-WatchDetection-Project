"""
Image Preprocessing and Target Extraction Pipeline.
Ensures strict consistency between Training, Evaluation, and Live Webcam Inference.
Uses MobileNetV2 normalization: scale [0, 255] -> [-1, 1].
"""

import os
from typing import Tuple, Optional
import cv2
import numpy as np


# Standard MobileNetV2 input dimensions
TARGET_IMAGE_SIZE: Tuple[int, int] = (224, 224)


def mobilenetv2_preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """
    Applies MobileNetV2 preprocessing: (pixel / 127.5) - 1.0.
    Operates on RGB float32 arrays, producing values in [-1.0, 1.0].
    """
    img_float = img_rgb.astype(np.float32)
    return (img_float / 127.5) - 1.0


def preprocess_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    is_bgr: bool = True,
) -> np.ndarray:
    """
    Prepares a single frame or crop for MobileNetV2 model inference.
    Since the Keras model includes tf.keras.applications.mobilenet_v2.preprocess_input
    in its architecture, inputs are resized RGB float32 images in [0.0, 255.0].
    
    Args:
        frame: OpenCV image array (H, W, C).
        target_size: Tuple (width, height) desired by model. Default (224, 224).
        is_bgr: True if image is in OpenCV BGR format, False if RGB.

    Returns:
        np.ndarray: Batch array with shape (1, height, width, 3), float32 in [0, 255].
    """
    if frame is None or frame.size == 0:
        raise ValueError("Invalid or empty frame provided to preprocess_frame.")

    # Convert BGR to RGB
    if is_bgr:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        rgb = frame.copy()

    # Resize with bilinear interpolation
    if (rgb.shape[1], rgb.shape[0]) != target_size:
        resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        resized = rgb

    # Float32 in range [0.0, 255.0] (the internal Keras model layer handles [-1, 1] scaling)
    img_float = resized.astype(np.float32)

    # Add batch dimension: (1, 224, 224, 3)
    batch = np.expand_dims(img_float, axis=0)
    return batch


def crop_roi(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Extracts Region of Interest from frame using (x, y, w, h) bounding box.
    Clamps bounds to ensure safe slicing.
    """
    h_img, w_img = frame.shape[:2]
    x, y, w, h = bbox

    x1 = max(0, min(x, w_img - 1))
    y1 = max(0, min(y, h_img - 1))
    x2 = max(x1 + 1, min(x + w, w_img))
    y2 = max(y1 + 1, min(y + h, h_img))

    crop = frame[y1:y2, x1:x2]
    return crop


def compute_frame_texture_energy(gray_crop: np.ndarray) -> float:
    """
    Computes Laplacian variance (texture energy / edge response) of a grayscale patch.
    Used as a secondary check to detect completely blank/empty fields (e.g. flat plain walls),
    preventing confident false-positive watch predictions on empty scenes.
    """
    if gray_crop is None or gray_crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def is_image_valid_file(file_path: str) -> bool:
    """
    Validates that an image file exists, is readable by OpenCV, and has valid dimensions.
    """
    if not os.path.isfile(file_path):
        return False
    try:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return False
        h, w = img.shape[:2]
        return h >= 32 and w >= 32
    except Exception:
        return False
