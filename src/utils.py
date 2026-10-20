"""
UI, Drawing, and Terminal State Utilities for Watch Classification.
Provides modern HUD graphics, corner brackets, confidence bars, and spam-free logging.
"""

import os
import time
from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np


# Palette definition (BGR format for OpenCV)
class HUDTheme:
    BG_DARK = (24, 24, 28)
    BG_PANEL = (35, 35, 42)
    BORDER_SUBTLE = (70, 70, 80)
    
    TEXT_MAIN = (250, 250, 252)
    TEXT_MUTED = (160, 160, 170)
    
    # Class-specific theme colors
    COLOR_ANALOG = (255, 178, 66)      # Amber / Gold
    COLOR_DIGITAL = (80, 220, 120)     # Neon Emerald / Casio Green
    COLOR_SMARTWATCH = (255, 105, 180) # Electric Violet / Cyan-Pink
    
    COLOR_UNCERTAIN = (60, 140, 255)   # Warm Orange Warning
    COLOR_ACCENT = (255, 200, 40)      # Sky Blue
    COLOR_RETICLE = (200, 200, 200)    # Soft White / Silver


CLASS_COLORS = {
    "analog": HUDTheme.COLOR_ANALOG,
    "digital": HUDTheme.COLOR_DIGITAL,
    "smartwatch": HUDTheme.COLOR_SMARTWATCH,
    "uncertain": HUDTheme.COLOR_UNCERTAIN,
}

CLASS_DISPLAY_NAMES = {
    "analog": "Analog Watch",
    "digital": "Digital Watch",
    "smartwatch": "Smartwatch",
}


def draw_corner_rect(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = HUDTheme.COLOR_RETICLE,
    thickness: int = 2,
    corner_len: int = 25,
) -> None:
    """
    Draw a sleek corner-bracket viewfinder reticle around the target region.
    bbox is (x, y, w, h).
    """
    x, y, w, h = bbox
    x2, y2 = x + w, y + h

    # Top-Left corner
    cv2.line(img, (x, y), (x + corner_len, y), color, thickness)
    cv2.line(img, (x, y), (x, y + corner_len), color, thickness)

    # Top-Right corner
    cv2.line(img, (x2, y), (x2 - corner_len, y), color, thickness)
    cv2.line(img, (x2, y), (x2, y + corner_len), color, thickness)

    # Bottom-Left corner
    cv2.line(img, (x, y2), (x + corner_len, y2), color, thickness)
    cv2.line(img, (x, y2), (x, y2 - corner_len), color, thickness)

    # Bottom-Right corner
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)


def draw_rounded_banner(
    img: np.ndarray,
    rect: Tuple[int, int, int, int],
    bg_color: Tuple[int, int, int],
    alpha: float = 0.85,
) -> None:
    """
    Draw a semi-transparent dark HUD banner over the frame.
    rect is (x, y, w, h).
    """
    x, y, w, h = rect
    h_img, w_img = img.shape[:2]

    # Clip coordinates to frame bounds
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))

    overlay = img[y:y + h, x:x + w].copy()
    colored_rect = np.full(overlay.shape, bg_color, dtype=np.uint8)
    blended = cv2.addWeighted(colored_rect, alpha, overlay, 1.0 - alpha, 0)
    img[y:y + h, x:x + w] = blended


def draw_floating_badge(
    img: np.ndarray,
    text: str,
    color: Tuple[int, int, int],
    center_x: int,
    top_y: int = 35,
    font_scale: float = 0.85,
    thickness: int = 2,
    alpha: float = 0.85,
) -> None:
    """
    Renders an elegant, floating glassmorphic pill badge with accent border.
    Displays ONLY the essential recognized watch category and confidence or uncertainty.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    pad_x = 28
    pad_y = 14
    box_w = text_w + (pad_x * 2)
    box_h = text_h + (pad_y * 2)
    x1 = int(center_x - box_w / 2)
    y1 = int(top_y)
    x2 = x1 + box_w
    y2 = y1 + box_h

    h_img, w_img = img.shape[:2]
    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(1, min(x2, w_img))
    y2 = max(1, min(y2, h_img))

    # Translucent dark pill background
    roi = img[y1:y2, x1:x2].copy()
    bg = np.full(roi.shape, (18, 18, 22), dtype=np.uint8)
    blended = cv2.addWeighted(bg, alpha, roi, 1.0 - alpha, 0)
    img[y1:y2, x1:x2] = blended

    # Sleek accent border
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

    # Centered text
    text_x = x1 + pad_x
    text_y = y1 + pad_y + text_h
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_minimal_hint(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_scale: float = 0.50,
) -> None:
    """Draws a subtle, non-intrusive keyboard helper hint."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x, y), font, font_scale, (170, 170, 180), 1, cv2.LINE_AA)


def draw_meter_bar(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    progress: float,
    color: Tuple[int, int, int],
    bg_color: Tuple[int, int, int] = (50, 50, 60),
) -> None:
    """
    Draw a smooth confidence progress bar.
    progress is in [0.0, 1.0].
    """
    progress = max(0.0, min(1.0, progress))
    # Draw background track
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    # Draw filled portion
    fill_w = int(w * progress)
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
    # Outer thin border
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 100), 1)


class TerminalNotifier:
    """
    Tracks predicted categories across consecutive frames and emits clean,
    single-line terminal messages ONLY when recognition state changes stably.
    Prevents repetitive terminal spam.
    """

    def __init__(self, stable_frames_required: int = 5):
        self.stable_frames_required = stable_frames_required
        self.candidate_class: Optional[str] = None
        self.candidate_count: int = 0
        self.last_announced_class: Optional[str] = None
        self.last_announcement_time: float = 0.0

    def update(
        self,
        predicted_class: Optional[str],
        confidence: float,
        is_uncertain: bool,
    ) -> Optional[str]:
        """
        Updates the internal state and prints a formatted terminal alert
        if a new category is stably confirmed.
        """
        active_class = "uncertain" if (is_uncertain or predicted_class is None) else predicted_class

        if active_class == self.candidate_class:
            self.candidate_count += 1
        else:
            self.candidate_class = active_class
            self.candidate_count = 1

        # Check if candidate has been stable long enough
        if self.candidate_count >= self.stable_frames_required:
            if active_class != self.last_announced_class:
                self.last_announced_class = active_class
                self.last_announcement_time = time.time()
                message = self._format_terminal_message(active_class, confidence)
                print(message, flush=True)
                return message

        return None

    def reset(self) -> None:
        """Reset history state."""
        self.candidate_class = None
        self.candidate_count = 0
        self.last_announced_class = None
        print("\n[INFO] Prediction history reset by user.", flush=True)

    def _format_terminal_message(self, category: str, confidence: float) -> str:
        timestamp = time.strftime("%H:%M:%S")
        if category == "uncertain":
            return f"[{timestamp}] [STATUS] State changed -> UNCERTAIN / NO WATCH CLEARLY VISIBLE"
        
        display_name = CLASS_DISPLAY_NAMES.get(category, category.title())
        conf_pct = confidence * 100.0
        return f"[{timestamp}] [WATCH DETECTED] -> {display_name.upper()} ({conf_pct:.1f}% confidence)"


def compute_dhash(image_gray: np.ndarray, hash_size: int = 8) -> int:
    """
    Computes a 64-bit difference hash (dHash) for an image.
    Used by prepare_dataset.py to detect exact and near-duplicate images across splits.
    """
    resized = cv2.resize(image_gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def hamming_distance(hash1: int, hash2: int) -> int:
    """Computes Hamming distance between two binary hashes."""
    return bin(hash1 ^ hash2).count("1")
