"""
Interactive Webcam Dataset Collection Tool for Watch Classification.
Allows the user to easily capture and organize training, validation, and test images
for Analog, Digital, and Smartwatch categories directly from their webcam.
"""

import argparse
import datetime
import os
import random
import time
from typing import Tuple
import cv2
import numpy as np

from src.utils import HUDTheme, draw_corner_rect, draw_rounded_banner


CLASSES = ["analog", "digital", "smartwatch"]
SPLITS = ["train", "validation", "test"]
SPLIT_PROBABILITIES = [0.70, 0.15, 0.15]  # 70% train, 15% val, 15% test


def assign_split() -> str:
    """Randomly assigns an image to train, validation, or test based on distribution."""
    return random.choices(SPLITS, weights=SPLIT_PROBABILITIES, k=1)[0]


def run_data_collector(
    camera_idx: int = 0,
    dataset_dir: str = "dataset",
) -> None:
    """Launches webcam capture studio for collecting watch images."""
    print("=" * 70)
    print("Watch Dataset Collection Studio")
    print("=" * 70)
    print("Controls:")
    print("  [1] : Capture ANALOG watch image")
    print("  [2] : Capture DIGITAL watch image")
    print("  [3] : Capture SMARTWATCH image")
    print("  [B] : Toggle Burst Mode (continuous capture every 0.8s)")
    print("  [Q] : Quit Studio")
    print("=" * 70)

    # Ensure directories exist
    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(dataset_dir, split, cls), exist_ok=True)

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {camera_idx}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    burst_mode = False
    burst_class = "analog"
    last_burst_time = 0.0
    burst_interval = 0.8

    window_name = "Watch Dataset Collector Studio"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 600)

    capture_counts = {
        cls: sum(len(os.listdir(os.path.join(dataset_dir, s, cls))) for s in SPLITS)
        for cls in CLASSES
    }

    last_toast = ""
    toast_timer = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            box_size = int(min(h, w) * 0.55)
            bx = (w - box_size) // 2
            by = (h - box_size) // 2
            roi_bbox = (bx, by, box_size, box_size)

            # Viewfinder
            draw_corner_rect(frame, roi_bbox, color=(80, 220, 120), thickness=3, corner_len=28)

            # Header
            draw_rounded_banner(frame, (0, 0, w, 70), HUDTheme.BG_DARK, alpha=0.88)
            cv2.putText(
                frame,
                "DATASET COLLECTION STUDIO",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                HUDTheme.TEXT_MAIN,
                2,
                cv2.LINE_AA,
            )
            stats_str = f"Counts -> Analog: {capture_counts['analog']} | Digital: {capture_counts['digital']} | Smartwatch: {capture_counts['smartwatch']}"
            cv2.putText(
                frame,
                stats_str,
                (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                HUDTheme.COLOR_ACCENT,
                1,
                cv2.LINE_AA,
            )

            # Footer
            footer_y = h - 75
            draw_rounded_banner(frame, (0, footer_y, w, 75), HUDTheme.BG_DARK, alpha=0.90)
            instructions = "[1] Analog   [2] Digital   [3] Smartwatch   [B] Burst Mode   [Q] Quit"
            cv2.putText(
                frame,
                instructions,
                (25, footer_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )

            # Toast notification
            if time.time() - toast_timer < 2.0:
                cv2.putText(
                    frame,
                    last_toast,
                    (bx, by - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 180),
                    2,
                    cv2.LINE_AA,
                )

            # Burst mode capture
            if burst_mode and (time.time() - last_burst_time >= burst_interval):
                last_burst_time = time.time()
                crop = frame[by:by + box_size, bx:bx + box_size].copy()
                split = assign_split()
                fname = f"{burst_class}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:18]}.jpg"
                save_path = os.path.join(dataset_dir, split, burst_class, fname)
                cv2.imwrite(save_path, crop)
                capture_counts[burst_class] += 1
                last_toast = f"Burst Saved [{burst_class.upper()}] -> {split}"
                toast_timer = time.time()

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord("q"), ord("Q"), 27]:
                break
            elif key in [ord("1"), ord("2"), ord("3")]:
                target_cls = "analog" if key == ord("1") else ("digital" if key == ord("2") else "smartwatch")
                crop = frame[by:by + box_size, bx:bx + box_size].copy()
                split = assign_split()
                fname = f"{target_cls}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:18]}.jpg"
                save_path = os.path.join(dataset_dir, split, target_cls, fname)
                cv2.imwrite(save_path, crop)
                capture_counts[target_cls] += 1
                last_toast = f"Saved: {target_cls.upper()} -> {split}/{fname}"
                toast_timer = time.time()
                print(f"[CAPTURED] Saved {target_cls} to {save_path}", flush=True)
            elif key in [ord("b"), ord("B")]:
                burst_mode = not burst_mode
                last_burst_time = time.time()
                print(f"[INFO] Burst mode: {'ACTIVE' if burst_mode else 'INACTIVE'}", flush=True)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Dataset Collection finished.")
        print(f"Final Counts: {capture_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch dataset collection tool.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Dataset root directory")
    args = parser.parse_args()
    run_data_collector(camera_idx=args.camera, dataset_dir=args.dataset_dir)
