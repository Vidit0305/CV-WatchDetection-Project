"""
Real-Time Watch Type Classification via Webcam and OpenCV.
Features true edge-to-edge 100% Fullscreen display (zero white margins/letterbox),
mirrored camera view (left-right corrected), ultra-fast async inference (no lag),
and a clean minimalist HUD showing only essential recognition details.
"""

import argparse
import datetime
import os
import subprocess
import sys
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from src.inference import WatchClassifier
from src.utils import (
    CLASS_COLORS,
    CLASS_DISPLAY_NAMES,
    HUDTheme,
    TerminalNotifier,
    draw_corner_rect,
    draw_floating_badge,
    draw_minimal_hint,
)


def get_screen_resolution() -> Tuple[int, int]:
    """Detects primary monitor display resolution via xrandr, falling back to 1920x1080."""
    try:
        out = subprocess.check_output("xrandr | grep '*'", shell=True, timeout=2).decode()
        res = out.strip().split()[0].split("x")
        return int(res[0]), int(res[1])
    except Exception:
        return 1920, 1080


def fit_frame_to_screen(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Center-crops camera frame to match target aspect ratio (e.g. 16:9) and scales to fill
    the entire screen edge-to-edge without distortion or blank/white canvas borders.
    """
    h_in, w_in = frame.shape[:2]
    target_aspect = target_w / float(target_h)
    in_aspect = w_in / float(h_in)

    if in_aspect > target_aspect:
        # Input is wider than screen -> crop width
        crop_w = int(h_in * target_aspect)
        start_x = (w_in - crop_w) // 2
        cropped = frame[:, start_x:start_x + crop_w]
    else:
        # Input is taller than screen (e.g. 4:3 input on 16:9 monitor) -> crop height
        crop_h = int(w_in / target_aspect)
        start_y = (h_in - crop_h) // 2
        cropped = frame[start_y:start_y + crop_h, :]

    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized


def find_available_camera(preferred_index: int = 0, max_tested: int = 4) -> Tuple[Optional[cv2.VideoCapture], int]:
    """Attempts to open preferred camera index, then probes alternative indices."""
    indices_to_try = [preferred_index] + [i for i in range(max_tested) if i != preferred_index]
    for idx in indices_to_try:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                return cap, idx
            cap.release()
    return None, -1


def run_webcam(
    camera_index: int = 0,
    model_path: str = "models/watch_classifier_mobilenetv2.keras",
    labels_path: str = "models/class_labels.json",
    confidence_threshold: float = 0.65,
    margin_threshold: float = 0.15,
    history_len: int = 8,
    min_texture_energy: float = 30.0,
    captures_dir: str = "captures",
    window_name: str = "Watch Classifier",
) -> None:
    """Main live webcam inference loop in true edge-to-edge fullscreen mode."""
    print("=" * 70)
    print("Real-Time Watch Type Classification \u2014 Fullscreen Mirror Mode")
    print("=" * 70)
    print("Controls:")
    print("  Q / ESC : Quit application")
    print("  F       : Toggle Fullscreen on/off")
    print("  S       : Save snapshot to captures/")
    print("  R       : Reset temporal prediction smoothing")
    print("  T       : Toggle Viewfinder Reticle on/off")
    print("=" * 70)

    os.makedirs(captures_dir, exist_ok=True)

    # 1. Detect Screen Resolution for True Fullscreen Scaling
    screen_w, screen_h = get_screen_resolution()
    print(f"[INFO] Detected screen resolution: {screen_w}x{screen_h}")

    # 2. Initialize High-Performance Async Watch Classifier
    classifier = WatchClassifier(
        model_path=model_path,
        labels_path=labels_path,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        history_len=history_len,
        min_texture_energy=min_texture_energy,
        enable_async=True,
    )

    # 3. Initialize Terminal Notifier (hysteresis logging)
    notifier = TerminalNotifier(stable_frames_required=5)

    # 4. Open Webcam
    cap, active_idx = find_available_camera(preferred_index=camera_index)
    if cap is None:
        print("\n[ERROR] Unable to access laptop webcam.")
        print("Please ensure no other app is using the camera.")
        return

    print(f"[INFO] Webcam initialized on /dev/video{active_idx}.")

    # Configure webcam capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 5. Set up Edge-to-Edge Fullscreen Window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    is_fullscreen = True

    # UI State flags
    show_reticle = True
    save_notification_timer = 0.0
    save_notification_text = ""

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                time.sleep(0.01)
                continue

            # 1. Correct Left-Right Inversion (Mirror View)
            mirrored_frame = cv2.flip(raw_frame, 1)

            # 2. Fit and scale to fill 100% of the screen (eliminates all white borders)
            if is_fullscreen:
                frame = fit_frame_to_screen(mirrored_frame, screen_w, screen_h)
            else:
                frame = mirrored_frame

            h, w = frame.shape[:2]

            # 3. Center Target Box (Viewfinder)
            box_size = int(min(h, w) * 0.46)
            box_x = (w - box_size) // 2
            box_y = (h - box_size) // 2
            roi_bbox = (box_x, box_y, box_size, box_size)

            # 4. Asynchronous Zero-Lag Inference (<0.01ms non-blocking)
            if classifier.is_model_ready:
                prediction_bbox = roi_bbox if show_reticle else None
                result = classifier.predict_async(frame, roi_bbox=prediction_bbox)
                predicted_class = result["class"]
                confidence = result["confidence"]
                is_uncertain = result["is_uncertain"]

                # Send update to terminal notifier (only prints on stable transitions)
                notifier.update(predicted_class, confidence, is_uncertain)
            else:
                predicted_class = None
                confidence = 0.0
                is_uncertain = True

            # -------------------------------------------------------------
            # MINIMALIST HUD (Only Face and Essential Information Visible)
            # -------------------------------------------------------------

            # 1. Subtle, clean center viewfinder reticle
            if show_reticle:
                reticle_color = (
                    (130, 130, 150)
                    if is_uncertain
                    else CLASS_COLORS.get(predicted_class, (200, 200, 200))
                )
                draw_corner_rect(frame, roi_bbox, color=reticle_color, thickness=2, corner_len=24)

            # 2. Top-Center Floating Pill Badge (Pure ASCII to avoid ??? glyphs)
            if not classifier.is_model_ready:
                badge_text = "MODEL NOT TRAINED | RUN PYTHON TRAIN.PY"
                badge_color = HUDTheme.COLOR_UNCERTAIN
            elif is_uncertain:
                badge_text = "UNCERTAIN - SHOW WATCH CLEARLY"
                badge_color = HUDTheme.COLOR_UNCERTAIN
            else:
                cls_name = CLASS_DISPLAY_NAMES.get(predicted_class, str(predicted_class).title()).upper()
                conf_pct = int(round(confidence * 100))
                badge_text = f"{cls_name}  |  {conf_pct}%"
                badge_color = CLASS_COLORS.get(predicted_class, HUDTheme.COLOR_ACCENT)

            # Render modern floating pill badge
            draw_floating_badge(
                img=frame,
                text=badge_text,
                color=badge_color,
                center_x=w // 2,
                top_y=36,
                font_scale=0.82,
                thickness=2,
            )

            # 3. Clean, subtle shortcut hints in bottom-right corner (ASCII only)
            draw_minimal_hint(
                img=frame,
                text="[Q] Quit   [F] Fullscreen   [S] Save   [T] Reticle",
                x=w - 380,
                y=h - 22,
                font_scale=0.48,
            )

            # 4. Save frame toast notification
            if time.time() - save_notification_timer < 2.0:
                toast_w = 340
                toast_x = (w - toast_w) // 2
                toast_y = h - 65
                cv2.rectangle(frame, (toast_x, toast_y), (toast_x + toast_w, toast_y + 36), (20, 90, 40), -1)
                cv2.rectangle(frame, (toast_x, toast_y), (toast_x + toast_w, toast_y + 36), (80, 230, 120), 1)
                cv2.putText(
                    frame,
                    save_notification_text,
                    (toast_x + 18, toast_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Render edge-to-edge frame
            cv2.imshow(window_name, frame)

            # -------------------------------------------------------------
            # KEYBOARD CONTROLS
            # -------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:  # 27 is ESC
                print("\n[INFO] Exit command received. Closing...", flush=True)
                break
            elif key in [ord("f"), ord("F")]:
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)
                print(f"[INFO] Fullscreen toggled: {'ON' if is_fullscreen else 'OFF'}", flush=True)
            elif key in [ord("s"), ord("S")]:
                now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_filename = f"watch_{now_str}.jpg"
                save_path = os.path.join(captures_dir, save_filename)
                cv2.imwrite(save_path, frame)
                save_notification_timer = time.time()
                save_notification_text = f"Snapshot Saved: {save_filename}"
                print(f"[CAPTURE] Snapshot saved to: {save_path}", flush=True)
            elif key in [ord("r"), ord("R")]:
                classifier.reset_history()
                notifier.reset()
                save_notification_timer = time.time()
                save_notification_text = "Smoothing History Reset"
            elif key in [ord("t"), ord("T")]:
                show_reticle = not show_reticle
                classifier.reset_history()

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Exiting...")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Releasing webcam resources...")
        classifier.close()
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Clean exit completed.")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Watch Type Classification (Edge-to-Edge Fullscreen).")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--model", type=str, default="models/watch_classifier_mobilenetv2.keras")
    parser.add_argument("--labels", type=str, default="models/class_labels.json")
    parser.add_argument("--threshold", type=float, default=0.65, help="Confidence threshold")
    parser.add_argument("--margin", type=float, default=0.15, help="Probability margin threshold")
    parser.add_argument("--history", type=int, default=8, help="Rolling window size for temporal smoothing")
    parser.add_argument("--min_energy", type=float, default=30.0, help="Minimum texture energy threshold")
    parser.add_argument("--captures", type=str, default="captures", help="Snapshots directory")
    args = parser.parse_args()

    run_webcam(
        camera_index=args.camera,
        model_path=args.model,
        labels_path=args.labels,
        confidence_threshold=args.threshold,
        margin_threshold=args.margin,
        history_len=args.history,
        min_texture_energy=args.min_energy,
        captures_dir=args.captures,
    )


if __name__ == "__main__":
    main()
