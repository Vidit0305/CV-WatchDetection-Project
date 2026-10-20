"""
Real-Time Watch Type Classification via Webcam and OpenCV.
Uses MobileNetV2 transfer learning, rolling-window temporal smoothing,
uncertainty gating, and spam-free terminal logging.
"""

import argparse
import datetime
import os
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
    draw_meter_bar,
    draw_rounded_banner,
)


def find_available_camera(preferred_index: int = 0, max_tested: int = 4) -> Tuple[Optional[cv2.VideoCapture], int]:
    """
    Attempts to open the preferred camera index, then probes alternative indices.
    """
    indices_to_try = [preferred_index] + [i for i in range(max_tested) if i != preferred_index]
    for idx in indices_to_try:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # Test grabbing a frame
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
    window_name: str = "Real-Time Watch Type Classifier (MobileNetV2)",
) -> None:
    """Main live webcam inference loop."""
    print("=" * 70)
    print("Real-Time Watch Type Classification \u2014 OpenCV Live Feed")
    print("=" * 70)
    print("Starting webcam... Press 'Q' or ESC to exit.")
    print("Keyboard shortcuts:")
    print("  Q / ESC : Quit application")
    print("  S       : Save current frame to captures/")
    print("  R       : Reset temporal prediction smoothing")
    print("  T       : Toggle between Viewfinder Crop and Full Frame classification")
    print("=" * 70)

    os.makedirs(captures_dir, exist_ok=True)

    # 1. Initialize Watch Classifier
    classifier = WatchClassifier(
        model_path=model_path,
        labels_path=labels_path,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        history_len=history_len,
        min_texture_energy=min_texture_energy,
    )

    # 2. Initialize Terminal Notifier
    notifier = TerminalNotifier(stable_frames_required=5)

    # 3. Open Webcam
    cap, active_idx = find_available_camera(preferred_index=camera_index)
    if cap is None:
        print("\n[ERROR] Unable to access laptop webcam.")
        print("Troubleshooting recommendations:")
        print("  1. Verify your webcam is connected or not disabled in system settings.")
        print("  2. Ensure no other application (e.g. Zoom, Meet, Browser) is using the camera.")
        print("  3. On Linux, check user permissions for /dev/video0 or run: 'sudo usermod -aG video $USER'")
        print("  4. Specify an alternative camera index via: 'python main.py --camera 1'\n")
        return

    print(f"[INFO] Webcam initialized successfully on /dev/video{active_idx}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # UI State flags
    use_roi = True
    save_notification_timer = 0.0
    save_notification_text = ""
    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 640)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[WARN] Failed to grab frame from webcam. Retrying...", flush=True)
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]

            # Calculate Center Target Box (Viewfinder)
            box_size = int(min(h, w) * 0.52)
            box_x = (w - box_size) // 2
            box_y = (h - box_size) // 2
            roi_bbox = (box_x, box_y, box_size, box_size)

            # Perform Model Prediction
            if classifier.is_model_ready:
                prediction_bbox = roi_bbox if use_roi else None
                result = classifier.predict(frame, roi_bbox=prediction_bbox)
                predicted_class = result["class"]
                confidence = result["confidence"]
                is_uncertain = result["is_uncertain"]
                uncertain_reason = result["uncertain_reason"]
                display_name = result["display_name"]
                probs = result["probabilities"]
                texture_energy = result["texture_energy"]

                # Send update to terminal notifier (logs only on stable state transitions)
                notifier.update(predicted_class, confidence, is_uncertain)
            else:
                predicted_class = None
                confidence = 0.0
                is_uncertain = True
                uncertain_reason = "Model not trained yet"
                display_name = "Model Not Trained"
                probs = {}
                texture_energy = 0.0

            # Compute FPS
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            # -------------------------------------------------------------
            # RENDER HUD OVERLAYS
            # -------------------------------------------------------------

            # 1. Viewfinder Box & Corner Reticles
            if use_roi:
                reticle_color = (
                    HUDTheme.COLOR_UNCERTAIN
                    if is_uncertain
                    else CLASS_COLORS.get(predicted_class, HUDTheme.COLOR_RETICLE)
                )
                # Outer dimming around ROI for enhanced visual focus
                draw_corner_rect(frame, roi_bbox, color=reticle_color, thickness=3, corner_len=30)
                # Target guide label
                guide_text = "Target Area: Center watch here"
                cv2.putText(
                    frame,
                    guide_text,
                    (box_x + 8, box_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

            # 2. Top Header Card (Title, Category Badge, FPS)
            header_h = 75
            draw_rounded_banner(frame, (0, 0, w, header_h), HUDTheme.BG_DARK, alpha=0.88)
            cv2.line(frame, (0, header_h), (w, header_h), HUDTheme.BORDER_SUBTLE, 1)

            # App Title & Subtitle
            cv2.putText(
                frame,
                "WATCH CLASSIFIER",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                HUDTheme.TEXT_MAIN,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "MobileNetV2 \u2022 Transfer Learning",
                (20, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                HUDTheme.TEXT_MUTED,
                1,
                cv2.LINE_AA,
            )

            # Category Badge in Center Header
            if not classifier.is_model_ready:
                badge_text = "TRAINING REQUIRED"
                badge_color = HUDTheme.COLOR_UNCERTAIN
            elif is_uncertain:
                badge_text = "UNCERTAIN"
                badge_color = HUDTheme.COLOR_UNCERTAIN
            else:
                badge_text = CLASS_DISPLAY_NAMES.get(predicted_class, str(predicted_class).upper()).upper()
                badge_color = CLASS_COLORS.get(predicted_class, HUDTheme.COLOR_ACCENT)

            # Badge pill
            badge_size, _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            badge_x = (w - badge_size[0]) // 2
            badge_pad_x = 20
            badge_rect = (badge_x - badge_pad_x, 14, badge_size[0] + (badge_pad_x * 2), 46)
            cv2.rectangle(frame, (badge_rect[0], badge_rect[1]),
                          (badge_rect[0] + badge_rect[2], badge_rect[1] + badge_rect[3]),
                          HUDTheme.BG_PANEL, -1)
            cv2.rectangle(frame, (badge_rect[0], badge_rect[1]),
                          (badge_rect[0] + badge_rect[2], badge_rect[1] + badge_rect[3]),
                          badge_color, 2)
            cv2.putText(
                frame,
                badge_text,
                (badge_x, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                badge_color,
                2,
                cv2.LINE_AA,
            )

            # Top Right: FPS and Mode indicator
            fps_str = f"FPS: {fps:.1f}"
            cv2.putText(
                frame,
                fps_str,
                (w - 120, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                HUDTheme.TEXT_MAIN,
                1,
                cv2.LINE_AA,
            )
            roi_mode_str = f"Crop: {'ROI' if use_roi else 'FULL'}"
            cv2.putText(
                frame,
                roi_mode_str,
                (w - 120, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                HUDTheme.TEXT_MUTED,
                1,
                cv2.LINE_AA,
            )

            # 3. Bottom Footer Card (Status, Meter, Controls)
            footer_h = 100
            footer_y = h - footer_h
            draw_rounded_banner(frame, (0, footer_y, w, footer_h), HUDTheme.BG_DARK, alpha=0.90)
            cv2.line(frame, (0, footer_y), (w, footer_y), HUDTheme.BORDER_SUBTLE, 1)

            if not classifier.is_model_ready:
                # Guidance prompt when model is not trained yet
                msg1 = "Model file not found! Please run 'python train.py' to train MobileNetV2."
                msg2 = "Tip: Press 'S' to take photos of watches, or run 'python collect_data.py'."
                cv2.putText(frame, msg1, (25, footer_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, HUDTheme.COLOR_UNCERTAIN, 2, cv2.LINE_AA)
                cv2.putText(frame, msg2, (25, footer_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, HUDTheme.TEXT_MAIN, 1, cv2.LINE_AA)
            elif is_uncertain:
                # Prominent Uncertainty Banner
                uncert_title = "Uncertain \u2014 show the watch clearly"
                cv2.putText(
                    frame,
                    uncert_title,
                    (25, footer_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    HUDTheme.COLOR_UNCERTAIN,
                    2,
                    cv2.LINE_AA,
                )
                reason_detail = f"Diagnostic: {uncertain_reason} (Texture Energy: {texture_energy:.1f})"
                cv2.putText(
                    frame,
                    reason_detail,
                    (25, footer_y + 64),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    HUDTheme.TEXT_MUTED,
                    1,
                    cv2.LINE_AA,
                )
            else:
                # Recognized Class and Confidence Meter
                conf_pct = confidence * 100.0
                class_color = CLASS_COLORS.get(predicted_class, HUDTheme.COLOR_ACCENT)
                cv2.putText(
                    frame,
                    f"Recognized: {display_name}",
                    (25, footer_y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    class_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Confidence: {conf_pct:.1f}%",
                    (360, footer_y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    HUDTheme.TEXT_MAIN,
                    1,
                    cv2.LINE_AA,
                )

                # Progress bar
                draw_meter_bar(frame, 25, footer_y + 46, 450, 10, confidence, class_color)

                # Probability breakdown line
                breakdown_str = " | ".join(
                    f"{c.capitalize()}: {probs.get(c, 0.0) * 100:.0f}%"
                    for c in ["analog", "digital", "smartwatch"]
                )
                cv2.putText(
                    frame,
                    breakdown_str,
                    (25, footer_y + 78),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    HUDTheme.TEXT_MUTED,
                    1,
                    cv2.LINE_AA,
                )

            # Footer Controls Cheatsheet (Right side)
            controls_x = w - 420
            cv2.putText(
                frame,
                "[Q/ESC] Quit   [S] Save Frame   [R] Reset   [T] Toggle ROI",
                (controls_x, footer_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (200, 200, 210),
                1,
                cv2.LINE_AA,
            )

            # Save Notification Toast
            if time.time() - save_notification_timer < 2.5:
                toast_rect = (w // 2 - 200, h // 2 - 25, 400, 50)
                draw_rounded_banner(frame, toast_rect, (20, 80, 40), alpha=0.92)
                cv2.rectangle(frame, (toast_rect[0], toast_rect[1]),
                              (toast_rect[0] + toast_rect[2], toast_rect[1] + toast_rect[3]),
                              (100, 240, 140), 2)
                cv2.putText(
                    frame,
                    save_notification_text,
                    (toast_rect[0] + 20, toast_rect[1] + 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Show the final frame
            cv2.imshow(window_name, frame)

            # -------------------------------------------------------------
            # KEYBOARD HANDLERS
            # -------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:  # 27 is ESC
                print("\n[INFO] Exit command received. Closing...", flush=True)
                break
            elif key in [ord("s"), ord("S")]:
                # Save frame
                now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_filename = f"watch_{now_str}.jpg"
                save_full_path = os.path.join(captures_dir, save_filename)
                cv2.imwrite(save_full_path, frame)
                save_notification_timer = time.time()
                save_notification_text = f"Saved: {save_filename}"
                print(f"[CAPTURE] Snapshot saved to: {save_full_path}", flush=True)
            elif key in [ord("r"), ord("R")]:
                classifier.reset_history()
                notifier.reset()
                save_notification_timer = time.time()
                save_notification_text = "Smoothing Buffer Reset"
            elif key in [ord("t"), ord("T")]:
                use_roi = not use_roi
                classifier.reset_history()
                print(f"[INFO] Classification mode toggled: {'ROI Viewfinder' if use_roi else 'Full Frame'}", flush=True)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Exiting cleanly...")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Proper resource cleanup
        print("[INFO] Releasing webcam resources...")
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam released and OpenCV windows closed. Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Watch Type Classification via Webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--model", type=str, default="models/watch_classifier_mobilenetv2.keras")
    parser.add_argument("--labels", type=str, default="models/class_labels.json")
    parser.add_argument("--threshold", type=float, default=0.65, help="Confidence threshold for certainty (default: 0.65)")
    parser.add_argument("--margin", type=float, default=0.15, help="Margin between top-1 and top-2 predictions")
    parser.add_argument("--history", type=int, default=8, help="Rolling window size for temporal smoothing (default: 8)")
    parser.add_argument("--min_energy", type=float, default=30.0, help="Minimum texture energy to consider scene non-empty")
    parser.add_argument("--captures", type=str, default="captures", help="Directory to save snapshot captures")
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
