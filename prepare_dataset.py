"""
Dataset Validation, Cleansing, and Leakage Prevention Script.
Checks dataset integrity, removes corrupted/unreadable images, verifies image dimensions,
and prevents data leakage across train/validation/test splits using perceptual dHash.
"""

import argparse
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import cv2
import numpy as np
from PIL import Image

from src.utils import compute_dhash, hamming_distance

SPLITS = ["train", "validation", "test"]
CLASSES = ["analog", "digital", "smartwatch"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def check_and_create_structure(base_dir: str = "dataset") -> None:
    """Ensures the standard folder hierarchy exists."""
    for split in SPLITS:
        for cls in CLASSES:
            path = os.path.join(base_dir, split, cls)
            os.makedirs(path, exist_ok=True)


def scan_and_clean_images(
    base_dir: str = "dataset",
    remove_corrupt: bool = True,
    quarantine_dir: str = "dataset/corrupted_quarantine",
) -> Dict[str, Dict[str, List[str]]]:
    """
    Inspects all images in the dataset, checks integrity, and filters valid files.
    Moves corrupt images to quarantine if remove_corrupt is True.
    """
    valid_files: Dict[str, Dict[str, List[str]]] = {
        split: {cls: [] for cls in CLASSES} for split in SPLITS
    }
    corrupted_count = 0

    print("=" * 70)
    print("STEP 1: Scanning Dataset Integrity & Cleansing Corrupted Files")
    print("=" * 70)

    for split in SPLITS:
        for cls in CLASSES:
            folder = os.path.join(base_dir, split, cls)
            if not os.path.isdir(folder):
                continue

            for root, _, files in os.walk(folder):
                for fname in sorted(files):
                    fpath = os.path.join(root, fname)
                    ext = os.path.splitext(fname)[1].lower()

                    # Extension check
                    if ext not in VALID_EXTENSIONS:
                        print(f"  [SKIP] Non-image file ignored: {fpath}")
                        continue

                    # File size check
                    if os.path.getsize(fpath) == 0:
                        print(f"  [CORRUPT] 0-byte file: {fpath}")
                        corrupted_count += 1
                        _quarantine(fpath, quarantine_dir, remove_corrupt)
                        continue

                    # PIL integrity check
                    try:
                        with Image.open(fpath) as im:
                            im.verify()
                    except Exception as e:
                        print(f"  [CORRUPT] PIL unreadable ({e}): {fpath}")
                        corrupted_count += 1
                        _quarantine(fpath, quarantine_dir, remove_corrupt)
                        continue

                    # OpenCV reading and dimension check
                    try:
                        cv_img = cv2.imread(fpath, cv2.IMREAD_COLOR)
                        if cv_img is None or cv_img.size == 0:
                            print(f"  [CORRUPT] OpenCV decode failed: {fpath}")
                            corrupted_count += 1
                            _quarantine(fpath, quarantine_dir, remove_corrupt)
                            continue
                        h, w = cv_img.shape[:2]
                        if h < 32 or w < 32:
                            print(f"  [TOO SMALL] Image dimensions {w}x{h} < 32x32: {fpath}")
                            corrupted_count += 1
                            _quarantine(fpath, quarantine_dir, remove_corrupt)
                            continue
                    except Exception as e:
                        print(f"  [CORRUPT] OpenCV exception ({e}): {fpath}")
                        corrupted_count += 1
                        _quarantine(fpath, quarantine_dir, remove_corrupt)
                        continue

                    valid_files[split][cls].append(fpath)

    print(f"\n[INTEGRITY SCAN COMPLETED] Corrupted/Invalid files found: {corrupted_count}")
    return valid_files


def _quarantine(file_path: str, quarantine_dir: str, do_remove: bool) -> None:
    if not do_remove:
        return
    try:
        os.makedirs(quarantine_dir, exist_ok=True)
        shutil.move(file_path, os.path.join(quarantine_dir, os.path.basename(file_path)))
    except Exception as e:
        print(f"    Failed to move {file_path} to quarantine: {e}")


def detect_data_leakage(
    valid_files: Dict[str, Dict[str, List[str]]],
    hamming_threshold: int = 3,
    remove_leakage: bool = True,
) -> int:
    """
    Uses perceptual dHash to detect identical or near-duplicate images across splits.
    Near-identical images appearing across train and (validation/test) represent data leakage.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Perceptual Hash Duplicate & Data Leakage Detection")
    print("=" * 70)

    hashes_by_split: Dict[str, List[Tuple[str, str, int]]] = {
        split: [] for split in SPLITS
    }

    # Compute hashes
    for split in SPLITS:
        for cls in CLASSES:
            for fpath in valid_files[split][cls]:
                try:
                    gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                    if gray is not None:
                        h = compute_dhash(gray)
                        hashes_by_split[split].append((cls, fpath, h))
                except Exception:
                    continue

    leakage_count = 0
    train_hashes = hashes_by_split["train"]

    # Check for leakage between train and validation / test
    for eval_split in ["validation", "test"]:
        eval_entries = hashes_by_split[eval_split]
        for e_cls, e_path, e_hash in eval_entries:
            for t_cls, t_path, t_hash in train_hashes:
                dist = hamming_distance(e_hash, t_hash)
                if dist <= hamming_threshold:
                    leakage_count += 1
                    print(
                        f"  [DATA LEAKAGE DETECTED] (Hamming dist: {dist})\n"
                        f"    Train: {t_path}\n"
                        f"    {eval_split.capitalize()}: {e_path}"
                    )
                    if remove_leakage:
                        try:
                            os.remove(e_path)
                            print(f"    -> Removed duplicate from {eval_split}: {e_path}")
                        except Exception as e:
                            print(f"    -> Error removing {e_path}: {e}")

    print(f"\n[LEAKAGE CHECK COMPLETED] Total duplicate/leaked images detected: {leakage_count}")
    return leakage_count


def report_statistics(base_dir: str = "dataset") -> Dict[str, Dict[str, int]]:
    """Prints a clear tabular breakdown of images per split and class."""
    print("\n" + "=" * 70)
    print("STEP 3: Dataset Summary Report")
    print("=" * 70)

    counts: Dict[str, Dict[str, int]] = {split: {} for split in SPLITS}
    total_all = 0

    header = f"{'Split':<12} | {'Analog':<10} | {'Digital':<10} | {'Smartwatch':<12} | {'Total':<10}"
    print(header)
    print("-" * len(header))

    for split in SPLITS:
        split_total = 0
        row_str = f"{split.capitalize():<12} | "
        for cls in CLASSES:
            folder = os.path.join(base_dir, split, cls)
            c = 0
            if os.path.isdir(folder):
                c = len([f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS])
            counts[split][cls] = c
            split_total += c
            row_str += f"{c:<10} | " if cls != "smartwatch" else f"{c:<12} | "
        row_str += f"{split_total:<10}"
        total_all += split_total
        print(row_str)

    print("-" * len(header))
    print(f"{'OVERALL':<12} | {sum(counts[s]['analog'] for s in SPLITS):<10} | "
          f"{sum(counts[s]['digital'] for s in SPLITS):<10} | "
          f"{sum(counts[s]['smartwatch'] for s in SPLITS):<12} | "
          f"{total_all:<10}\n")

    if total_all == 0:
        print("[NOTICE] Your dataset is currently empty.")
        print("Place your watch photos directly into the class folders:")
        print("    dataset/train/analog/, dataset/train/digital/, dataset/train/smartwatch/")
        print("Or use 'S' in main.py to capture snapshots to captures/.\n")

    return counts


def rebalance_dataset(base_dir: str = "dataset", val_count: int = 3, test_count: int = 3) -> None:
    """
    Rebalances images within each class to ensure each split has at least val_count
    and test_count images, with the remainder in train.
    """
    print("\n" + "=" * 70)
    print("STEP 2.5: Rebalancing Splits to Guarantee Sufficient Val & Test Samples")
    print("=" * 70)

    random.seed(42)
    for cls in CLASSES:
        all_class_files = []
        for split in SPLITS:
            folder = os.path.join(base_dir, split, cls)
            if os.path.isdir(folder):
                for f in sorted(os.listdir(folder)):
                    if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS:
                        all_class_files.append(os.path.join(folder, f))

        random.shuffle(all_class_files)
        total = len(all_class_files)
        if total < (val_count + test_count + 1):
            print(f"  [WARN] Category {cls} has only {total} images. Skipping rebalance.")
            continue

        test_files = all_class_files[:test_count]
        val_files = all_class_files[test_count:test_count + val_count]
        train_files = all_class_files[test_count + val_count:]

        # Move files to appropriate destinations
        for split, flist in [("test", test_files), ("validation", val_files), ("train", train_files)]:
            dest_dir = os.path.join(base_dir, split, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for fpath in flist:
                dest_path = os.path.join(dest_dir, os.path.basename(fpath))
                if fpath != dest_path:
                    shutil.move(fpath, dest_path)

        print(f"  [{cls.upper()}] Total: {total} -> Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare, clean, and validate watch dataset.")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Root dataset directory")
    parser.add_argument("--quarantine", action="store_true", default=True, help="Quarantine corrupt files")
    parser.add_argument("--fix_leakage", action="store_true", default=True, help="Remove duplicate leakage images")
    parser.add_argument("--rebalance", action="store_true", help="Ensure balanced distribution across splits")
    parser.add_argument("--threshold", type=int, default=3, help="Hamming distance for duplicate detection")
    args = parser.parse_args()

    check_and_create_structure(args.dataset_dir)
    valid_files = scan_and_clean_images(args.dataset_dir, remove_corrupt=args.quarantine)
    detect_data_leakage(valid_files, hamming_threshold=args.threshold, remove_leakage=args.fix_leakage)
    if args.rebalance:
        rebalance_dataset(args.dataset_dir, val_count=3, test_count=3)
    report_statistics(args.dataset_dir)


if __name__ == "__main__":
    main()
