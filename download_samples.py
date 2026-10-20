"""
Fast, Legal Sample Dataset Downloader for Watch Classification.
Fetches Creative Commons and Public Domain watch images from Wikimedia Commons
using optimized thumbnail APIs (~30KB per image) for rapid training set initialization.
"""

import argparse
import os
import random
import time
from typing import Dict, List
import cv2
import numpy as np
import requests

CLASSES = {
    "analog": [
        "Category:Mechanical wristwatches",
        "Category:Analog chronographs (watches)",
        "Category:Quartz wristwatches",
        "Category:Wristwatches for males",
    ],
    "digital": [
        "Category:Digital watches",
        "Category:Calculator watches",
        "Category:Casio watches",
        "Category:G-Shock",
    ],
    "smartwatch": [
        "Category:Smartwatches",
        "Category:Apple Watch",
        "Category:Wear OS",
        "Category:Pebble (watch)",
        "Category:Activity trackers",
    ],
}

SPLITS = ["train", "validation", "test"]
SPLIT_WEIGHTS = [0.70, 0.15, 0.15]
USER_AGENT = "WatchClassifierEduProject/1.0 (https://github.com/educational-ml; contact@example.edu)"


def fetch_category_thumb_urls(category: str, limit: int = 25, width: int = 500) -> List[str]:
    """Queries Wikimedia Commons API for optimized thumbnail image URLs."""
    endpoint = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "iiurlwidth": width,
    }
    headers = {"User-Agent": USER_AGENT}

    try:
        resp = requests.get(endpoint, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        urls = []
        for p in pages.values():
            ii = p.get("imageinfo", [{}])[0]
            thumb = ii.get("thumburl", "")
            raw = ii.get("url", "")
            chosen = thumb if thumb else raw
            if chosen:
                clean_url = chosen.split("?")[0]
                urls.append(clean_url)
        return urls
    except Exception as e:
        print(f"  [WARN] Failed to query {category}: {e}")
        return []


def download_and_verify(url: str, output_path: str, target_size: int = 384) -> bool:
    """Downloads thumbnail, verifies validity with OpenCV, and saves."""
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200 or len(r.content) < 2048:
            return False

        nparr = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return False

        h, w = img.shape[:2]
        if h < 48 or w < 48:
            return False

        # Resize to standardized dimensions
        if max(h, w) > target_size:
            scale = target_size / float(max(h, w))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        cv2.imwrite(output_path, img)
        return True
    except Exception:
        return False


def populate_dataset(
    dataset_dir: str = "dataset",
    samples_per_class: int = 24,
) -> None:
    """Downloads images and organizes them cleanly across train/validation/test."""
    print("=" * 70)
    print("Rapid Open-Access Sample Watch Dataset Downloader")
    print("======================================================================")
    print(f"Acquiring ~{samples_per_class} verified samples per category.")

    random.seed(42)

    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(dataset_dir, split, cls), exist_ok=True)

    summary_counts: Dict[str, int] = {cls: 0 for cls in CLASSES}

    for cls_name, categories in CLASSES.items():
        print(f"\n[FETCHING] Category: {cls_name.upper()} ...")
        candidate_urls: List[str] = []

        for cat in categories:
            urls = fetch_category_thumb_urls(cat, limit=18, width=500)
            for u in urls:
                if u not in candidate_urls:
                    candidate_urls.append(u)
            if len(candidate_urls) >= samples_per_class * 2:
                break
            time.sleep(0.15)

        print(f"  Acquired {len(candidate_urls)} URLs. Downloading thumbnails...")
        saved_count = 0

        for url in candidate_urls:
            if saved_count >= samples_per_class:
                break

            split = random.choices(SPLITS, weights=SPLIT_WEIGHTS, k=1)[0]
            filename = f"{cls_name}_{saved_count + 1:03d}.jpg"
            out_path = os.path.join(dataset_dir, split, cls_name, filename)

            if download_and_verify(url, out_path):
                saved_count += 1
                summary_counts[cls_name] += 1
                print(f"    [{saved_count:02d}/{samples_per_class}] -> {split}/{cls_name}/{filename}")
            time.sleep(0.08)

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    for k, v in summary_counts.items():
        print(f"  {k:<14}: {v} verified images")
    print("\nNext step: Run 'python prepare_dataset.py' to run integrity and deduplication checks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download sample watch dataset.")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Target dataset folder")
    parser.add_argument("--per_class", type=int, default=20, help="Target images per category")
    args = parser.parse_args()

    populate_dataset(dataset_dir=args.dataset_dir, samples_per_class=args.per_class)
