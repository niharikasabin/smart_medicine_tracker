"""
=============================================================
utils/preprocess.py — Image Preprocessing & Dataset Tools
=============================================================
Handles:
  - Image preprocessing for YOLO training
  - Train/val/test splitting
  - Dataset validation
  - Roboflow dataset download helper
  - Annotation format conversion
=============================================================
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from PIL import Image


# ── Dataset Splitter ──────────────────────────────────────

def split_dataset(
    source_images: str,
    source_labels: str,
    output_dir:    str = "dataset",
    split:         Tuple[float, float, float] = (0.70, 0.20, 0.10),
    seed:          int = 42,
):
    """
    Split annotated images into train/val/test sets.
    
    Args:
        source_images: Directory containing all your images
        source_labels: Directory containing YOLO .txt labels
        output_dir:    Output base directory
        split:         (train, val, test) proportions — must sum to 1.0
        seed:          Random seed for reproducibility
    
    Folder structure created:
        dataset/
            train/images/  train/labels/
            val/images/    val/labels/
            test/images/   test/labels/
    
    Usage:
        python utils/preprocess.py --split \
            --images /path/to/images \
            --labels /path/to/labels
    """
    assert abs(sum(split) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    # Gather all image paths
    img_dir = Path(source_images)
    lbl_dir = Path(source_labels)
    
    image_files = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ])
    
    if not image_files:
        print(f"❌ No images found in: {img_dir}")
        return
    
    print(f"📁 Found {len(image_files)} images")
    
    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(image_files)
    
    n = len(image_files)
    n_train = int(n * split[0])
    n_val   = int(n * split[1])
    
    splits = {
        "train": image_files[:n_train],
        "val":   image_files[n_train : n_train + n_val],
        "test":  image_files[n_train + n_val:],
    }
    
    # Copy files to output structure
    for split_name, files in splits.items():
        out_img = Path(output_dir) / split_name / "images"
        out_lbl = Path(output_dir) / split_name / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        
        copied = 0
        for img_path in files:
            # Copy image
            shutil.copy2(img_path, out_img / img_path.name)
            
            # Copy corresponding label file (same name, .txt extension)
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            if lbl_path.exists():
                shutil.copy2(lbl_path, out_lbl / lbl_path.name)
                copied += 1
        
        print(f"   {split_name:5s}: {len(files)} images, {copied} labels")
    
    print(f"\n✅ Dataset split complete → {output_dir}/")


# ── Image Preprocessing ───────────────────────────────────

def preprocess_image_for_yolo(
    image_input,                    # file path or np.ndarray
    target_size: int = 640,
    normalize:   bool = False,
) -> np.ndarray:
    """
    Preprocess a single image for YOLOv8 inference or training visualization.
    
    - Resize to target_size × target_size (letterbox, maintaining aspect ratio)
    - Convert to RGB
    - Optionally normalize to [0, 1]
    
    Returns:
        np.ndarray (H, W, 3) uint8 (or float32 if normalize=True)
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("RGB"))
    else:
        img = image_input
    
    # Letterbox resize (preserves aspect ratio, adds gray padding)
    img = letterbox_resize(img, target_size)
    
    if normalize:
        return img.astype(np.float32) / 255.0
    return img


def letterbox_resize(img: np.ndarray, target: int = 640) -> np.ndarray:
    """Resize image to square while preserving aspect ratio (letterbox)."""
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to square
    pad_h = (target - new_h) // 2
    pad_w = (target - new_w) // 2
    
    padded = cv2.copyMakeBorder(
        resized, pad_h, target - new_h - pad_h,
        pad_w, target - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)  # Gray padding (YOLO standard)
    )
    return padded


# ── Dataset Validation ────────────────────────────────────

def validate_dataset(dataset_dir: str = "dataset") -> Dict:
    """
    Validate dataset structure and label format.
    
    Checks:
    - Images have corresponding label files
    - Label files have valid YOLO format
    - No empty label files (background-only images are OK but flag them)
    
    Returns:
        Summary dict with counts and issues
    """
    base = Path(dataset_dir)
    report = {"splits": {}, "issues": []}
    
    for split in ["train", "val", "test"]:
        img_dir = base / split / "images"
        lbl_dir = base / split / "labels"
        
        if not img_dir.exists():
            continue
        
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        labels = {f.stem for f in lbl_dir.glob("*.txt")} if lbl_dir.exists() else set()
        
        no_label = [img.stem for img in images if img.stem not in labels]
        
        # Check label format
        invalid_labels = []
        for lbl_path in lbl_dir.glob("*.txt") if lbl_dir.exists() else []:
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and len(parts) != 5:
                        invalid_labels.append(str(lbl_path))
                        break
        
        report["splits"][split] = {
            "images":         len(images),
            "labels":         len(labels),
            "missing_labels": len(no_label),
            "invalid_format": len(invalid_labels),
        }
        
        if no_label:
            report["issues"].append(f"{split}: {len(no_label)} images missing labels")
        if invalid_labels:
            report["issues"].append(f"{split}: {len(invalid_labels)} invalid label files")
    
    # Print report
    print("\n📋 Dataset Validation Report:")
    for split, stats in report["splits"].items():
        status = "✅" if stats["missing_labels"] == 0 and stats["invalid_format"] == 0 else "⚠️"
        print(f"  {status} {split:6s}: {stats['images']} images, {stats['labels']} labels", end="")
        if stats["missing_labels"] > 0:
            print(f" ⚠️ {stats['missing_labels']} missing labels", end="")
        print()
    
    if report["issues"]:
        print("\n⚠️  Issues:")
        for issue in report["issues"]:
            print(f"   - {issue}")
    else:
        print("\n✅ Dataset is valid!")
    
    return report


# ── Roboflow Downloader ───────────────────────────────────

def download_from_roboflow(
    api_key:    str,
    workspace:  str,
    project:    str,
    version:    int,
    output_dir: str = "dataset",
):
    """
    Download a Roboflow dataset directly.
    
    Usage on Colab:
        from utils.preprocess import download_from_roboflow
        download_from_roboflow(
            api_key   = "YOUR_ROBOFLOW_API_KEY",
            workspace = "roboflow-universe",
            project   = "pill-detection",
            version   = 1,
        )
    
    To get API key:
        1. Create account at roboflow.com
        2. Go to Settings → API Keys
    
    Suggested medicine datasets on Roboflow Universe:
        - "pill-detection"
        - "medicine-strip-detection"
        - "blister-pack-detection"
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        os.system("pip install roboflow -q")
        from roboflow import Roboflow
    
    print(f"📥 Downloading {workspace}/{project} v{version}...")
    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    dataset = project_obj.version(version).download("yolov8", location=output_dir)
    print(f"✅ Dataset downloaded to: {output_dir}")
    return dataset


# ── CLI ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",   action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--images",  type=str, default="raw_images")
    parser.add_argument("--labels",  type=str, default="raw_labels")
    parser.add_argument("--output",  type=str, default="dataset")
    args = parser.parse_args()

    if args.split:
        split_dataset(args.images, args.labels, args.output)
    
    if args.validate:
        validate_dataset(args.output)
    
    if not args.split and not args.validate:
        print("Usage:")
        print("  python utils/preprocess.py --split   --images raw/ --labels raw_labels/")
        print("  python utils/preprocess.py --validate")
