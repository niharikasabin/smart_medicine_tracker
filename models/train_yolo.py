"""
=============================================================
models/train_yolo.py — YOLOv8 Training Script
=============================================================
Train a custom YOLOv8 model on your medicine dataset.

Usage:
    python models/train_yolo.py

Google Colab:
    !python models/train_yolo.py --epochs 50 --colab
=============================================================
"""

import argparse
import os
import sys
from pathlib import Path

# ── Dependency check ──────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics -q")
    from ultralytics import YOLO


# ── Configuration ─────────────────────────────────────────
CONFIG = {
    "model":       "yolov8n.pt",        # nano = fastest; use yolov8s.pt for better accuracy
    "data":        "dataset/medicine.yaml",
    "epochs":      50,                  # Increase to 100 for better results
    "imgsz":       640,                 # Input image size
    "batch":       16,                  # Reduce to 8 if GPU OOM
    "device":      "0",                 # '0' for GPU, 'cpu' for CPU
    "project":     "models/runs",       # Where to save results
    "name":        "medicine_detect",
    "patience":    10,                  # Early stopping patience
    "save":        True,
    "plots":       True,                # Save training plots
    "val":         True,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Medicine Detection")
    parser.add_argument("--epochs",  type=int,   default=CONFIG["epochs"])
    parser.add_argument("--batch",   type=int,   default=CONFIG["batch"])
    parser.add_argument("--imgsz",   type=int,   default=CONFIG["imgsz"])
    parser.add_argument("--device",  type=str,   default=CONFIG["device"])
    parser.add_argument("--model",   type=str,   default=CONFIG["model"])
    parser.add_argument("--colab",   action="store_true", help="Running on Google Colab")
    return parser.parse_args()


def setup_colab_paths():
    """Adjust paths when running on Google Colab with Drive mounted."""
    print("📁 Setting up Google Colab paths...")
    
    # Mount Google Drive if not already mounted
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("✅ Google Drive mounted")
    except ImportError:
        print("ℹ️  Not running in Colab — skipping Drive mount")
        return
    
    # Adjust dataset path to Drive location
    drive_dataset = "/content/drive/MyDrive/MedicineTracker/dataset"
    if os.path.exists(drive_dataset):
        CONFIG["data"] = f"{drive_dataset}/medicine.yaml"
        print(f"✅ Using dataset from Drive: {drive_dataset}")
    else:
        print(f"⚠️  Dataset not found at {drive_dataset}")
        print("   Please upload your dataset to Google Drive at:")
        print("   MyDrive/MedicineTracker/dataset/")


def download_pretrained_model(model_name: str):
    """Download YOLOv8 pretrained weights (auto-handled by Ultralytics)."""
    print(f"📥 Loading pretrained model: {model_name}")
    # Ultralytics automatically downloads weights on first use
    model = YOLO(model_name)
    print(f"✅ Model loaded: {model_name}")
    return model


def train(args):
    """Main training function."""
    print("\n" + "="*60)
    print("  🏥 Smart Medicine Tracker — YOLOv8 Training")
    print("="*60)
    
    # Handle Colab environment
    if args.colab:
        setup_colab_paths()
    
    # Verify dataset exists
    yaml_path = Path(CONFIG["data"])
    if not yaml_path.exists():
        print(f"\n❌ Dataset config not found: {yaml_path}")
        print("\n📋 To fix this:")
        print("   1. Download a medicine dataset from Roboflow")
        print("   2. Or collect 100–300 images and annotate with LabelImg")
        print("   3. Place dataset in: dataset/ folder")
        print("   4. Run: python utils/preprocess.py --split")
        sys.exit(1)
    
    # Load pretrained YOLOv8 model (transfer learning)
    model = download_pretrained_model(args.model)
    
    print(f"\n🚀 Starting training with config:")
    print(f"   Model:   {args.model}")
    print(f"   Epochs:  {args.epochs}")
    print(f"   Batch:   {args.batch}")
    print(f"   ImgSz:   {args.imgsz}")
    print(f"   Device:  {args.device}")
    print(f"   Data:    {CONFIG['data']}\n")
    
    # ── TRAIN ─────────────────────────────────────────────
    results = model.train(
        data=CONFIG["data"],
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=CONFIG["project"],
        name=CONFIG["name"],
        patience=CONFIG["patience"],
        save=CONFIG["save"],
        plots=CONFIG["plots"],
        val=CONFIG["val"],
        augment=True,               # Data augmentation (flip, mosaic, etc.)
        degrees=10.0,               # Rotation augmentation
        translate=0.1,              # Translation augmentation
        scale=0.5,                  # Scale augmentation
        flipud=0.0,                 # No vertical flip (medicine orientation matters)
        fliplr=0.5,                 # Horizontal flip
        mosaic=1.0,                 # Mosaic augmentation (4 images combined)
        verbose=True,
    )
    
    # ── VALIDATE ──────────────────────────────────────────
    print("\n📊 Running validation...")
    metrics = model.val()
    
    print("\n✅ Training Complete!")
    print(f"   mAP50:    {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Precision:{metrics.box.mp:.4f}")
    print(f"   Recall:   {metrics.box.mr:.4f}")
    
    # ── EXPORT BEST MODEL ─────────────────────────────────
    best_model_path = Path(CONFIG["project"]) / CONFIG["name"] / "weights" / "best.pt"
    final_path = Path("models/medicine_yolo.pt")
    
    if best_model_path.exists():
        import shutil
        shutil.copy(best_model_path, final_path)
        print(f"\n💾 Best model saved to: {final_path}")
        
        # Also save to Drive if on Colab
        if args.colab:
            drive_save = "/content/drive/MyDrive/MedicineTracker/models/"
            os.makedirs(drive_save, exist_ok=True)
            shutil.copy(best_model_path, f"{drive_save}medicine_yolo.pt")
            print(f"☁️  Model saved to Google Drive: {drive_save}")
    
    return results


if __name__ == "__main__":
    args = parse_args()
    train(args)


# ============================================================
# GOOGLE COLAB COMMANDS (run these in a Colab cell):
# ============================================================
# !pip install ultralytics -q
# from google.colab import drive
# drive.mount('/content/drive')
#
# # Clone or upload project
# import os
# os.chdir('/content/smart_medicine_tracker')
#
# # Train (GPU enabled — set Runtime > Change Runtime > T4 GPU)
# !python models/train_yolo.py --epochs 50 --colab
#
# # Monitor training in real-time
# %load_ext tensorboard
# %tensorboard --logdir models/runs
# ============================================================
