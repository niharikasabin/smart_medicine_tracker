"""
=============================================================
models/inference.py — YOLOv8 Medicine Detection Inference
=============================================================
Run detection on a single image or camera feed.
Returns bounding boxes, class names, confidence scores,
and an annotated image ready for display.
=============================================================
"""

import os
import io
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import cv2
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    os.system("pip install ultralytics -q")
    from ultralytics import YOLO


# ── Data Classes ──────────────────────────────────────────

@dataclass
class Detection:
    """Single detection result."""
    class_id:    int
    class_name:  str
    confidence:  float
    bbox:        Tuple[int, int, int, int]   # x1, y1, x2, y2 (pixels)
    center:      Tuple[float, float]          # normalized cx, cy


@dataclass
class InferenceResult:
    """Full inference output for one image."""
    detections:      List[Detection]
    annotated_image: np.ndarray              # BGR image with drawn boxes
    medicine_found:  bool
    count:           int
    class_counts:    Dict[str, int]          # e.g. {'pill_strip': 2}


# ── Detector Class ────────────────────────────────────────

class MedicineDetector:
    """
    YOLOv8-based medicine detector.
    
    Usage:
        detector = MedicineDetector("models/medicine_yolo.pt")
        result = detector.detect(image_path="photo.jpg")
        cv2.imshow("result", result.annotated_image)
    """

    # Fallback to pretrained COCO model if custom not trained yet
    DEFAULT_MODEL = "yolov8n.pt"
    
    # Class name → display color (BGR)
    CLASS_COLORS = {
        "pill_strip":    (0,   200, 100),    # Green
        "pill_bottle":   (255, 140,  0),     # Orange
        "tablet":        (0,   150, 255),    # Blue
        "medicine_box":  (200,  50, 200),    # Purple
        "default":       (0,   255, 255),    # Cyan fallback
    }

    def __init__(self, model_path: str = None, conf_threshold: float = 0.45):
        """
        Args:
            model_path: Path to trained .pt weights. 
                        If None, tries models/medicine_yolo.pt then falls back to yolov8n.pt
            conf_threshold: Minimum confidence to count as detection (0–1)
        """
        self.conf = conf_threshold
        self.model_path = self._resolve_model(model_path)
        print(f"🔍 Loading detector: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names   # Dict[int, str]
        print(f"✅ Detector ready | Classes: {list(self.class_names.values())}")

    def _resolve_model(self, path: Optional[str]) -> str:
        """Resolve model path with fallback logic."""
        candidates = [
            path,
            "models/medicine_yolo.pt",
            "medicine_yolo.pt",
        ]
        for p in candidates:
            if p and Path(p).exists():
                return p
        print(f"⚠️  Custom model not found. Using pretrained YOLOv8 ({self.DEFAULT_MODEL})")
        print("   Train your model first: python models/train_yolo.py")
        return self.DEFAULT_MODEL

    # ── Core Detection ────────────────────────────────────

    def detect(
        self,
        image_input,                    # file path, PIL Image, or np.ndarray
        draw: bool = True,
    ) -> InferenceResult:
        """
        Run detection on an image.
        
        Args:
            image_input: Can be:
                - str / Path: image file path
                - PIL.Image: uploaded via Streamlit file_uploader
                - np.ndarray: camera frame (BGR or RGB)
            draw: Whether to draw bounding boxes on output image
        
        Returns:
            InferenceResult with detections and annotated image
        """
        # ── Normalize input to np.ndarray (BGR) ──────────
        img_bgr = self._load_image(image_input)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ── Run YOLO ──────────────────────────────────────
        results = self.model(img_rgb, conf=self.conf, verbose=False)[0]

        # ── Parse detections ──────────────────────────────
        detections: List[Detection] = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_score      = float(box.conf[0])
            cls_id          = int(box.cls[0])
            cls_name        = self.class_names.get(cls_id, f"class_{cls_id}")
            h, w            = img_bgr.shape[:2]

            detections.append(Detection(
                class_id   = cls_id,
                class_name = cls_name,
                confidence = round(conf_score, 3),
                bbox       = (x1, y1, x2, y2),
                center     = ((x1 + x2) / (2 * w), (y1 + y2) / (2 * h)),
            ))

        # ── Draw bounding boxes ───────────────────────────
        annotated = self._draw_boxes(img_bgr.copy(), detections) if draw else img_bgr

        # ── Class counts ──────────────────────────────────
        class_counts: Dict[str, int] = {}
        for d in detections:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

        return InferenceResult(
            detections      = detections,
            annotated_image = annotated,
            medicine_found  = len(detections) > 0,
            count           = len(detections),
            class_counts    = class_counts,
        )

    def detect_from_bytes(self, image_bytes: bytes) -> InferenceResult:
        """Convenience: detect from raw bytes (Streamlit file_uploader.read())."""
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.detect(pil_img)

    # ── Drawing ───────────────────────────────────────────

    def _draw_boxes(self, img: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw professional-looking bounding boxes with labels."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self.CLASS_COLORS.get(det.class_name, self.CLASS_COLORS["default"])
            
            # Box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label background
            label = f"{det.class_name}  {det.confidence:.0%}"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x1, y1 - lh - baseline - 6), (x1 + lw + 8, y1), color, -1)
            
            # Label text
            cv2.putText(
                img, label,
                (x1 + 4, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
        
        # ── Detection count overlay ───────────────────────
        if detections:
            msg = f"Detected: {len(detections)} object(s)"
            cv2.putText(img, msg, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 120), 2, cv2.LINE_AA)
        
        return img

    # ── Input Normalization ───────────────────────────────

    def _load_image(self, image_input) -> np.ndarray:
        """Convert any supported input type to BGR numpy array."""
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not read image: {image_input}")
            return img

        if isinstance(image_input, Image.Image):
            return cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)

        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 3 and image_input.shape[2] == 3:
                # Assume RGB if from Streamlit/PIL, keep as-is if BGR from cv2
                return image_input
            raise ValueError(f"Unexpected array shape: {image_input.shape}")

        raise TypeError(f"Unsupported image type: {type(image_input)}")

    # ── Annotated PIL Image (for Streamlit) ───────────────

    def to_pil(self, result: InferenceResult) -> Image.Image:
        """Convert annotated BGR array to PIL for Streamlit st.image()."""
        return Image.fromarray(cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB))


# ── CLI Usage ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    detector = MedicineDetector()
    result = detector.detect(image_path)

    print(f"\n📦 Detections: {result.count}")
    print(f"💊 Medicine found: {result.medicine_found}")
    print(f"📊 Class counts: {json.dumps(result.class_counts, indent=2)}")
    
    for i, d in enumerate(result.detections):
        print(f"  [{i+1}] {d.class_name} | conf={d.confidence:.2%} | bbox={d.bbox}")

    # Save annotated output
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, result.annotated_image)
    print(f"\n✅ Annotated image saved: {output_path}")
