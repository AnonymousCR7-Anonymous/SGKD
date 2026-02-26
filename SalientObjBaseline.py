"""
YOLOX-Tiny + Classical CV Saliency Post-Processing Baseline for Salient Object Detection

This script is a baseline comparison for the KD approach.
It uses YOLOX-Tiny for detection and then applies classical CV saliency to rescore
detections (object proposals) WITHOUT any learned teacher or KD head.
"""

import os
import sys
import glob
import math
from typing import List, Tuple, Dict
from collections import defaultdict

import cv2
import numpy as np
import torch

# Add YOLOX to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolox.exp import get_exp
from yolox.utils import postprocess

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAFFICGAZE_ROOT = ".../TrafficGaze"

# YOLOX-Tiny configuration
YOLOX_EXP_FILE = "exps/default/yolox_tiny.py"
YOLOX_WEIGHTS = "yolox_tiny.pth"

# JSON split files (matches saliency map estimation)
JSON_SPLIT_DIR = ".../Traffic_Gaze"
TRAIN_JSON = os.path.join(JSON_SPLIT_DIR, "train.json")
VALID_JSON = os.path.join(JSON_SPLIT_DIR, "valid.json")
TEST_JSON  = os.path.join(JSON_SPLIT_DIR, "test.json")

# Detection parameters
CONF_TH = 0.10  # Low threshold to get more candidates for post-processing
NMS_IOU = 0.60
YOLOX_INPUT_SIZE = (416, 416)

# Evaluation threshold (after post-processing)
EVAL_TOPK = 50  # Keep top-K after rescoring

# Classical saliency configuration
SALIENCY_METHOD = "spectral_residual"  # only one implemented here (SR)
SALIENCY_POOLING = "p95"              # {"mean", "max", "p95"}
SALIENCY_BLUR = 5                     # blur kernel for saliency map (odd int); 0 disables
SALIENCY_EPS = 1e-9
SALIENCY_ALPHA = 0.5                  # weight for combining conf + saliency in SR_CONF_SALIENCY

# 7 Classes
NUM_CLASSES = 7
OUR_CLASSES = ["people", "car", "motorcycle", "traffic-light", "traffic-sign", "bus", "truck"]

# COCO to Our class mapping
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

COCO_TO_OUR_NAME = {
    "person": "people",
    "car": "car",
    "motorcycle": "motorcycle",
    "traffic light": "traffic-light",
    "stop sign": "traffic-sign",
    "bus": "bus",
    "truck": "truck",
}

# Output directory
OUTPUT_DIR = os.path.join(TRAFFICGAZE_ROOT, "cv_baseline_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Data Loading (JSON-based splits)
# ============================================================

def video_dir(vid: int) -> str:
    return os.path.join(TRAFFICGAZE_ROOT, f"Video{vid}_salient_dataset")


def parse_json_split(json_path: str) -> set:
    """Parse JSON split file and return set of (video_id, frame_name) tuples."""
    entries = set()
    if not os.path.exists(json_path):
        print(f"  WARNING: JSON split file not found: {json_path}")
        return entries

    with open(json_path, "r") as f:
        for line in f:
            line = line.strip().strip('"').strip("'")
            if not line or line.startswith("#"):
                continue
            if "/" in line:
                vid_str, frame_name = line.split("/", 1)
                try:
                    vid = int(vid_str)
                    entries.add((vid, frame_name))
                except ValueError:
                    continue
    return entries


def list_frames_from_json(json_path: str) -> List[Tuple[int, str, str]]:
    """List frames from JSON split file, filtering to existing salient dataset frames."""
    entries = parse_json_split(json_path)
    out = []

    for vid, frame_name in entries:
        vdir = video_dir(vid)
        img_dir = os.path.join(vdir, "images")
        lbl_dir = os.path.join(vdir, "labels")

        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue

        base_name = os.path.splitext(frame_name)[0]
        img_jpg = os.path.join(img_dir, base_name + ".jpg")
        img_png = os.path.join(img_dir, base_name + ".png")
        lbl_path = os.path.join(lbl_dir, base_name + ".txt")

        if os.path.isfile(img_jpg) and os.path.isfile(lbl_path):
            out.append((vid, img_jpg, lbl_path))
        elif os.path.isfile(img_png) and os.path.isfile(lbl_path):
            out.append((vid, img_png, lbl_path))

    out.sort(key=lambda x: (x[0], os.path.basename(x[1])))
    return out


# ============================================================
# YOLOX-Tiny Detector
# ============================================================

class YOLOXTinyDetector:
    def __init__(self, exp_file: str, weights_path: str):
        self.exp = get_exp(exp_file, None)
        self.model = self.exp.get_model()

        if os.path.exists(weights_path):
            print(f"  Loading YOLOX-Tiny weights: {weights_path}")
            ckpt = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(ckpt["model"])
        else:
            print(f"  WARNING: Weights not found: {weights_path}")

        self.model.to(DEVICE).eval()
        self.num_classes = self.exp.num_classes

        # Build COCO to Our class mapping
        self.coco_to_our = {}
        for coco_id, coco_name in enumerate(COCO_CLASSES):
            if coco_name in COCO_TO_OUR_NAME:
                our_name = COCO_TO_OUR_NAME[coco_name]
                our_id = OUR_CLASSES.index(our_name)
                self.coco_to_our[coco_id] = our_id

    def detect(self, img_bgr: np.ndarray) -> List[Dict]:
        """
        Run detection and return list of detections.
        Each detection: {cls_id, conf, x1, y1, x2, y2, cx, cy, area_ratio}
        """
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize and pad
        r = min(YOLOX_INPUT_SIZE[0] / orig_h, YOLOX_INPUT_SIZE[1] / orig_w)
        new_h, new_w = int(orig_h * r), int(orig_w * r)
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        padded_img = np.ones((YOLOX_INPUT_SIZE[0], YOLOX_INPUT_SIZE[1], 3), dtype=np.uint8) * 114
        padded_img[:new_h, :new_w] = img_resized

        img_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            outputs = postprocess(outputs, self.num_classes, CONF_TH, NMS_IOU, class_agnostic=False)

        if outputs[0] is None:
            return []

        output = outputs[0].cpu().numpy()
        scale = min(YOLOX_INPUT_SIZE[0] / orig_h, YOLOX_INPUT_SIZE[1] / orig_w)
        output[:, :4] /= scale

        detections = []
        for det in output:
            if len(det) >= 7:
                x1, y1, x2, y2, obj_conf, cls_conf, coco_cls = det[:7]
                conf = float(obj_conf * cls_conf)
            else:
                x1, y1, x2, y2, conf, coco_cls = det[:6]
                conf = float(conf)

            coco_cls = int(coco_cls)
            if coco_cls not in self.coco_to_our:
                continue

            our_cls = self.coco_to_our[coco_cls]

            # Clip boxes to image
            x1 = float(np.clip(x1, 0, orig_w - 1))
            y1 = float(np.clip(y1, 0, orig_h - 1))
            x2 = float(np.clip(x2, 0, orig_w - 1))
            y2 = float(np.clip(y2, 0, orig_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            # Compute center and area
            cx = (x1 + x2) / 2.0 / orig_w  # Normalized [0, 1]
            cy = (y1 + y2) / 2.0 / orig_h
            bw = (x2 - x1) / orig_w
            bh = (y2 - y1) / orig_h
            area_ratio = bw * bh

            detections.append({
                "cls_id": our_cls,
                "conf": conf,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "cx": cx, "cy": cy,
                "area_ratio": area_ratio,
            })

        return detections


# ============================================================
# Classical CV Saliency (Spectral Residual) + bbox pooling
# ============================================================

def compute_sr_saliency_map(img_bgr: np.ndarray) -> np.ndarray:
    """
    Compute Spectral Residual saliency map.
    - If OpenCV contrib saliency module exists, use it.
    - Otherwise, fallback to a NumPy FFT implementation.
    Returns float32 map normalized to [0, 1] with shape (H, W).
    """
    # Preferred: OpenCV contrib (cv2.saliency)
    if hasattr(cv2, "saliency") and hasattr(cv2.saliency, "StaticSaliencySpectralResidual_create"):
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, sal_map = sal.computeSaliency(img_bgr)  # float map
        if ok and sal_map is not None:
            sal_map = sal_map.astype(np.float32)
            if SALIENCY_BLUR and SALIENCY_BLUR > 1 and SALIENCY_BLUR % 2 == 1:
                sal_map = cv2.GaussianBlur(sal_map, (SALIENCY_BLUR, SALIENCY_BLUR), 0)
            mn, mx = float(sal_map.min()), float(sal_map.max())
            return ((sal_map - mn) / (mx - mn + SALIENCY_EPS)).astype(np.float32)

    # Fallback: NumPy FFT Spectral Residual
    return _spectral_residual_saliency_numpy(img_bgr)


def _spectral_residual_saliency_numpy(img_bgr: np.ndarray) -> np.ndarray:
    """
    NumPy implementation of Spectral Residual saliency (Hou & Zhang style).
    Steps:
      - FFT -> log amplitude + phase
      - Smooth log amplitude (mean filter)
      - Spectral residual = logA - smooth(logA)
      - Reconstruct -> IFFT -> energy map
      - Blur + normalize
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # FFT
    F = np.fft.fft2(gray)
    A = np.abs(F)
    P = np.angle(F)

    # Log amplitude and its local average
    logA = np.log(A + SALIENCY_EPS).astype(np.float32)
    logA_avg = cv2.blur(logA, (3, 3))  # simple average filter

    # Spectral residual
    R = logA - logA_avg

    # Reconstruct spectrum: exp(R + jP) = exp(R) * (cos(P) + j sin(P))
    expR = np.exp(R).astype(np.float32)
    real = expR * np.cos(P)
    imag = expR * np.sin(P)
    F_res = real + 1j * imag

    # Inverse FFT -> saliency energy
    sal = np.fft.ifft2(F_res)
    sal = np.abs(sal) ** 2
    sal = sal.astype(np.float32)

    # Optional blur
    if SALIENCY_BLUR and SALIENCY_BLUR > 1 and SALIENCY_BLUR % 2 == 1:
        sal = cv2.GaussianBlur(sal, (SALIENCY_BLUR, SALIENCY_BLUR), 0)

    # Normalize to [0,1]
    mn, mx = float(sal.min()), float(sal.max())
    sal = (sal - mn) / (mx - mn + SALIENCY_EPS)
    return sal.astype(np.float32)


def saliency_score_in_box(sal_map01: np.ndarray, box_xyxy: List[float], pooling: str = "p95") -> float:
    """
    Pool saliency map values inside bbox region.
    pooling in {"mean", "max", "p95"}.
    """
    h, w = sal_map01.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    x1i = int(np.clip(np.floor(x1), 0, w - 1))
    y1i = int(np.clip(np.floor(y1), 0, h - 1))
    x2i = int(np.clip(np.ceil(x2),  0, w - 1))
    y2i = int(np.clip(np.ceil(y2),  0, h - 1))
    if x2i <= x1i or y2i <= y1i:
        return 0.0

    roi = sal_map01[y1i:y2i, x1i:x2i]
    if roi.size == 0:
        return 0.0

    if pooling == "mean":
        return float(np.mean(roi))
    if pooling == "max":
        return float(np.max(roi))
    if pooling == "p95":
        return float(np.percentile(roi, 95))
    raise ValueError(f"Unknown pooling: {pooling}")


# ============================================================
# CV-Based Heuristics (geometric priors)
# ============================================================

def score_center_bias(cx: float, cy: float) -> float:
    dist_from_center = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
    max_dist = math.sqrt(0.5 ** 2 + 0.5 ** 2)
    return 1.0 - (dist_from_center / max_dist)


def score_size_bias(area_ratio: float) -> float:
    return min(1.0, area_ratio * 5.0)


def score_bottom_center_bias(cx: float, cy: float) -> float:
    target_x, target_y = 0.5, 0.7
    dist = math.sqrt((cx - target_x) ** 2 + (cy - target_y) ** 2)
    max_dist = math.sqrt(0.5 ** 2 + 0.7 ** 2)
    return 1.0 - min(1.0, dist / max_dist)


def score_vertical_position(cy: float) -> float:
    return cy


def apply_cv_postprocessing(
    detections: List[Dict],
    method: str,
    sal_map01: np.ndarray = None
) -> List[Tuple[int, float, List[float]]]:
    """
    Apply post-processing to rescore detections.
    For saliency-based methods, pass sal_map01 (H,W) in [0,1].

    Returns:
        List of (cls_id, score, [x1,y1,x2,y2]) sorted by score desc; keeps top-K.
    """
    results = []

    for det in detections:
        cls_id = det["cls_id"]
        conf = float(det["conf"])
        cx, cy = float(det["cx"]), float(det["cy"])
        area = float(det["area_ratio"])
        box = [float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"])]

        if method == "raw":
            score = conf

        elif method == "center":
            center_score = score_center_bias(cx, cy)
            score = 0.5 * conf + 0.5 * center_score

        elif method == "size":
            size_score = score_size_bias(area)
            score = 0.5 * conf + 0.5 * size_score

        elif method == "bottom_center":
            bc_score = score_bottom_center_bias(cx, cy)
            score = 0.5 * conf + 0.5 * bc_score

        elif method == "vertical":
            vert_score = score_vertical_position(cy)
            score = 0.5 * conf + 0.5 * vert_score

        elif method == "combined":
            center_score = score_center_bias(cx, cy)
            size_score = score_size_bias(area)
            bc_score = score_bottom_center_bias(cx, cy)

            # weights: conf=0.4, bottom_center=0.3, size=0.2, center=0.1
            score = (conf * 0.4 +
                     bc_score * 0.3 +
                     size_score * 0.2 +
                     center_score * 0.1)

        elif method == "sr_saliency":
            if sal_map01 is None:
                raise ValueError("sr_saliency requires sal_map01.")
            s_score = saliency_score_in_box(sal_map01, box, pooling=SALIENCY_POOLING)
            score = s_score

        elif method == "sr_conf_saliency":
            if sal_map01 is None:
                raise ValueError("sr_conf_saliency requires sal_map01.")
            s_score = saliency_score_in_box(sal_map01, box, pooling=SALIENCY_POOLING)
            # Normalize conf to [0,1] if needed (YOLOX conf already in [0,1])
            score = SALIENCY_ALPHA * conf + (1.0 - SALIENCY_ALPHA) * s_score

        else:
            raise ValueError(f"Unknown method: {method}")

        results.append((cls_id, float(score), box))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:EVAL_TOPK]


# ============================================================
# Ground Truth Parsing
# ============================================================

def parse_attended_gt(label_path: str) -> List[Tuple[int, List[float]]]:
    attended = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 6 or parts[0] != "A":
                continue
            cls_id = int(parts[1])
            if 0 <= cls_id < NUM_CLASSES:
                cx, cy, w, h = map(float, parts[2:6])
                attended.append((cls_id, [cx, cy, w, h]))
    return attended


def cxcywhn_to_xyxy(cx, cy, w, h, W, H) -> List[float]:
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    return [x1, y1, x2, y2]


def compute_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
    area2 = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))

    return inter / (area1 + area2 - inter + 1e-9)


# ============================================================
# Evaluation
# ============================================================

def evaluate_method(
    detector: YOLOXTinyDetector,
    frames: List[Tuple[int, str, str]],
    method: str,
    split_name: str
) -> float:
    """
    Evaluate a post-processing method on a split. Returns mAP@0.5 (11-pt).
    """
    all_preds = []
    all_gts = []

    for vid, img_path, lbl_path in frames:
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # Compute saliency map only for saliency-based methods (speed)
        sal_map01 = None
        if method in ("sr_saliency", "sr_conf_saliency"):
            sal_map01 = compute_sr_saliency_map(img)  # (H,W) float in [0,1]

        # Get detections
        detections = detector.detect(img)

        # Apply post-processing
        preds = apply_cv_postprocessing(detections, method, sal_map01=sal_map01)
        all_preds.append(preds)

        # Parse ground truth
        attended = parse_attended_gt(lbl_path)
        gts = []
        for cls_id, (cx, cy, bw, bh) in attended:
            box = cxcywhn_to_xyxy(cx, cy, bw, bh, w, h)
            gts.append((cls_id, box))
        all_gts.append(gts)

    # Compute mAP@0.5 (per-class AP then mean)
    ap_per_class = []
    for cls in range(NUM_CLASSES):
        cls_preds = []
        for img_idx, preds in enumerate(all_preds):
            for pred_cls, conf, box in preds:
                if pred_cls == cls:
                    cls_preds.append((img_idx, conf, box))

        cls_gts = defaultdict(list)
        num_gt = 0
        for img_idx, gts in enumerate(all_gts):
            for gt_cls, box in gts:
                if gt_cls == cls:
                    cls_gts[img_idx].append(box)
                    num_gt += 1

        if num_gt == 0:
            continue

        cls_preds.sort(key=lambda x: x[1], reverse=True)

        tp = np.zeros(len(cls_preds), dtype=np.float32)
        fp = np.zeros(len(cls_preds), dtype=np.float32)
        matched = defaultdict(set)

        for i, (img_idx, conf, pred_box) in enumerate(cls_preds):
            gt_boxes = cls_gts.get(img_idx, [])
            if len(gt_boxes) == 0:
                fp[i] = 1
                continue

            max_iou = 0.0
            max_j = -1
            for j, gt_box in enumerate(gt_boxes):
                iou = compute_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_j = j

            if max_iou >= 0.5 and max_j not in matched[img_idx]:
                tp[i] = 1
                matched[img_idx].add(max_j)
            else:
                fp[i] = 1

        if len(cls_preds) == 0:
            continue

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / max(num_gt, 1)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)

        # 11-point interpolation AP
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0.0
            else:
                p = float(np.max(precisions[recalls >= t]))
            ap += p / 11.0

        ap_per_class.append((cls, float(ap), int(num_gt)))

    mAP = float(np.mean([ap for _, ap, _ in ap_per_class])) if len(ap_per_class) > 0 else 0.0

    print(f"  [{split_name}] {method:18s} mAP@0.5: {mAP:.4f}")
    for cls, ap, num_gt in ap_per_class:
        print(f"    {OUR_CLASSES[cls]:15s}: AP={ap:.4f} (GT={num_gt})")

    return mAP


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("YOLOX-Tiny + Classical CV Post-Processing Baseline")
    print("(No Teacher, No KD - Geometric Priors + Spectral Residual Saliency)")
    print("=" * 80)

    # Load data
    print("\n[Data Loading] JSON-based splits")
    train_frames = list_frames_from_json(TRAIN_JSON)
    val_frames = list_frames_from_json(VALID_JSON)
    test_frames = list_frames_from_json(TEST_JSON)
    print(f"  TRAIN: {len(train_frames)} frames")
    print(f"  VAL:   {len(val_frames)} frames")
    print(f"  TEST:  {len(test_frames)} frames")

    # Load detector
    print("\n[Model] Loading YOLOX-Tiny")
    detector = YOLOXTinyDetector(YOLOX_EXP_FILE, YOLOX_WEIGHTS)

    # Methods to evaluate (added saliency-based methods)
    methods = [
        "raw",
        "center",
        "size",
        "bottom_center",
        "vertical",
        "combined",
        "sr_saliency",
        "sr_conf_saliency",
    ]

    # Evaluate on VAL
    print("\n" + "=" * 80)
    print("VALIDATION SET RESULTS")
    print("=" * 80)

    val_results = {}
    for method in methods:
        mAP = evaluate_method(detector, val_frames, method, "VAL")
        val_results[method] = mAP
        print()

    # Evaluate on TEST
    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)

    test_results = {}
    for method in methods:
        mAP = evaluate_method(detector, test_frames, method, "TEST")
        test_results[method] = mAP
        print()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: CV Baseline Methods")
    print("=" * 80)
    print(f"\n{'Method':<22} {'VAL mAP@0.5':<15} {'TEST mAP@0.5':<15}")
    print("-" * 60)
    for method in methods:
        print(f"{method:<22} {val_results[method]:<15.4f} {test_results[method]:<15.4f}")

    best_val_method = max(val_results, key=val_results.get)
    best_test_method = max(test_results, key=test_results.get)

    print("\n" + "-" * 60)
    print(f"Best VAL method:  {best_val_method} ({val_results[best_val_method]:.4f})")
    print(f"Best TEST method: {best_test_method} ({test_results[best_test_method]:.4f})")

    # Save results
    summary_path = os.path.join(OUTPUT_DIR, "cv_baseline_summary.txt")
    with open(summary_path, "w") as f:
        f.write("YOLOX-Tiny + Classical CV Post-Processing Baseline\n")
        f.write("=" * 60 + "\n\n")
        f.write("Methods:\n")
        f.write("  - raw: detection confidence only\n")
        f.write("  - center: conf + center bias\n")
        f.write("  - size: conf + size bias\n")
        f.write("  - bottom_center: conf + bottom-center prior\n")
        f.write("  - vertical: conf + vertical prior\n")
        f.write("  - combined: weighted geometric priors\n")
        f.write("  - sr_saliency: Spectral Residual saliency pooled inside bbox\n")
        f.write("  - sr_conf_saliency: combine conf + SR saliency pooled in bbox\n\n")
        f.write(f"SALIENCY_POOLING={SALIENCY_POOLING}, SALIENCY_ALPHA={SALIENCY_ALPHA}\n\n")
        f.write(f"{'Method':<22} {'VAL mAP@0.5':<15} {'TEST mAP@0.5':<15}\n")
        f.write("-" * 60 + "\n")
        for method in methods:
            f.write(f"{method:<22} {val_results[method]:<15.4f} {test_results[method]:<15.4f}\n")
        f.write("\n")
        f.write(f"Best VAL method:  {best_val_method} ({val_results[best_val_method]:.4f})\n")
        f.write(f"Best TEST method: {best_test_method} ({test_results[best_test_method]:.4f})\n")

    print(f"\nResults saved to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

