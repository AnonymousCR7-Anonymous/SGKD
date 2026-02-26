"""
TrafficGaze – Saliency→Attended-Object Detection via KD (teacher-free inference) Rescoring
======================================================================================

Goal:
- Training: use a frozen saliency teacher (ViT-B/16 DINO) ONLY to guide training of a small MLP rescoring head.
- Inference / deployment: teacher is NOT used.
    => Inference = YOLOX-Tiny detector + MLP rescoring head (re-rank + top-K / threshold)

Modified in this version:
1) Data splitting is now JSON-based (train.json / valid.json / test.json),
   matching your saliency-map-estimation split files.
2) The old fixed video-ID split (TRAIN_VIDS / VAL_VIDS / TEST_VIDS) is removed.
3) Everything else in the KD / teacher-free inference logic is preserved.
"""

import os
import sys
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

# Add YOLOX to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolox.exp import get_exp
from yolox.utils import postprocess


# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAFFICGAZE_ROOT = ".../TrafficGaze"
TEACHER_CKPT = ".../KD/teacher_saliency_dino_224.pth"  

# YOLOX-Tiny configuration
YOLOX_EXP_FILE = "exps/default/yolox_tiny.py"
YOLOX_WEIGHTS = "yolox_tiny.pth"  # Pretrained weights (REQUIRED for detection to work!)

# JSON split files (matches saliency map estimation)
JSON_SPLIT_DIR = ".../Traffic_Gaze"
TRAIN_JSON = os.path.join(JSON_SPLIT_DIR, "train.json")
VALID_JSON = os.path.join(JSON_SPLIT_DIR, "valid.json")
TEST_JSON  = os.path.join(JSON_SPLIT_DIR, "test.json")

# -----------------------------
# Class handling
# -----------------------------
CLASS_MODE = "mapped"  # "mapped" or "agnostic"
NUM_CLASSES_MODE = 7   # 7 or 5

# 7-class mode (all classes)
OUR_CLASSES_7 = [
    "people",          # 0
    "car",             # 1
    "motorcycle",      # 2
    "traffic-light",   # 3
    "traffic-sign",    # 4
    "bus",             # 5
    "truck",           # 6
]
OUR_TO_ID_7 = {n: i for i, n in enumerate(OUR_CLASSES_7)}

# 5-class mode (exclude bus/truck)
OUR_CLASSES_5 = [
    "people",          # 0
    "car",             # 1
    "motorcycle",      # 2
    "traffic-light",   # 3
    "traffic-sign",    # 4
]
OUR_TO_ID_5 = {n: i for i, n in enumerate(OUR_CLASSES_5)}

if NUM_CLASSES_MODE == 5:
    OUR_CLASSES = OUR_CLASSES_5
    OUR_TO_ID = OUR_TO_ID_5
else:
    OUR_CLASSES = OUR_CLASSES_7
    OUR_TO_ID = OUR_TO_ID_7

# COCO names -> OUR names
COCO_TO_OUR_NAME_7 = {
    "person": "people",
    "car": "car",
    "motorcycle": "motorcycle",
    "traffic light": "traffic-light",
    "stop sign": "traffic-sign",
    "bus": "bus",
    "truck": "truck",
}
COCO_TO_OUR_NAME_5 = {
    "person": "people",
    "car": "car",
    "motorcycle": "motorcycle",
    "traffic light": "traffic-light",
    "stop sign": "traffic-sign",
}
COCO_TO_OUR_NAME = COCO_TO_OUR_NAME_5 if NUM_CLASSES_MODE == 5 else COCO_TO_OUR_NAME_7

# COCO class IDs (YOLOX uses COCO dataset)
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

# -----------------------------
# Detection candidate rules
# -----------------------------
CONF_TH_RAW = 0.35
NMS_IOU = 0.60

MAX_DET = 300
TOPK = 300
IOU_POS = 0.50

# YOLOX-Tiny input size
YOLOX_INPUT_SIZE = (416, 416)

# -----------------------------
# KD head training
# -----------------------------
EPOCHS = 15
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4

KD_LOSS = "bce_posweight"  # "bce_posweight" or "focal"
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Teacher distillation (TRAINING ONLY)
USE_TEACHER_DISTILL = True
LAMBDA_DISTILL = 0.5  # weight of teacher soft loss term (MSE)

# -----------------------------
# Inference selection
# -----------------------------
AUTO_TUNE_KD_SELECTION = True
KD_TOPK_GRID = [10, 30, 50, 100, 200, 300]
KD_TH_GRID   = [None, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

KD_TOPK_INFER = 100
KD_TH_INFER   = 0.20

# -----------------------------
# Outputs
# -----------------------------
RUN_DIR = os.path.join(TRAFFICGAZE_ROOT, "kd_yolox_tiny_runs_teacher_free_infer")
VIS_DIR = os.path.join(TRAFFICGAZE_ROOT, "kd_yolox_tiny_vis_teacher_free_infer")
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Teacher model definition (used ONLY in training + optional ablations)
# ============================================================
class SaliencyTeacher(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224.dino",
        img_size: int = 224,
        pretrained: bool = False,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = 16

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=""
        )
        embed_dim = self.backbone.num_features

        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1)
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0

        tokens = self.backbone.forward_features(x)
        tokens = tokens[:, 1:, :]
        B, N, Cc = tokens.shape

        h = H // self.patch_size
        w = W // self.patch_size
        feat = tokens.transpose(1, 2).reshape(B, Cc, h, w)
        feat = self.decoder(feat)
        sal_logits = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
        return sal_logits


def build_teacher(ckpt_path: str) -> nn.Module:
    print(f"[Teacher] Loading ckpt: {ckpt_path}")
    teacher = SaliencyTeacher(
        backbone_name="vit_base_patch16_224.dino",
        img_size=224,
        pretrained=False,
        freeze_backbone=True
    )
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    teacher.load_state_dict(sd, strict=True)

    teacher.to(DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("[Teacher] Ready (eval, frozen).")
    return teacher


@torch.no_grad()
def teacher_saliency_map(
    teacher: nn.Module,
    img_bgr: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)

    x = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x_t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    logits = teacher(x_t)
    sm = torch.sigmoid(logits)[0, 0].float().cpu().numpy().astype(np.float32)

    if gamma != 1.0:
        sm = np.power(np.clip(sm, 0.0, 1.0), gamma).astype(np.float32)

    sm = cv2.resize(sm, (W, H), interpolation=cv2.INTER_CUBIC)
    sm = np.clip(sm, 0.0, 1.0).astype(np.float32)
    return sm


# ============================================================
# YOLOX Student Model Wrapper
# ============================================================
class YOLOXStudent:
    def __init__(self, exp_file: str, weights_path: Optional[str] = None):
        self.exp = get_exp(exp_file, None)
        self.model = self.exp.get_model()

        if weights_path is not None and os.path.exists(weights_path):
            print(f"[YOLOX] Loading weights from {weights_path}")
            ckpt = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(ckpt["model"])
        else:
            print("[YOLOX] Using model without pretrained weights (will train from scratch)")

        self.model.to(DEVICE)
        self.model.eval()

        self.num_classes = self.exp.num_classes
        self.conf_thre = 0.01
        self.nms_thre = NMS_IOU

    def predict(self, img_bgr: np.ndarray, conf_thre: float = None, max_det: int = 300):
        """
        Returns postprocess output rows typically:
          [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
        """
        if conf_thre is None:
            conf_thre = self.conf_thre

        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        input_h, input_w = YOLOX_INPUT_SIZE
        r = min(input_h / orig_h, input_w / orig_w)
        new_h, new_w = int(orig_h * r), int(orig_w * r)

        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded_img = np.ones((input_h, input_w, 3), dtype=np.uint8) * 114
        padded_img[:new_h, :new_w] = img_resized

        img_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            outputs = postprocess(
                outputs,
                self.num_classes,
                conf_thre,
                self.nms_thre,
                class_agnostic=False
            )

        if outputs[0] is None:
            return []

        output = outputs[0].cpu().numpy()

        scale = min(input_h / orig_h, input_w / orig_w)  # == r
        output[:, :4] /= scale

        # Keep at most max_det (rough sort by obj_conf)
        if len(output) > max_det:
            indices = np.argsort(-output[:, 4])[:max_det]
            output = output[indices]

        return output


# ============================================================
# Data utilities (JSON-based splits)
# ============================================================
def video_dir(vid: int) -> str:
    return os.path.join(TRAFFICGAZE_ROOT, f"Video{vid}_salient_dataset")


def parse_json_split(json_path: str) -> set:
    """
    Parse split file and return a set of (video_id, frame_name) tuples.

    Expected line format (matching your splitting code):
        3/000123.jpg
        7/001045.png

    Note:
    - This parser treats the split files as line-based text entries.
    - If your files are true JSON arrays/objects, let me know and I will switch
      this to json.load(...) parsing.
    """
    entries = set()

    if not os.path.exists(json_path):
        print(f"[Split] WARNING: JSON split file not found: {json_path}")
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
    """
    List frames from JSON split file, filtering to existing salient dataset frames.
    Returns:
        [(vid, img_path, lbl_path), ...]
    """
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


def cxcywhn_to_xyxy_px(cx, cy, w, h, W, H):
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    x1 = float(np.clip(x1, 0, W - 1))
    y1 = float(np.clip(y1, 0, H - 1))
    x2 = float(np.clip(x2, 0, W - 1))
    y2 = float(np.clip(y2, 0, H - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def parse_attended_gt(
    label_path: str,
    W: int,
    H: int,
    class_mode: str,
    num_classes: int = 7
) -> List[Tuple[int, List[float]]]:
    gts = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 6:
                continue

            tag, cls_s, cx_s, cy_s, w_s, h_s = parts
            if tag != "A":
                continue

            cls = int(cls_s)

            # For 5-class mode, skip bus (5) and truck (6)
            if num_classes == 5 and cls in [5, 6]:
                continue

            if class_mode == "agnostic":
                cls = 0

            cx = float(cx_s)
            cy = float(cy_s)
            ww = float(w_s)
            hh = float(h_s)
            gts.append((cls, cxcywhn_to_xyxy_px(cx, cy, ww, hh, W, H)))

    return gts


# ============================================================
# Class mapping: COCO -> OUR
# ============================================================
def build_coco_to_our_id(num_classes: int = 7) -> Dict[int, int]:
    mapping = COCO_TO_OUR_NAME_5 if num_classes == 5 else COCO_TO_OUR_NAME_7
    our_to_id = OUR_TO_ID_5 if num_classes == 5 else OUR_TO_ID_7

    coco_to_our = {}
    for coco_id, coco_name in enumerate(COCO_CLASSES):
        coco_name_norm = coco_name.strip().lower().replace("_", " ")
        if coco_name_norm in mapping:
            our_name = mapping[coco_name_norm]
            coco_to_our[int(coco_id)] = our_to_id[our_name]
    return coco_to_our


# ============================================================
# Student predictions
# ============================================================
@torch.no_grad()
def yolox_predict(
    model: YOLOXStudent,
    img_bgr: np.ndarray,
    class_mode: str,
    coco_to_our_id: Optional[Dict[int, int]],
    conf_th: float,
    max_det: int,
    mode: str,
) -> List[Tuple[int, float, List[float]]]:
    # low conf for "proposal" to keep many candidates
    pred_conf = float(conf_th) if mode == "raw" else 0.001

    detections = model.predict(img_bgr, conf_thre=pred_conf, max_det=max_det)
    if len(detections) == 0:
        return []

    out = []
    for det in detections:
        # Typical: [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
        if len(det) >= 7:
            x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det[:7]
            conf = float(obj_conf * cls_conf)
        else:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            conf = float(conf)

        if conf < conf_th:
            continue

        cls_id = int(cls_id)

        if class_mode == "mapped":
            if coco_to_our_id is None:
                raise RuntimeError("coco_to_our_id is required for class_mode='mapped'")
            if cls_id not in coco_to_our_id:
                continue
            c = int(coco_to_our_id[cls_id])
        else:
            c = 0

        out.append((c, conf, [float(x1), float(y1), float(x2), float(y2)]))

    return out


# ============================================================
# IoU + AP
# ============================================================
def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


def ap_from_pr(prec: np.ndarray, rec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@dataclass
class PredBox:
    cls: int
    score: float
    xyxy: List[float]
    is_tp: bool = False
    matched_gt_idx: int = -1


def compute_map50(
    preds_by_image: List[List[PredBox]],
    gts_by_image: List[List[Tuple[int, List[float]]]],
    class_mode: str
) -> Tuple[float, Dict[int, float], int]:
    if class_mode == "agnostic":
        classes = [0]
    else:
        s = set()
        for gts in gts_by_image:
            for c, _ in gts:
                s.add(c)
        classes = sorted(list(s))

    ap_per_cls = {}
    total_gt = 0

    for c in classes:
        all_preds = []
        gt_count = 0

        for img_i, (preds, gts) in enumerate(zip(preds_by_image, gts_by_image)):
            if class_mode == "agnostic":
                gt_count += len(gts)
                for p in preds:
                    all_preds.append((img_i, p))
            else:
                gt_c = [(gc, gb) for (gc, gb) in gts if gc == c]
                gt_count += len(gt_c)
                for p in preds:
                    if p.cls == c:
                        all_preds.append((img_i, p))

        total_gt += gt_count
        if gt_count == 0:
            continue

        all_preds.sort(key=lambda x: -x[1].score)

        used = {}
        for i in range(len(gts_by_image)):
            if class_mode == "agnostic":
                used[i] = np.zeros(len(gts_by_image[i]), dtype=bool)
            else:
                used[i] = np.zeros(sum(1 for gc, _ in gts_by_image[i] if gc == c), dtype=bool)

        tp = []
        fp = []

        for (img_i, p) in all_preds:
            if class_mode == "agnostic":
                gt_boxes = [gb for (_, gb) in gts_by_image[img_i]]
            else:
                gt_boxes = [gb for (gc, gb) in gts_by_image[img_i] if gc == c]

            if len(gt_boxes) == 0:
                fp.append(1.0)
                tp.append(0.0)
                continue

            best_iou = 0.0
            best_j = -1
            for j, gb in enumerate(gt_boxes):
                iou = iou_xyxy(p.xyxy, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= IOU_POS and best_j >= 0 and (not used[img_i][best_j]):
                used[img_i][best_j] = True
                tp.append(1.0)
                fp.append(0.0)
            else:
                fp.append(1.0)
                tp.append(0.0)

        tp = np.array(tp, dtype=np.float32)
        fp = np.array(fp, dtype=np.float32)
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / (gt_count + 1e-9)
        prec = tp_cum / (tp_cum + fp_cum + 1e-9)
        ap = ap_from_pr(prec, rec)
        ap_per_cls[c] = ap

    if len(ap_per_cls) == 0:
        return 0.0, {}, total_gt
    map50 = float(np.mean(list(ap_per_cls.values())))
    return map50, ap_per_cls, total_gt


# ============================================================
# Teacher helper (training distillation + optional baseline)
# ============================================================
def saliency_energy_in_box(sal: np.ndarray, xyxy: List[float], mode: str = "p95") -> float:
    H, W = sal.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    patch = sal[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0

    if mode == "max":
        return float(patch.max())
    if mode == "mean":
        return float(patch.mean())
    return float(np.percentile(patch, 95))


def box_features_from_xyxy(conf: float, xyxy: List[float], W: int, H: int) -> Tuple[float, float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = ((x1 + x2) * 0.5) / (W + 1e-9)
    cy = ((y1 + y2) * 0.5) / (H + 1e-9)
    area = (bw * bh) / ((W * H) + 1e-9)
    aspect = (bw / (W + 1e-9)) / ((bh / (H + 1e-9)) + 1e-9)  # normalized w/h
    return float(conf), float(cx), float(cy), float(area), float(aspect)


# ============================================================
# KD Rescoring head (teacher-free inference features)
# ============================================================
class RescoreHead(nn.Module):
    """
    Inputs:
      - conf
      - cx, cy
      - area
      - aspect ratio
      - class embedding (optional via cls_idx)

    Output:
      - score in [0,1] = sigmoid(logit)
    """
    def __init__(self, num_classes: int, hidden: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.cls_emb = nn.Embedding(num_classes, 16)
        self.mlp = nn.Sequential(
            nn.Linear(5 + 16, hidden),  # 5 numeric + 16 class emb
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, conf, cx, cy, area, aspect, cls_idx):
        emb = self.cls_emb(cls_idx)
        x = torch.stack([conf, cx, cy, area, aspect], dim=1)
        x = torch.cat([x, emb], dim=1)
        logits = self.mlp(x).squeeze(1)
        score = torch.sigmoid(logits)
        return score, logits


def binary_focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    w = alpha * targets + (1 - alpha) * (1 - targets)
    loss = w * (1 - pt).pow(gamma) * bce
    return loss.mean()


# ============================================================
# Visualization
# ============================================================
def draw_boxes(img_bgr, boxes: List[Tuple[List[float], str]], color=(0, 0, 255)):
    out = img_bgr.copy()
    for xyxy, text in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out, text, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
    return out


def save_eval_overlay(out_path, img_bgr, preds: List[PredBox], gts: List[Tuple[int, List[float]]], title: str, class_mode: str):
    used = [False] * len(gts)
    preds_sorted = sorted(preds, key=lambda p: -p.score)

    for p in preds_sorted:
        best_iou, best_j = 0.0, -1
        for j, (gc, gb) in enumerate(gts):
            if class_mode != "agnostic" and gc != p.cls:
                continue
            i = iou_xyxy(p.xyxy, gb)
            if i > best_iou:
                best_iou, best_j = i, j
        if best_iou >= IOU_POS and best_j >= 0 and (not used[best_j]):
            used[best_j] = True
            p.is_tp = True
            p.matched_gt_idx = best_j
        else:
            p.is_tp = False

    fn = [j for j, u in enumerate(used) if not u]
    canvas = img_bgr.copy()

    gt_boxes = [(gb, f"GT c={gc}") for (gc, gb) in gts]
    canvas = draw_boxes(canvas, gt_boxes, color=(0, 255, 0))

    tp_boxes, fp_boxes = [], []
    for p in preds_sorted:
        if p.is_tp:
            tp_boxes.append((p.xyxy, f"TP c={p.cls} s={p.score:.2f}"))
        else:
            fp_boxes.append((p.xyxy, f"FP c={p.cls} s={p.score:.2f}"))
    canvas = draw_boxes(canvas, tp_boxes, color=(255, 0, 0))
    canvas = draw_boxes(canvas, fp_boxes, color=(0, 0, 255))

    fn_boxes = []
    for j in fn:
        gc, gb = gts[j]
        fn_boxes.append((gb, f"FN c={gc}"))
    canvas = draw_boxes(canvas, fn_boxes, color=(0, 255, 255))

    cv2.putText(
        canvas, title, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not out_path.lower().endswith(".jpg"):
        out_path = out_path + ".jpg"
    cv2.imwrite(out_path, canvas)


# ============================================================
# Build per-image preds
# ============================================================
def build_preds_for_split(
    student: YOLOXStudent,
    frames: List[Tuple[int, str, str]],
    score_mode: str,
    class_mode: str,
    coco_to_our: Optional[Dict[int, int]],
    head: Optional[RescoreHead] = None,
    teacher: Optional[nn.Module] = None,      # used ONLY if score_mode == "teacher" baseline
    sal_pool: str = "p95",
    topk: int = TOPK,
    score_th: Optional[float] = None,
    num_classes: int = 7,
) -> Tuple[List[List[PredBox]], List[List[Tuple[int, List[float]]]]]:
    """
    score_mode:
      - "raw"     : YOLOX confidence
      - "teacher" : (ablation) teacher saliency energy per box (requires teacher at eval time)
      - "kd"      : teacher-free inference: YOLOX proposals + MLP(head) using ONLY student/box features
    """
    preds_all = []
    gts_all = []

    for (vid, img_path, lbl_path) in frames:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            preds_all.append([])
            gts_all.append([])
            continue

        H, W = img.shape[:2]
        gts = parse_attended_gt(lbl_path, W, H, class_mode=class_mode, num_classes=num_classes)

        if score_mode == "raw":
            dets = yolox_predict(
                model=student,
                img_bgr=img,
                class_mode=class_mode,
                coco_to_our_id=coco_to_our,
                conf_th=CONF_TH_RAW,
                max_det=MAX_DET,
                mode="raw",
            )
        else:
            dets = yolox_predict(
                model=student,
                img_bgr=img,
                class_mode=class_mode,
                coco_to_our_id=coco_to_our,
                conf_th=0.0,
                max_det=MAX_DET,
                mode="proposal",
            )

        sal = None
        if score_mode == "teacher":
            if teacher is None:
                raise RuntimeError("teacher must be provided for score_mode='teacher'")
            sal = teacher_saliency_map(teacher, img)

        preds = []
        for (c, conf, xyxy) in dets:
            if score_mode == "raw":
                s = conf

            elif score_mode == "teacher":
                s = saliency_energy_in_box(sal, xyxy, mode=sal_pool)

            elif score_mode == "kd":
                if head is None:
                    raise RuntimeError("head must be provided for score_mode='kd'")
                conf_f, cx, cy, area, aspect = box_features_from_xyxy(conf, xyxy, W, H)
                with torch.no_grad():
                    conf_t = torch.tensor([conf_f], device=DEVICE, dtype=torch.float32)
                    cx_t   = torch.tensor([cx], device=DEVICE, dtype=torch.float32)
                    cy_t   = torch.tensor([cy], device=DEVICE, dtype=torch.float32)
                    area_t = torch.tensor([area], device=DEVICE, dtype=torch.float32)
                    asp_t  = torch.tensor([aspect], device=DEVICE, dtype=torch.float32)
                    cls_t  = torch.tensor([c], device=DEVICE, dtype=torch.long)
                    score, _ = head(conf_t, cx_t, cy_t, area_t, asp_t, cls_t)
                    s = float(score.item())
            else:
                raise ValueError(score_mode)

            preds.append(PredBox(cls=c, score=float(s), xyxy=xyxy))

        preds = sorted(preds, key=lambda p: -p.score)
        if score_th is not None:
            th = float(score_th)
            preds = [p for p in preds if p.score >= th]
        preds = preds[:int(topk)]

        preds_all.append(preds)
        gts_all.append(gts)

    return preds_all, gts_all


def tune_kd_selection_on_val(
    student: YOLOXStudent,
    head: RescoreHead,
    val_frames: List[Tuple[int, str, str]],
    class_mode: str,
    coco_to_our: Optional[Dict[int, int]],
    k_grid: List[int],
    th_grid: List[Optional[float]],
    num_classes: int,
) -> Tuple[int, Optional[float], float]:
    best_map = -1.0
    best_k = None
    best_th = None

    for k in k_grid:
        for th in th_grid:
            val_preds_kd, val_gts = build_preds_for_split(
                student=student,
                frames=val_frames,
                score_mode="kd",
                class_mode=class_mode,
                coco_to_our=coco_to_our,
                head=head,
                teacher=None,          # teacher-free inference
                sal_pool="p95",
                topk=k,
                score_th=th,
                num_classes=num_classes,
            )
            map50, _, _ = compute_map50(val_preds_kd, val_gts, class_mode=class_mode)
            print(f"[Tune KD] K={k:3d} | th={th} | VAL mAP@0.5={map50:.4f}")

            if map50 > best_map:
                best_map = map50
                best_k = k
                best_th = th

    print(f"[Tune KD] BEST: K={best_k} | th={best_th} | VAL mAP@0.5={best_map:.4f}")
    return int(best_k), best_th, float(best_map)


# ============================================================
# KD training samples
# ============================================================
@dataclass
class Sample:
    conf: float
    cx: float
    cy: float
    area: float
    aspect: float
    cls: int
    y_tp: float       # hard label from attended GT (A-lines)
    y_sal: float      # soft label from teacher saliency in the box (0..1)


def make_kd_samples(
    student: YOLOXStudent,
    frames: List[Tuple[int, str, str]],
    class_mode: str,
    coco_to_our: Optional[Dict[int, int]],
    teacher: Optional[nn.Module] = None,
    max_images: Optional[int] = None,
    sal_pool: str = "p95",
    num_classes: int = 7,
) -> List[Sample]:
    """
    Build training samples from YOLOX proposals.
    Teacher is used ONLY to compute y_sal (soft target) when USE_TEACHER_DISTILL is True.
    """
    out = []
    use_frames = frames[:max_images] if max_images is not None else frames

    for (vid, img_path, lbl_path) in use_frames:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        H, W = img.shape[:2]
        gts = parse_attended_gt(lbl_path, W, H, class_mode=class_mode, num_classes=num_classes)

        dets = yolox_predict(
            model=student,
            img_bgr=img,
            class_mode=class_mode,
            coco_to_our_id=coco_to_our,
            conf_th=0.0,
            max_det=MAX_DET,
            mode="proposal",
        )
        if len(dets) == 0:
            continue

        sal = None
        if USE_TEACHER_DISTILL:
            if teacher is None:
                raise RuntimeError("teacher must be provided when USE_TEACHER_DISTILL=True")
            sal = teacher_saliency_map(teacher, img)

        for (c, conf, xyxy) in dets:
            conf_f, cx, cy, area, aspect = box_features_from_xyxy(conf, xyxy, W, H)

            # Hard TP label from attended GT
            y_tp = 0.0
            for (gc, gb) in gts:
                if class_mode != "agnostic" and gc != c:
                    continue
                if iou_xyxy(xyxy, gb) >= IOU_POS:
                    y_tp = 1.0
                    break

            # Soft label from teacher saliency in this box (optional)
            y_sal = 0.0
            if USE_TEACHER_DISTILL and sal is not None:
                y_sal = float(saliency_energy_in_box(sal, xyxy, mode=sal_pool))

            out.append(Sample(
                conf=conf_f, cx=cx, cy=cy, area=area, aspect=aspect,
                cls=c, y_tp=y_tp, y_sal=y_sal
            ))

    return out


# ============================================================
# Training loop
# ============================================================
def train_kd_head(
    head: RescoreHead,
    train_samples: List[Sample],
    val_frames: List[Tuple[int, str, str]],
    student: YOLOXStudent,
    class_mode: str,
    coco_to_our: Optional[Dict[int, int]],
    num_classes: int,
) -> str:
    head.to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_map = -1.0
    best_path = os.path.join(RUN_DIR, "kd_rescore_head_best.pth")

    ys = np.array([s.y_tp for s in train_samples], dtype=np.float32)
    pos = float(ys.sum())
    neg = float(len(ys) - pos)
    pos_weight = torch.tensor([neg / (pos + 1e-9)], device=DEVICE, dtype=torch.float32)

    print(f"[KD] Samples: {len(train_samples)} | pos={pos:.0f} neg={neg:.0f} pos_weight={pos_weight.item():.3f}")
    print(f"[KD] Teacher distill: {USE_TEACHER_DISTILL} | lambda={LAMBDA_DISTILL}")

    def batch_iter(samples: List[Sample], bs: int):
        random.shuffle(samples)
        for i in range(0, len(samples), bs):
            yield samples[i:i + bs]

    for ep in range(1, EPOCHS + 1):
        head.train()
        losses = []

        for b in batch_iter(train_samples, BATCH_SIZE):
            conf   = torch.tensor([s.conf for s in b], device=DEVICE, dtype=torch.float32)
            cx     = torch.tensor([s.cx for s in b], device=DEVICE, dtype=torch.float32)
            cy     = torch.tensor([s.cy for s in b], device=DEVICE, dtype=torch.float32)
            area   = torch.tensor([s.area for s in b], device=DEVICE, dtype=torch.float32)
            aspect = torch.tensor([s.aspect for s in b], device=DEVICE, dtype=torch.float32)
            cls    = torch.tensor([s.cls for s in b], device=DEVICE, dtype=torch.long)

            y_tp   = torch.tensor([s.y_tp for s in b], device=DEVICE, dtype=torch.float32)
            y_sal  = torch.tensor([s.y_sal for s in b], device=DEVICE, dtype=torch.float32)

            _, logits = head(conf, cx, cy, area, aspect, cls)

            # Main loss: match attended-object GT labels
            if KD_LOSS == "bce_posweight":
                loss_main = F.binary_cross_entropy_with_logits(logits, y_tp, pos_weight=pos_weight)
            elif KD_LOSS == "focal":
                loss_main = binary_focal_loss_with_logits(logits, y_tp, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
            else:
                raise ValueError(f"Unknown KD_LOSS: {KD_LOSS}")

            # Distill loss: make scores consistent with teacher saliency (TRAINING ONLY)
            if USE_TEACHER_DISTILL:
                loss_distill = F.mse_loss(torch.sigmoid(logits), y_sal)
                loss = loss_main + LAMBDA_DISTILL * loss_distill
            else:
                loss = loss_main

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        head.eval()

        # Validation uses TEACHER-FREE inference path
        val_preds, val_gts = build_preds_for_split(
            student=student,
            frames=val_frames,
            score_mode="kd",
            class_mode=class_mode,
            coco_to_our=coco_to_our,
            head=head,
            teacher=None,
            topk=TOPK,
            score_th=None,
            num_classes=num_classes,
        )
        map50, _, _ = compute_map50(val_preds, val_gts, class_mode=class_mode)

        print(f"Epoch {ep:02d}/{EPOCHS} | TrainLoss: {np.mean(losses):.4f} | VAL KD (teacher-free) mAP@0.5: {map50:.4f}")

        if map50 > best_map:
            best_map = map50
            torch.save(head.state_dict(), best_path)
            print(f"[Checkpoint] Saved best head: {best_path} (best VAL mAP@0.5={best_map:.4f})")

    return best_path


# ============================================================
# Evaluation
# ============================================================
def evaluate_and_visualize(
    tag: str,
    frames: List[Tuple[int, str, str]],
    preds_by_img: List[List[PredBox]],
    gts_by_img: List[List[Tuple[int, List[float]]]],
    out_dir: str,
    class_mode: str,
    max_vis: int = 60
) -> float:
    map50, ap_per_cls, total_gt = compute_map50(preds_by_img, gts_by_img, class_mode=class_mode)
    print(f"[{tag}] mAP@0.5: {map50:.4f} | classes_with_gt={len(ap_per_cls)} | total_gt={total_gt}")

    idxs = list(range(len(frames)))
    random.Random(SEED).shuffle(idxs)
    idxs = idxs[:min(max_vis, len(idxs))]

    for k, i in enumerate(idxs):
        vid, img_path, _ = frames[i]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        title = f"{tag} | Video{vid} | {os.path.basename(img_path)}"
        base = f"{k:03d}_V{vid}_{os.path.basename(img_path)}"
        save_path = os.path.join(out_dir, base)
        save_eval_overlay(save_path, img, preds_by_img[i], gts_by_img[i], title, class_mode=class_mode)

    return map50


# ============================================================
# MAIN
# ============================================================
def main():
    global KD_TOPK_INFER, KD_TH_INFER

    print(f"[Device] {DEVICE}")
    print(f"[Class mode] {CLASS_MODE} | NUM_CLASSES_MODE={NUM_CLASSES_MODE}")

    print(f"[Student] Loading YOLOX-Tiny detector")
    student = YOLOXStudent(YOLOX_EXP_FILE, YOLOX_WEIGHTS)

    coco_to_our = None
    if CLASS_MODE == "mapped":
        coco_to_our = build_coco_to_our_id(num_classes=NUM_CLASSES_MODE)
        print(f"[Mapping] COCO->OUR: {coco_to_our}")
        if len(coco_to_our) == 0:
            raise RuntimeError("No class mapping was built.")
    else:
        print("[Mapping] class-agnostic: all detections/GT forced to cls=0")

    # Teacher is loaded for training distillation (and optional teacher baseline)
    teacher = build_teacher(TEACHER_CKPT) if USE_TEACHER_DISTILL else None

    # JSON-based splits
    train_frames = list_frames_from_json(TRAIN_JSON)
    val_frames   = list_frames_from_json(VALID_JSON)
    test_frames  = list_frames_from_json(TEST_JSON)

    print(f"[Split files] train={TRAIN_JSON}")
    print(f"[Split files] val  ={VALID_JSON}")
    print(f"[Split files] test ={TEST_JSON}")
    print(f"[Split] Train frames: {len(train_frames)} | Val frames: {len(val_frames)} | Test frames: {len(test_frames)}")

    SAL_POOL = "p95"

    # ---------------------------
    # Baseline 1: raw detector
    # ---------------------------
    print("\n===== EVAL 1: Student RAW (VAL split) =====")
    val_preds_raw, val_gts = build_preds_for_split(
        student=student,
        frames=val_frames,
        score_mode="raw",
        class_mode=CLASS_MODE,
        coco_to_our=coco_to_our,
        head=None,
        teacher=None,
        topk=TOPK,
        score_th=None,
        num_classes=NUM_CLASSES_MODE,
    )
    vis_raw_dir = os.path.join(VIS_DIR, "VAL_RAW")
    os.makedirs(vis_raw_dir, exist_ok=True)
    map_raw = evaluate_and_visualize("VAL_RAW", val_frames, val_preds_raw, val_gts, vis_raw_dir, class_mode=CLASS_MODE)

    # ---------------------------
    # Optional baseline: teacher-score (NOT deployment)
    # ---------------------------
    print("\n===== EVAL 2 (ABlation): Student proposals scored by TEACHER saliency (VAL split) =====")
    val_preds_teacher, val_gts2 = build_preds_for_split(
        student=student,
        frames=val_frames,
        score_mode="teacher",
        class_mode=CLASS_MODE,
        coco_to_our=coco_to_our,
        head=None,
        teacher=teacher if teacher is not None else build_teacher(TEACHER_CKPT),
        sal_pool=SAL_POOL,
        topk=TOPK,
        score_th=None,
        num_classes=NUM_CLASSES_MODE,
    )
    vis_t_dir = os.path.join(VIS_DIR, "VAL_TEACHER_SCORE")
    os.makedirs(vis_t_dir, exist_ok=True)
    map_teacher = evaluate_and_visualize("VAL_TEACHER_SCORE", val_frames, val_preds_teacher, val_gts2, vis_t_dir, class_mode=CLASS_MODE)

    # ---------------------------
    # Train KD head (teacher used ONLY for distillation targets, not inference)
    # ---------------------------
    print("\n===== TRAIN: KD rescoring head on TRAIN split =====")
    train_samples = make_kd_samples(
        student=student,
        frames=train_frames,
        class_mode=CLASS_MODE,
        coco_to_our=coco_to_our,
        teacher=teacher,          # used only to compute y_sal (soft target)
        max_images=None,
        sal_pool=SAL_POOL,
        num_classes=NUM_CLASSES_MODE,
    )
    print(f"[KD] Train detection-samples: {len(train_samples)}")

    num_head_classes = 1 if CLASS_MODE == "agnostic" else len(OUR_CLASSES)
    head = RescoreHead(num_classes=num_head_classes, hidden=128)

    best_head_path = train_kd_head(
        head=head,
        train_samples=train_samples,
        val_frames=val_frames,
        student=student,
        class_mode=CLASS_MODE,
        coco_to_our=coco_to_our,
        num_classes=NUM_CLASSES_MODE,
    )

    head.load_state_dict(torch.load(best_head_path, map_location="cpu"))
    head.to(DEVICE).eval()
    print(f"[Best head] {best_head_path}")

    # ---------------------------
    # Tune (Top-K, threshold) for TEACHER-FREE inference
    # ---------------------------
    if AUTO_TUNE_KD_SELECTION:
        print("\n===== TUNE: (Top-K, threshold) for KD inference selection on VAL (teacher-free) =====")
        KD_TOPK_INFER, KD_TH_INFER, best_map_sel = tune_kd_selection_on_val(
            student=student,
            head=head,
            val_frames=val_frames,
            class_mode=CLASS_MODE,
            coco_to_our=coco_to_our,
            k_grid=KD_TOPK_GRID,
            th_grid=KD_TH_GRID,
            num_classes=NUM_CLASSES_MODE,
        )
        print(f"[KD Selection] BEST K={KD_TOPK_INFER} | th={KD_TH_INFER} | VAL mAP@0.5={best_map_sel:.4f}")
    else:
        print(f"\n[KD Selection] Using fixed K={KD_TOPK_INFER} | th={KD_TH_INFER}")

    # ---------------------------
    # Evaluate KD (teacher-free inference)
    # ---------------------------
    print("\n===== EVAL 3: Student + TRAINED KD RESCORING HEAD (VAL split) [teacher-free inference] =====")
    val_preds_kd, val_gts3 = build_preds_for_split(
        student=student,
        frames=val_frames,
        score_mode="kd",
        class_mode=CLASS_MODE,
        coco_to_our=coco_to_our,
        head=head,
        teacher=None,   # teacher-free inference
        topk=KD_TOPK_INFER,
        score_th=KD_TH_INFER,
        num_classes=NUM_CLASSES_MODE,
    )
    vis_kd_dir = os.path.join(VIS_DIR, "VAL_KD_RESCORE_TEACHER_FREE")
    os.makedirs(vis_kd_dir, exist_ok=True)
    map_kd = evaluate_and_visualize("VAL_KD_RESCORE_TF", val_frames, val_preds_kd, val_gts3, vis_kd_dir, class_mode=CLASS_MODE)

    # ---------------------------
    # Test split evaluation
    # ---------------------------
    if len(test_frames) > 0:
        print("\n===== TEST: RAW vs TEACHER_SCORE (ablation) vs KD (teacher-free) (TEST split) =====")

        test_raw, test_gts = build_preds_for_split(
            student=student,
            frames=test_frames,
            score_mode="raw",
            class_mode=CLASS_MODE,
            coco_to_our=coco_to_our,
            topk=TOPK,
            score_th=None,
            num_classes=NUM_CLASSES_MODE
        )
        test_teacher, _ = build_preds_for_split(
            student=student,
            frames=test_frames,
            score_mode="teacher",
            class_mode=CLASS_MODE,
            coco_to_our=coco_to_our,
            teacher=teacher if teacher is not None else build_teacher(TEACHER_CKPT),
            sal_pool=SAL_POOL,
            topk=TOPK,
            score_th=None,
            num_classes=NUM_CLASSES_MODE
        )
        test_kd, _ = build_preds_for_split(
            student=student,
            frames=test_frames,
            score_mode="kd",
            class_mode=CLASS_MODE,
            coco_to_our=coco_to_our,
            head=head,
            teacher=None,  # teacher-free inference
            topk=KD_TOPK_INFER,
            score_th=KD_TH_INFER,
            num_classes=NUM_CLASSES_MODE
        )

        test_raw_dir = os.path.join(VIS_DIR, "TEST_RAW")
        test_t_dir   = os.path.join(VIS_DIR, "TEST_TEACHER_SCORE")
        test_kd_dir  = os.path.join(VIS_DIR, "TEST_KD_TEACHER_FREE")
        os.makedirs(test_raw_dir, exist_ok=True)
        os.makedirs(test_t_dir, exist_ok=True)
        os.makedirs(test_kd_dir, exist_ok=True)

        _ = evaluate_and_visualize("TEST_RAW", test_frames, test_raw, test_gts, test_raw_dir, class_mode=CLASS_MODE, max_vis=40)
        _ = evaluate_and_visualize("TEST_TEACHER_SCORE", test_frames, test_teacher, test_gts, test_t_dir, class_mode=CLASS_MODE, max_vis=40)
        _ = evaluate_and_visualize("TEST_KD_TF", test_frames, test_kd, test_gts, test_kd_dir, class_mode=CLASS_MODE, max_vis=40)

    # ---------------------------
    # Save summary
    # ---------------------------
    summary_path = os.path.join(RUN_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("TrafficGaze KD Rescoring (YOLOX-Tiny) - TEACHER-FREE INFERENCE\n")
        f.write(f"CLASS_MODE={CLASS_MODE} NUM_CLASSES_MODE={NUM_CLASSES_MODE}\n")
        f.write(f"CONF_TH_RAW={CONF_TH_RAW} NMS_IOU={NMS_IOU} MAX_DET={MAX_DET} TOPK_DEFAULT={TOPK} IOU_POS={IOU_POS}\n")
        f.write(f"USE_TEACHER_DISTILL={USE_TEACHER_DISTILL} LAMBDA_DISTILL={LAMBDA_DISTILL}\n")
        f.write(f"KD_LOSS={KD_LOSS}\n")
        f.write(f"TRAIN_JSON={TRAIN_JSON}\n")
        f.write(f"VALID_JSON={VALID_JSON}\n")
        f.write(f"TEST_JSON={TEST_JSON}\n")
        f.write(f"Train_frames={len(train_frames)}\n")
        f.write(f"Val_frames={len(val_frames)}\n")
        f.write(f"Test_frames={len(test_frames)}\n")
        f.write(f"KD_SELECTION_AUTO_TUNE={AUTO_TUNE_KD_SELECTION}\n")
        f.write(f"KD_TOPK_INFER={KD_TOPK_INFER}\n")
        f.write(f"KD_TH_INFER={KD_TH_INFER}\n")
        f.write(f"RAW_mAP50={map_raw:.4f}\n")
        f.write(f"TEACHER_SCORE_mAP50={map_teacher:.4f}  (ablation; uses teacher at eval)\n")
        f.write(f"KD_TEACHER_FREE_mAP50={map_kd:.4f}\n")
        f.write(f"Best head: {best_head_path}\n")
        f.write(f"VIS_DIR: {VIS_DIR}\n")

    print("\n================ SUMMARY (VAL split) ================")
    print(f"CLASS_MODE         = {CLASS_MODE} | NUM_CLASSES_MODE={NUM_CLASSES_MODE}")
    print(f"RAW               mAP@0.5 = {map_raw:.4f}")
    print(f"TEACHER_SCORE     mAP@0.5 = {map_teacher:.4f}   (ablation; teacher evaluated)")
    print(f"KD (teacher-free) mAP@0.5 = {map_kd:.4f}      (deployment path)")
    print(f"KD Selection: K={KD_TOPK_INFER} | th={KD_TH_INFER}")
    print(f"[Saved] Runs: {RUN_DIR}")
    print(f"[Saved] Visualizations: {VIS_DIR}")
    print(f"[Saved] Summary: {summary_path}")


if __name__ == "__main__":
    main()
