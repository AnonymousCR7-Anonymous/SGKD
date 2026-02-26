#!/usr/bin/env python3
"""

TrafficGaze saliency with MDS-ViTNet model using fixation-based metrics:
  - AUC-Borji: Area Under ROC Curve (Borji variant with uniform random sampling)
  - AUC-Judd: Area Under ROC Curve (Judd variant with saliency map as importance)
  - NSS: Normalized Scanpath Saliency
  - IG: Information Gain (relative to center bias baseline)
"""

import os
import json
import math
import time
import gc
import random
import argparse
import contextlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# NumPy 2.0 compatibility: trapz was renamed to trapezoid
if hasattr(np, 'trapezoid'):
    np_trapz = np.trapezoid
else:
    np_trapz = np.trapz

from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models import swin_t

# Import MDS-ViTNet components
from utils.TransformerEncoder import Encoder


# ----------------------------
# Config
# ----------------------------
@dataclass
class CFG:
    TG_ROOT: str = ".../Traffic_Gaze"
    TRAFFICFRAME_ROOT: str = os.path.join(TG_ROOT, "trafficframe")
    SALIENCY_ROOT: str = os.path.join(TG_ROOT, "saliencyframe")

    TRAIN_JSON: str = os.path.join(TG_ROOT, "train.json")
    VALID_JSON: str = os.path.join(TG_ROOT, "valid.json")
    TEST_JSON: str = os.path.join(TG_ROOT, "test.json")

    # Fixed image size
    IMG_H: int = 192
    IMG_W: int = 320

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SEED: int = 123

    # Training
    EPOCHS: int = 10
    BATCH_SIZE: int = 4
    ACCUM_STEPS: int = 4
    NUM_WORKERS: int = 2
    AMP: bool = True

    # Fine-tuning schedule
    FREEZE_EPOCHS: int = 2
    LR_ENCODER: float = 1e-5
    LR_DECODER: float = 1e-4
    LR_MERGE: float = 1e-4
    WEIGHT_DECAY: float = 0.01
    GRAD_CLIP_NORM: float = 0.5
    WARMUP_EPOCHS: int = 2

    # Fixation-aware loss weights
    LAMBDA_BCE: float = 1.0
    LAMBDA_NSS: float = 1.0
    LAMBDA_KL: float = 0.35
    LAMBDA_CC: float = 0.30

    # Multi-decoder loss weights
    LAMBDA_DEC1: float = 0.5
    LAMBDA_DEC2: float = 0.5

    # Fixation map generation (threshold percentile for deriving fixation from saliency)
    FIXATION_THRESHOLD_PERCENTILE: float = 95.0  # Top 5% of saliency as fixations

    # AUC settings
    AUC_NUM_SPLITS: int = 10  # Number of random splits for AUC-Borji
    AUC_NUM_THRESHOLDS: int = 100  # Number of thresholds for ROC curve

    # Eval
    USE_TTA: bool = True
    VAL_BATCH_SIZE: int = 4

    OUT_DIR: str = os.path.join(TG_ROOT, "mds_vitnet_fixation_metrics")

    def __post_init__(self):
        os.makedirs(self.OUT_DIR, exist_ok=True)


cfg = CFG()
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ----------------------------
# MDS-ViTNet Transformer Encoder Configs
# ----------------------------
cfg_enc6 = {
    "hidden_size": 768,
    "mlp_dim": 768 * 4,
    "num_heads": 12,
    "num_layers": 2,
    "attention_dropout_rate": 0,
    "dropout_rate": 0.0,
}

cfg_enc5 = {
    "hidden_size": 768,
    "mlp_dim": 768 * 4,
    "num_heads": 12,
    "num_layers": 2,
    "attention_dropout_rate": 0,
    "dropout_rate": 0.0,
}

cfg_enc4 = {
    "hidden_size": 768,
    "mlp_dim": 768 * 4,
    "num_heads": 12,
    "num_layers": 2,
    "attention_dropout_rate": 0,
    "dropout_rate": 0.0,
}

cfg_enc3 = {
    "hidden_size": 768,
    "mlp_dim": 768 * 4,
    "num_heads": 12,
    "num_layers": 2,
    "attention_dropout_rate": 0,
    "dropout_rate": 0.0,
}

cfg_enc2 = {
    "hidden_size": 512,
    "mlp_dim": 512 * 4,
    "num_heads": 8,
    "num_layers": 2,
    "attention_dropout_rate": 0,
    "dropout_rate": 0.0,
}

cfg_enc1 = {
    "hidden_size": 512,
    "mlp_dim": 512 * 4,
    "num_heads": 8,
    "num_layers": 2,
    "attention_dropout_rate": 0,
    "dropout_rate": 0.0,
}


# ----------------------------
# Utilities
# ----------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def print_memory() -> str:
    try:
        import psutil
        cpu = psutil.Process().memory_info().rss / 1e9
        gpu = (torch.cuda.memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
        return f"CPU: {cpu:.2f}GB | GPU: {gpu:.2f}GB"
    except Exception:
        return ""


def make_amp(device: torch.device, enabled: bool):
    if (not enabled) or device.type != "cuda":
        return None, contextlib.nullcontext()
    try:
        scaler = torch.amp.GradScaler('cuda')
        autocast = lambda: torch.amp.autocast('cuda')
        return scaler, autocast
    except Exception:
        return None, contextlib.nullcontext()


# ----------------------------
# Path reading
# ----------------------------
def _extract_paths(obj) -> List[str]:
    out = []

    def rec(x):
        if isinstance(x, str):
            s = x.strip().replace("\\", "/").lstrip("/")
            if s.lower().endswith(IMG_EXTS) and "/" in s:
                out.append(s)
        elif isinstance(x, list):
            for it in x:
                rec(it)
        elif isinstance(x, dict):
            for v in x.values():
                rec(v)

    rec(obj)
    return out


def read_paths(path: str) -> List[str]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt:
        return []

    def norm(rel: str) -> str:
        rel = rel.strip().replace("\\", "/").lstrip("/")
        for p in ("trafficframe/", "saliencyframe/"):
            if rel.startswith(p):
                rel = rel[len(p):]
        parts = rel.split("/")
        if len(parts) >= 2 and parts[0].isdigit():
            parts[0] = parts[0].zfill(2)
        return "/".join(parts)

    try:
        data = json.loads(txt)
        paths = [data] if isinstance(data, str) else _extract_paths(data)
        return [norm(p) for p in paths]
    except Exception:
        pass

    result = []
    for line in txt.splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith("#"):
            continue
        try:
            val = json.loads(line)
        except Exception:
            val = line.strip().strip('"\'')
        if isinstance(val, str) and val.lower().endswith(IMG_EXTS):
            result.append(norm(val))
        elif not isinstance(val, str):
            result.extend([norm(p) for p in _extract_paths(val)])
    return result


# ----------------------------
# Transforms
# ----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    return TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)


class TrainTransform:
    def __init__(self, h: int, w: int):
        self.h = h
        self.w = w

    def __call__(self, img: Image.Image, gt: Image.Image):
        if random.random() < 0.5:
            img = TF.hflip(img)
            gt = TF.hflip(gt)
        if random.random() < 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.9, 1.1))
            img = TF.adjust_contrast(img, random.uniform(0.9, 1.1))

        img = TF.resize(img, (self.h, self.w), interpolation=TF.InterpolationMode.BILINEAR)
        gt = TF.resize(gt, (self.h, self.w), interpolation=TF.InterpolationMode.BILINEAR)

        img_t = normalize_tensor(TF.to_tensor(img))
        gt_t = TF.to_tensor(gt)
        return img_t, gt_t


class EvalTransform:
    def __init__(self, h: int, w: int):
        self.h = h
        self.w = w

    def __call__(self, img: Image.Image, gt: Image.Image):
        img = TF.resize(img, (self.h, self.w), interpolation=TF.InterpolationMode.BILINEAR)
        gt = TF.resize(gt, (self.h, self.w), interpolation=TF.InterpolationMode.BILINEAR)
        img_t = normalize_tensor(TF.to_tensor(img))
        gt_t = TF.to_tensor(gt)
        return img_t, gt_t


# ----------------------------
# Dataset
# ----------------------------
class TrafficGazeDataset(Dataset):
    def __init__(self, traffic_root: str, saliency_root: str, rel_list: List[str], transform):
        self.traffic_root = traffic_root
        self.saliency_root = saliency_root
        self.transform = transform
        self.samples: List[Tuple[str, str]] = []

        for rel in rel_list:
            rel = rel.replace("\\", "/").lstrip("/")
            if "/" not in rel:
                continue

            img_path = os.path.join(traffic_root, rel)
            if not os.path.isfile(img_path):
                stem = os.path.splitext(img_path)[0]
                found = None
                for ext in (".jpg", ".jpeg", ".png"):
                    cand = stem + ext
                    if os.path.isfile(cand):
                        found = cand
                        break
                if found is None:
                    continue
                img_path = found

            rel_noext = os.path.splitext(rel)[0]
            gt_path = os.path.join(saliency_root, rel_noext + ".png")
            if not os.path.isfile(gt_path):
                alt = os.path.join(saliency_root, rel_noext + ".jpg")
                if os.path.isfile(alt):
                    gt_path = alt
                else:
                    continue

            self.samples.append((img_path, gt_path))

        print(f"[Dataset] {len(self.samples)} samples")
        if not self.samples:
            raise RuntimeError("No valid (image, gt) pairs found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, gt_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        gt = Image.open(gt_path).convert("L")
        img_t, gt_t = self.transform(img, gt)
        img.close()
        gt.close()
        return img_t, gt_t


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    gts = torch.stack([b[1] for b in batch], dim=0)
    return imgs, gts


# ----------------------------
# Fixation Map Generation
# ----------------------------
def saliency_to_fixation_map(saliency: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """
    Generate binary fixation map from continuous saliency map.

    Args:
        saliency: Continuous saliency map [H, W] in [0, 1]
        percentile: Threshold percentile (top N% becomes fixations)

    Returns:
        Binary fixation map [H, W] with 1s at fixation locations
    """
    threshold = np.percentile(saliency, percentile)
    fixation_map = (saliency >= threshold).astype(np.float32)
    return fixation_map


def create_center_bias(height: int, width: int, sigma_ratio: float = 0.2) -> np.ndarray:
    """
    Create a center bias baseline for Information Gain calculation.

    Args:
        height: Image height
        width: Image width
        sigma_ratio: Gaussian sigma as ratio of image diagonal

    Returns:
        Center bias map [H, W] normalized as probability distribution
    """
    y = np.arange(height)
    x = np.arange(width)
    xx, yy = np.meshgrid(x, y)

    center_y, center_x = height / 2, width / 2
    sigma = sigma_ratio * np.sqrt(height**2 + width**2)

    gaussian = np.exp(-((xx - center_x)**2 + (yy - center_y)**2) / (2 * sigma**2))

    # Normalize to probability distribution
    gaussian = gaussian / (gaussian.sum() + 1e-10)
    return gaussian.astype(np.float32)


# ----------------------------
# Fixation-Based Metrics (NumPy)
# ----------------------------
def nss_np(saliency_map: np.ndarray, fixation_map: np.ndarray, eps: float = 1e-10) -> float:
    """
    Normalized Scanpath Saliency (NSS)

    NSS measures the average normalized saliency at fixation locations.
    Higher values indicate better prediction of fixation locations.

    Formula: NSS = mean((S - mean(S)) / std(S) at fixation locations)

    Args:
        saliency_map: Predicted saliency map [H, W]
        fixation_map: Binary fixation map [H, W] (1 = fixation, 0 = non-fixation)

    Returns:
        NSS score (higher is better, typically 0-3)
    """
    saliency_map = saliency_map.astype(np.float64)
    fixation_map = fixation_map.astype(np.float64)

    # Check for valid inputs
    if not np.isfinite(saliency_map).all():
        saliency_map = np.nan_to_num(saliency_map, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize saliency map to zero mean and unit variance
    mean_sal = saliency_map.mean()
    std_sal = saliency_map.std() + eps
    normalized_sal = (saliency_map - mean_sal) / std_sal

    # Get fixation locations
    fixation_mask = fixation_map > 0.5

    if fixation_mask.sum() == 0:
        return 0.0

    # Compute NSS as mean of normalized saliency at fixation locations
    nss_score = normalized_sal[fixation_mask].mean()

    if not np.isfinite(nss_score):
        return 0.0

    return float(nss_score)


def auc_judd_np(saliency_map: np.ndarray, fixation_map: np.ndarray,
                num_thresholds: int = 100, eps: float = 1e-10) -> float:
    """
    AUC-Judd: Area Under ROC Curve using Judd's method.

    Uses the saliency map values at non-fixation locations as the
    negative distribution (importance-weighted sampling).

    Args:
        saliency_map: Predicted saliency map [H, W]
        fixation_map: Binary fixation map [H, W]
        num_thresholds: Number of thresholds for ROC curve

    Returns:
        AUC score in [0, 1] (0.5 = chance, 1.0 = perfect)
    """
    saliency_map = saliency_map.astype(np.float64).flatten()
    fixation_map = fixation_map.astype(np.float64).flatten()

    # Normalize saliency map to [0, 1]
    sal_min, sal_max = saliency_map.min(), saliency_map.max()
    if sal_max - sal_min < eps:
        return 0.5  # Random chance if saliency is constant
    saliency_map = (saliency_map - sal_min) / (sal_max - sal_min)

    # Get fixation and non-fixation indices
    fix_indices = np.where(fixation_map > 0.5)[0]
    nonfix_indices = np.where(fixation_map <= 0.5)[0]

    if len(fix_indices) == 0 or len(nonfix_indices) == 0:
        return 0.5

    # Saliency values at fixation and non-fixation locations
    sal_at_fix = saliency_map[fix_indices]
    sal_at_nonfix = saliency_map[nonfix_indices]

    # Compute ROC curve
    thresholds = np.linspace(0, 1, num_thresholds + 1)
    tpr = np.zeros(len(thresholds))  # True positive rate (sensitivity)
    fpr = np.zeros(len(thresholds))  # False positive rate (1 - specificity)

    n_fix = len(sal_at_fix)
    n_nonfix = len(sal_at_nonfix)

    for i, thresh in enumerate(thresholds):
        # True positives: fixations with saliency >= threshold
        tp = (sal_at_fix >= thresh).sum()
        # False positives: non-fixations with saliency >= threshold
        fp = (sal_at_nonfix >= thresh).sum()

        tpr[i] = tp / n_fix
        fpr[i] = fp / n_nonfix

    # Sort by FPR for AUC calculation
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]

    # Compute AUC using trapezoidal rule
    auc = np_trapz(tpr_sorted, fpr_sorted)

    return float(np.clip(auc, 0.0, 1.0))


def auc_borji_np(saliency_map: np.ndarray, fixation_map: np.ndarray,
                  num_splits: int = 10, num_samples: int = None,
                  num_thresholds: int = 100, eps: float = 1e-10) -> float:
    """
    AUC-Borji: Area Under ROC Curve using Borji's method.

    Uses uniform random sampling from non-fixation locations as the
    negative distribution (more robust to center bias).

    Args:
        saliency_map: Predicted saliency map [H, W]
        fixation_map: Binary fixation map [H, W]
        num_splits: Number of random sampling iterations
        num_samples: Number of negative samples per split (default: num fixations)
        num_thresholds: Number of thresholds for ROC curve

    Returns:
        AUC score in [0, 1] (0.5 = chance, 1.0 = perfect)
    """
    saliency_map = saliency_map.astype(np.float64).flatten()
    fixation_map = fixation_map.astype(np.float64).flatten()

    # Normalize saliency map to [0, 1]
    sal_min, sal_max = saliency_map.min(), saliency_map.max()
    if sal_max - sal_min < eps:
        return 0.5
    saliency_map = (saliency_map - sal_min) / (sal_max - sal_min)

    # Get fixation and non-fixation indices
    fix_indices = np.where(fixation_map > 0.5)[0]
    nonfix_indices = np.where(fixation_map <= 0.5)[0]

    if len(fix_indices) == 0 or len(nonfix_indices) == 0:
        return 0.5

    # Default: sample same number of negatives as positives
    if num_samples is None:
        num_samples = len(fix_indices)
    num_samples = min(num_samples, len(nonfix_indices))

    # Saliency values at fixation locations
    sal_at_fix = saliency_map[fix_indices]
    n_fix = len(sal_at_fix)

    auc_scores = []

    for _ in range(num_splits):
        # Random sample from non-fixation locations (uniform)
        sampled_nonfix = np.random.choice(nonfix_indices, size=num_samples, replace=False)
        sal_at_nonfix = saliency_map[sampled_nonfix]

        # Compute ROC curve for this split
        thresholds = np.linspace(0, 1, num_thresholds + 1)
        tpr = np.zeros(len(thresholds))
        fpr = np.zeros(len(thresholds))

        n_nonfix = len(sal_at_nonfix)

        for i, thresh in enumerate(thresholds):
            tp = (sal_at_fix >= thresh).sum()
            fp = (sal_at_nonfix >= thresh).sum()

            tpr[i] = tp / n_fix
            fpr[i] = fp / n_nonfix

        # Sort and compute AUC
        sorted_indices = np.argsort(fpr)
        auc = np_trapz(tpr[sorted_indices], fpr[sorted_indices])
        auc_scores.append(auc)

    mean_auc = np.mean(auc_scores)
    return float(np.clip(mean_auc, 0.0, 1.0))


def information_gain_np(saliency_map: np.ndarray, fixation_map: np.ndarray,
                         center_bias: np.ndarray = None, eps: float = 1e-10) -> float:
    """
    Information Gain (IG): Measures information gain over a baseline model.

    IG quantifies how much better the saliency map predicts fixations compared
    to a center-biased baseline, measured in bits per fixation.

    Formula: IG = mean(log2(P(fix) / P_baseline(fix))) at fixation locations

    Args:
        saliency_map: Predicted saliency map [H, W]
        fixation_map: Binary fixation map [H, W]
        center_bias: Baseline probability map [H, W] (default: Gaussian center bias)

    Returns:
        Information gain in bits (positive = better than baseline)
    """
    saliency_map = saliency_map.astype(np.float64)
    fixation_map = fixation_map.astype(np.float64)

    # Check for valid inputs
    if not np.isfinite(saliency_map).all():
        saliency_map = np.nan_to_num(saliency_map, nan=0.0, posinf=0.0, neginf=0.0)

    # Create center bias baseline if not provided
    if center_bias is None:
        center_bias = create_center_bias(saliency_map.shape[0], saliency_map.shape[1])
    center_bias = center_bias.astype(np.float64)

    # Ensure non-negative
    saliency_map = np.maximum(saliency_map, 0.0)
    center_bias = np.maximum(center_bias, eps)

    # Normalize saliency to probability distribution
    sal_sum = saliency_map.sum()
    if sal_sum < eps:
        saliency_map = np.ones_like(saliency_map) / saliency_map.size
    else:
        saliency_map = saliency_map / sal_sum

    # Ensure baseline is normalized
    bias_sum = center_bias.sum()
    if bias_sum > eps:
        center_bias = center_bias / bias_sum

    # Add small epsilon to avoid log(0)
    saliency_map = saliency_map + eps
    center_bias = center_bias + eps

    # Re-normalize after adding epsilon
    saliency_map = saliency_map / saliency_map.sum()
    center_bias = center_bias / center_bias.sum()

    # Get fixation locations
    fixation_mask = fixation_map > 0.5

    if fixation_mask.sum() == 0:
        return 0.0

    # Compute information gain at each fixation location
    # IG = log2(P_model / P_baseline)
    log_ratio = np.log2(saliency_map / center_bias)
    ig = log_ratio[fixation_mask].mean()

    if not np.isfinite(ig):
        return 0.0

    return float(ig)


# ----------------------------
# Additional Saliency Metrics (for completeness)
# ----------------------------
def norm_minmax_np(m: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Min-max normalize array to [0, 1] range."""
    m = m.astype(np.float64)
    if not np.isfinite(m).all():
        m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(m.min())
    mx = float(m.max())
    if mx - mn < eps:
        return np.full_like(m, 0.5, dtype=np.float32)
    result = (m - mn) / (mx - mn)
    return result.astype(np.float32)


def cc_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-9) -> float:
    """Pearson's Correlation Coefficient."""
    p = pred.reshape(-1).astype(np.float64)
    g = gt.reshape(-1).astype(np.float64)
    if not (np.isfinite(p).all() and np.isfinite(g).all()):
        return 0.0
    p_mean = p.mean()
    g_mean = g.mean()
    cov = np.mean((p - p_mean) * (g - g_mean))
    sigma_p = p.std() + eps
    sigma_g = g.std() + eps
    cc = cov / (sigma_p * sigma_g)
    return float(np.clip(cc, -1.0, 1.0))


def sim_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-10) -> float:
    """Similarity (SIM)."""
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)
    if not (np.isfinite(p).all() and np.isfinite(g).all()):
        return 0.0
    p = np.maximum(p, 0.0)
    g = np.maximum(g, 0.0)
    p_sum = p.sum()
    g_sum = g.sum()
    if p_sum > eps:
        p = p / p_sum
    else:
        p = np.ones_like(p) / p.size
    if g_sum > eps:
        g = g / g_sum
    else:
        g = np.ones_like(g) / g.size
    sim = np.minimum(p, g).sum()
    return float(np.clip(sim, 0.0, 1.0))


def kld_np(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-10) -> float:
    """Kullback-Leibler Divergence."""
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)
    if not (np.isfinite(p).all() and np.isfinite(g).all()):
        return 0.0
    p = np.maximum(p, 0.0)
    g = np.maximum(g, 0.0)
    p_sum = p.sum()
    g_sum = g.sum()
    if p_sum < eps or g_sum < eps:
        return 0.0
    p_norm = p / p_sum
    g_norm = g / g_sum
    p_norm = p_norm + eps
    g_norm = g_norm + eps
    p_norm = p_norm / p_norm.sum()
    g_norm = g_norm / g_norm.sum()
    kld = (g_norm * np.log(g_norm / p_norm)).sum()
    if not np.isfinite(kld):
        return 0.0
    return float(max(0.0, kld))


# ----------------------------
# PyTorch Loss Functions
# ----------------------------
def prob_norm_torch(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    x = x.float()
    x = x - x.amin(dim=(-2, -1), keepdim=True)
    x = x / (x.amax(dim=(-2, -1), keepdim=True) + eps)
    s = x.sum(dim=(-2, -1), keepdim=True)
    return x / (s + eps)


def kl_torch(gt01: torch.Tensor, pred01: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    Q = prob_norm_torch(gt01, eps=eps)
    P = prob_norm_torch(pred01, eps=eps)
    return (Q * torch.log((Q + eps) / (P + eps))).sum(dim=(-2, -1)).mean()


def cc_loss_torch(gt01: torch.Tensor, pred01: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    g = gt01.flatten(2).float()
    p = pred01.flatten(2).float()
    g = (g - g.mean(dim=2, keepdim=True)) / (g.std(dim=2, keepdim=True) + eps)
    p = (p - p.mean(dim=2, keepdim=True)) / (p.std(dim=2, keepdim=True) + eps)
    cc = (g * p).mean(dim=2)
    return (1.0 - cc).mean()


def nss_loss_torch(pred: torch.Tensor, gt_saliency: torch.Tensor,
                   fixation_threshold: float = 0.95, eps: float = 1e-9) -> torch.Tensor:
    """
    NSS-based loss for training.

    Args:
        pred: Predicted saliency [B, 1, H, W] (probabilities)
        gt_saliency: Ground truth saliency [B, 1, H, W]
        fixation_threshold: Percentile threshold for fixation map generation

    Returns:
        Negative NSS (for minimization)
    """
    B = pred.size(0)
    losses = []

    for i in range(B):
        pred_i = pred[i, 0]  # [H, W]
        gt_i = gt_saliency[i, 0]  # [H, W]

        # Normalize prediction to zero mean and unit variance
        pred_norm = (pred_i - pred_i.mean()) / (pred_i.std() + eps)

        # Create fixation mask from GT (top percentile)
        threshold = torch.quantile(gt_i.flatten(), fixation_threshold)
        fix_mask = (gt_i >= threshold).float()

        # NSS = mean of normalized prediction at fixation locations
        if fix_mask.sum() > 0:
            nss = (pred_norm * fix_mask).sum() / (fix_mask.sum() + eps)
            losses.append(-nss)  # Negative because we want to maximize NSS
        else:
            losses.append(torch.tensor(0.0, device=pred.device))

    return torch.stack(losses).mean()


def compute_single_loss(logits: torch.Tensor, gt: torch.Tensor, loss_mode: str) -> torch.Tensor:
    """
    Compute loss for a single prediction map.

    Args:
        logits: [B, 1, H, W] raw logits from model
        gt: [B, 1, H, W] ground truth saliency in [0, 1]
        loss_mode: 'bce', 'nss', 'combo', or 'combo_fix'
    """
    if loss_mode == "bce":
        return F.binary_cross_entropy_with_logits(logits, gt)

    pred01 = torch.sigmoid(logits)

    if loss_mode == "nss":
        # Pure NSS-based loss
        return nss_loss_torch(pred01, gt)

    if loss_mode == "combo":
        # Original combo: BCE + KLD + CC
        loss = cfg.LAMBDA_BCE * F.binary_cross_entropy_with_logits(logits, gt)
        loss = loss + cfg.LAMBDA_KL * kl_torch(gt, pred01)
        loss = loss + cfg.LAMBDA_CC * cc_loss_torch(gt, pred01)
        return loss

    if loss_mode == "combo_fix":
        # Fixation-aware combo: BCE + NSS + KLD + CC
        loss = cfg.LAMBDA_BCE * F.binary_cross_entropy_with_logits(logits, gt)
        loss = loss + cfg.LAMBDA_NSS * nss_loss_torch(pred01, gt)
        loss = loss + cfg.LAMBDA_KL * kl_torch(gt, pred01)
        loss = loss + cfg.LAMBDA_CC * cc_loss_torch(gt, pred01)
        return loss

    raise ValueError(f"Unknown loss_mode={loss_mode}")


# ----------------------------
# LR schedule
# ----------------------------
def set_lrs(opt, epoch: int, total_epochs: int, warmup_epochs: int,
            lr_enc: float, lr_dec: float, lr_merge: float = None):
    if epoch <= warmup_epochs:
        t = epoch / max(1, warmup_epochs)
    else:
        t = 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)))

    opt.param_groups[0]["lr"] = lr_enc * t
    opt.param_groups[1]["lr"] = lr_dec * t
    if len(opt.param_groups) > 2 and lr_merge is not None:
        opt.param_groups[2]["lr"] = lr_merge * t


# ----------------------------
# MDS-ViTNet Model Components
# ----------------------------
class TransEncoder(nn.Module):
    """Transformer Encoder module from MDS-ViTNet."""

    def __init__(self, in_channels, spatial_size, cfg):
        super(TransEncoder, self).__init__()

        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels,
            out_channels=cfg['hidden_size'],
            kernel_size=1,
            stride=1
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, spatial_size, cfg['hidden_size']))
        self.transformer_encoder = Encoder(cfg)

    def forward(self, x):
        a, b = x.shape[2], x.shape[3]
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x + self.position_embeddings
        x = self.transformer_encoder(embeddings)
        B, n_patch, hidden = x.shape
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, a, b)

        return x


class SwinTEncoder(nn.Module):
    """Swin-T encoder that extracts multi-scale features."""

    def __init__(self):
        super(SwinTEncoder, self).__init__()
        base_model = swin_t(pretrained=True)
        base_layers = list(base_model.children())[0][:-1]
        self.encoder = nn.ModuleList(base_layers)

    def forward(self, x):
        outputs_1, outputs_2 = [], []
        for ii, layer in enumerate(self.encoder):
            x = layer(x)
            if ii in {2, 4, 6}:
                outputs_1.append(x)
            elif ii in {1, 3, 5}:
                outputs_2.append(x)
        return outputs_1, outputs_2


class MDSViTNetDecoder1(nn.Module):
    """First decoder path (processes outputs from layers 2, 4, 6)."""

    def __init__(self, img_h: int, img_w: int, output_logits: bool = True):
        super(MDSViTNetDecoder1, self).__init__()
        self.output_logits = output_logits

        self.conv1 = nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

        self.batchnorm1 = nn.BatchNorm2d(768)
        self.batchnorm2 = nn.BatchNorm2d(512)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.batchnorm5 = nn.BatchNorm2d(64)
        self.batchnorm6 = nn.BatchNorm2d(32)

        self.TransEncoder2 = TransEncoder(in_channels=192, spatial_size=24 * 40, cfg=cfg_enc2)
        self.TransEncoder4 = TransEncoder(in_channels=384, spatial_size=12 * 20, cfg=cfg_enc4)
        self.TransEncoder6 = TransEncoder(in_channels=768, spatial_size=6 * 10, cfg=cfg_enc6)

        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2, x4, x6 = x

        x2 = x2.permute(0, 3, 1, 2)
        x4 = x4.permute(0, 3, 1, 2)
        x6 = x6.permute(0, 3, 1, 2)

        x6 = self.TransEncoder6(x6)
        x6 = self.conv1(x6)
        x6 = self.batchnorm1(x6)
        x6 = self.relu(x6)
        x6 = self.upsample(x6)

        x4_a = self.TransEncoder4(x4)
        x4 = x6 * x4_a
        x4 = self.relu(x4)
        x4 = self.conv2(x4)
        x4 = self.batchnorm2(x4)
        x4 = self.relu(x4)
        x4 = self.upsample(x4)

        x2_a = self.TransEncoder2(x2)
        x2 = x4 * x2_a
        x2 = self.relu(x2)
        x2 = self.conv3(x2)
        x2 = self.batchnorm3(x2)
        x2 = self.relu(x2)
        x2 = self.upsample(x2)

        x = self.conv4(x2)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.relu(x)
        x = self.conv7(x)

        if self.output_logits:
            return x
        else:
            return self.sigmoid(x)


class MDSViTNetDecoder2(nn.Module):
    """Second decoder path (processes outputs from layers 1, 3, 5)."""

    def __init__(self, img_h: int, img_w: int, output_logits: bool = True):
        super(MDSViTNetDecoder2, self).__init__()
        self.output_logits = output_logits

        self.conv1 = nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

        self.batchnorm1 = nn.BatchNorm2d(768)
        self.batchnorm2 = nn.BatchNorm2d(512)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.batchnorm5 = nn.BatchNorm2d(64)
        self.batchnorm6 = nn.BatchNorm2d(32)

        self.TransEncoder1 = TransEncoder(in_channels=96, spatial_size=48 * 80, cfg=cfg_enc1)
        self.TransEncoder3 = TransEncoder(in_channels=192, spatial_size=24 * 40, cfg=cfg_enc3)
        self.TransEncoder5 = TransEncoder(in_channels=384, spatial_size=12 * 20, cfg=cfg_enc5)

        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, x3, x5 = x

        x1 = x1.permute(0, 3, 1, 2)
        x3 = x3.permute(0, 3, 1, 2)
        x5 = x5.permute(0, 3, 1, 2)

        x5 = self.TransEncoder5(x5)
        x5 = self.conv1(x5)
        x5 = self.batchnorm1(x5)
        x5 = self.relu(x5)
        x5 = self.upsample(x5)

        x3_a = self.TransEncoder3(x3)
        x3 = x5 * x3_a
        x3 = self.relu(x3)
        x3 = self.conv2(x3)
        x3 = self.batchnorm2(x3)
        x3 = self.relu(x3)
        x3 = self.upsample(x3)

        x1_a = self.TransEncoder1(x1)
        x1 = x3 * x1_a
        x1 = self.relu(x1)
        x1 = self.conv3(x1)
        x1 = self.batchnorm3(x1)
        x1 = self.relu(x1)
        x1 = self.upsample(x1)

        x = self.conv4(x1)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.relu(x)
        x = self.conv7(x)

        if self.output_logits:
            return x
        else:
            return self.sigmoid(x)


class CNNMerge(nn.Module):
    """CNN merge module to combine outputs from two decoders."""

    def __init__(self, output_logits: bool = True):
        super(CNNMerge, self).__init__()
        self.output_logits = output_logits

        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, map1, map2):
        x = torch.cat([map1, map2], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)

        if self.output_logits:
            return x
        else:
            return self.sigmoid(x)


class MDSViTNet(nn.Module):
    """Complete MDS-ViTNet model with encoder and dual decoders."""

    def __init__(self, img_h: int, img_w: int, use_merge: bool = False, output_logits: bool = True):
        super(MDSViTNet, self).__init__()
        self.use_merge = use_merge
        self.output_logits = output_logits

        self.encoder = SwinTEncoder()
        self.decoder_1 = MDSViTNetDecoder1(img_h, img_w, output_logits=output_logits)
        self.decoder_2 = MDSViTNetDecoder2(img_h, img_w, output_logits=output_logits)

        if use_merge:
            self.merge = CNNMerge(output_logits=output_logits)

    def forward(self, x):
        outputs_1, outputs_2 = self.encoder(x)
        map1 = self.decoder_1(outputs_1)
        map2 = self.decoder_2(outputs_2)

        if self.use_merge:
            merged = self.merge(map1, map2)
            return map1, map2, merged
        else:
            return map1, map2


# ----------------------------
# Train / Eval
# ----------------------------
def train_epoch(model, loader, opt, device, scaler, autocast_ctx,
                loss_mode: str, use_merge: bool) -> float:
    model.train()
    total, n = 0.0, 0
    nan_count = 0
    opt.zero_grad(set_to_none=True)

    for step, (imgs, gt) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        with autocast_ctx():
            if use_merge:
                logits1, logits2, logits_merged = model(imgs)
                loss1 = compute_single_loss(logits1, gt, loss_mode)
                loss2 = compute_single_loss(logits2, gt, loss_mode)
                loss_merged = compute_single_loss(logits_merged, gt, loss_mode)
                loss = (cfg.LAMBDA_DEC1 * loss1 +
                        cfg.LAMBDA_DEC2 * loss2 +
                        loss_merged)
            else:
                logits1, logits2 = model(imgs)
                loss1 = compute_single_loss(logits1, gt, loss_mode)
                loss2 = compute_single_loss(logits2, gt, loss_mode)
                loss = cfg.LAMBDA_DEC1 * loss1 + cfg.LAMBDA_DEC2 * loss2

            loss = loss / cfg.ACCUM_STEPS

        if not torch.isfinite(loss):
            nan_count += 1
            if nan_count <= 5:
                print(f"  [Warning] NaN/Inf loss detected at step {step}, skipping batch...")
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.update()
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            if (step + 1) % cfg.ACCUM_STEPS == 0:
                scaler.unscale_(opt)
                valid_gradients = True
                for param in model.parameters():
                    if param.grad is not None:
                        if not torch.isfinite(param.grad).all():
                            valid_gradients = False
                            break

                if valid_gradients:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
                    scaler.step(opt)
                else:
                    nan_count += 1
                    if nan_count <= 5:
                        print(f"  [Warning] NaN gradients detected at step {step}, skipping update...")

                scaler.update()
                opt.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (step + 1) % cfg.ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
                opt.step()
                opt.zero_grad(set_to_none=True)

        bs = imgs.size(0)
        total += float(loss.item()) * bs * cfg.ACCUM_STEPS
        n += bs

        if step % 500 == 0:
            print(f"  Step {step}/{len(loader)}, loss={loss.item() * cfg.ACCUM_STEPS:.4f}")

    if nan_count > 0:
        print(f"  [Warning] Total NaN/Inf occurrences this epoch: {nan_count}")

    return total / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, loss_mode: str, use_tta: bool,
             use_merge: bool) -> Dict[str, float]:
    """
    Evaluate model with fixation-based metrics.

    Returns metrics:
        - loss: Training loss
        - CC: Correlation Coefficient
        - SIM: Similarity
        - KLD: KL Divergence
        - NSS: Normalized Scanpath Saliency
        - AUC_Judd: AUC using Judd's method
        - AUC_Borji: AUC using Borji's method
        - IG: Information Gain
    """
    model.eval()

    # Pre-compute center bias for IG
    center_bias = create_center_bias(cfg.IMG_H, cfg.IMG_W)

    total_loss, n = 0.0, 0

    # Saliency metrics
    cc_l, sim_l, kld_l = [], [], []

    # Fixation metrics
    nss_l, auc_judd_l, auc_borji_l, ig_l = [], [], [], []

    for imgs, gt in loader:
        imgs = imgs.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        if use_tta:
            if use_merge:
                logits1, logits2, logits_merged = model(imgs)
                logits1_flip, logits2_flip, logits_merged_flip = model(torch.flip(imgs, dims=[3]))
                logits1 = 0.5 * (logits1 + torch.flip(logits1_flip, dims=[3]))
                logits2 = 0.5 * (logits2 + torch.flip(logits2_flip, dims=[3]))
                logits_merged = 0.5 * (logits_merged + torch.flip(logits_merged_flip, dims=[3]))
            else:
                logits1, logits2 = model(imgs)
                logits1_flip, logits2_flip = model(torch.flip(imgs, dims=[3]))
                logits1 = 0.5 * (logits1 + torch.flip(logits1_flip, dims=[3]))
                logits2 = 0.5 * (logits2 + torch.flip(logits2_flip, dims=[3]))
        else:
            if use_merge:
                logits1, logits2, logits_merged = model(imgs)
            else:
                logits1, logits2 = model(imgs)

        # Compute loss
        if use_merge:
            loss1 = compute_single_loss(logits1, gt, loss_mode)
            loss2 = compute_single_loss(logits2, gt, loss_mode)
            loss_merged = compute_single_loss(logits_merged, gt, loss_mode)
            loss = cfg.LAMBDA_DEC1 * loss1 + cfg.LAMBDA_DEC2 * loss2 + loss_merged
            pred_logits = logits_merged
        else:
            loss1 = compute_single_loss(logits1, gt, loss_mode)
            loss2 = compute_single_loss(logits2, gt, loss_mode)
            loss = cfg.LAMBDA_DEC1 * loss1 + cfg.LAMBDA_DEC2 * loss2
            pred_logits = 0.5 * (logits1 + logits2)

        bs = imgs.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

        # Convert to numpy for metrics
        pred01 = torch.sigmoid(pred_logits)
        pred_np = pred01[:, 0].detach().cpu().numpy()
        gt_np = gt[:, 0].detach().cpu().numpy()

        for i in range(pred_np.shape[0]):
            p = norm_minmax_np(pred_np[i])
            g = norm_minmax_np(gt_np[i])

            # Generate fixation map from GT saliency
            fix_map = saliency_to_fixation_map(g, cfg.FIXATION_THRESHOLD_PERCENTILE)

            # Saliency metrics
            cc_l.append(cc_np(p, g))
            sim_l.append(sim_np(p, g))
            kld_l.append(kld_np(g, p))

            # Fixation metrics
            nss_l.append(nss_np(p, fix_map))
            auc_judd_l.append(auc_judd_np(p, fix_map, cfg.AUC_NUM_THRESHOLDS))
            auc_borji_l.append(auc_borji_np(p, fix_map, cfg.AUC_NUM_SPLITS,
                                            num_thresholds=cfg.AUC_NUM_THRESHOLDS))
            ig_l.append(information_gain_np(p, fix_map, center_bias))

    mean = lambda x: float(np.mean(x)) if len(x) else float("nan")

    return {
        "loss": total_loss / max(1, n),
        # Saliency metrics
        "CC": mean(cc_l),
        "SIM": mean(sim_l),
        "KLD": mean(kld_l),
        # Fixation metrics
        "NSS": mean(nss_l),
        "AUC_Judd": mean(auc_judd_l),
        "AUC_Borji": mean(auc_borji_l),
        "IG": mean(ig_l),
    }


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, default="combo_fix",
                        choices=["bce", "nss", "combo", "combo_fix"],
                        help="Loss mode: 'bce', 'nss', 'combo' (BCE+KLD+CC), or 'combo_fix' (BCE+NSS+KLD+CC).")
    parser.add_argument("--use_merge", action="store_true",
                        help="Use CNN merge module to combine decoder outputs.")
    parser.add_argument("--fixation_percentile", type=float, default=95.0,
                        help="Percentile threshold for generating fixation maps from saliency (default: 95)")
    args = parser.parse_args()

    # Update config with args
    cfg.FIXATION_THRESHOLD_PERCENTILE = args.fixation_percentile

    seed_all(cfg.SEED)
    device = torch.device(cfg.DEVICE)

    print("=" * 80)
    print("MDS-ViTNet Training with Fixation-Based Metrics")
    print("=" * 80)
    print(f"[Device] {device}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[Config] HxW={cfg.IMG_H}x{cfg.IMG_W} | batch={cfg.BATCH_SIZE} "
          f"accum={cfg.ACCUM_STEPS} workers={cfg.NUM_WORKERS} AMP={cfg.AMP}")
    print(f"[LossMode] {args.loss}")
    print(f"[UseMerge] {args.use_merge}")
    print(f"[FixationPercentile] {cfg.FIXATION_THRESHOLD_PERCENTILE}")
    print("=" * 80)

    train_paths = read_paths(cfg.TRAIN_JSON)
    val_paths = read_paths(cfg.VALID_JSON)
    test_paths = read_paths(cfg.TEST_JSON)
    print(f"[Data] train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")

    train_ds = TrafficGazeDataset(cfg.TRAFFICFRAME_ROOT, cfg.SALIENCY_ROOT,
                                  train_paths, TrainTransform(cfg.IMG_H, cfg.IMG_W))
    val_ds = TrafficGazeDataset(cfg.TRAFFICFRAME_ROOT, cfg.SALIENCY_ROOT,
                                val_paths, EvalTransform(cfg.IMG_H, cfg.IMG_W))
    test_ds = TrafficGazeDataset(cfg.TRAFFICFRAME_ROOT, cfg.SALIENCY_ROOT,
                                 test_paths, EvalTransform(cfg.IMG_H, cfg.IMG_W))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print("[Creating MDS-ViTNet model...]")
    model = MDSViTNet(cfg.IMG_H, cfg.IMG_W, use_merge=args.use_merge).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {n_params:,} params")

    # Optimizer groups
    enc_params = list(model.encoder.parameters())
    dec_params = list(model.decoder_1.parameters()) + list(model.decoder_2.parameters())

    if args.use_merge:
        merge_params = list(model.merge.parameters())
        opt = torch.optim.AdamW([
            {"params": enc_params, "lr": cfg.LR_ENCODER},
            {"params": dec_params, "lr": cfg.LR_DECODER},
            {"params": merge_params, "lr": cfg.LR_MERGE},
        ], weight_decay=cfg.WEIGHT_DECAY)
    else:
        opt = torch.optim.AdamW([
            {"params": enc_params, "lr": cfg.LR_ENCODER},
            {"params": dec_params, "lr": cfg.LR_DECODER},
        ], weight_decay=cfg.WEIGHT_DECAY)

    scaler, autocast_fn = make_amp(device, cfg.AMP)

    def autocast_ctx():
        if scaler is None:
            return contextlib.nullcontext()
        return autocast_fn()

    best_score = float("-inf")  # Higher is better for fixation metrics
    best_epoch = 0

    # Output directory
    merge_suffix = "_merge" if args.use_merge else ""
    out_dir = os.path.join(cfg.OUT_DIR, f"loss_{args.loss}{merge_suffix}")
    os.makedirs(out_dir, exist_ok=True)
    best_ckpt = os.path.join(out_dir, "best.pth")
    last_ckpt = os.path.join(out_dir, "last.pth")
    summary_txt = os.path.join(out_dir, "test_summary.txt")

    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        mem = print_memory()
        print(f"\n[Epoch {epoch:02d}/{cfg.EPOCHS}] {mem}")

        # Freeze/unfreeze encoder
        freeze = epoch <= cfg.FREEZE_EPOCHS
        for p in model.encoder.parameters():
            p.requires_grad = (not freeze)

        lr_merge = cfg.LR_MERGE if args.use_merge else None
        set_lrs(opt, epoch, cfg.EPOCHS, cfg.WARMUP_EPOCHS,
                cfg.LR_ENCODER, cfg.LR_DECODER, lr_merge)

        lr_enc = opt.param_groups[0]["lr"]
        lr_dec = opt.param_groups[1]["lr"]
        if args.use_merge:
            lr_mrg = opt.param_groups[2]["lr"]
            print(f"  freeze_enc={freeze} | lr_enc={lr_enc:.2e} "
                  f"lr_dec={lr_dec:.2e} lr_merge={lr_mrg:.2e}")
        else:
            print(f"  freeze_enc={freeze} | lr_enc={lr_enc:.2e} lr_dec={lr_dec:.2e}")

        print("  Training...")
        tr_loss = train_epoch(model, train_loader, opt, device, scaler,
                              autocast_ctx, args.loss, args.use_merge)

        print("  Evaluating...")
        val = evaluate(model, val_loader, device, args.loss,
                       use_tta=False, use_merge=args.use_merge)

        # Score based on fixation metrics (higher is better)
        # Combine NSS + AUC_Judd + AUC_Borji (all higher is better)
        score = val["NSS"] + val["AUC_Judd"] + val["AUC_Borji"]

        dt = time.time() - t0
        print(f"  train_loss={tr_loss:.4f} | val_loss={val['loss']:.4f}")
        print(f"  Saliency:  CC={val['CC']:.4f} SIM={val['SIM']:.4f} KLD={val['KLD']:.4f}")
        print(f"  Fixation:  NSS={val['NSS']:.4f} AUC_J={val['AUC_Judd']:.4f} "
              f"AUC_B={val['AUC_Borji']:.4f} IG={val['IG']:.4f}")
        print(f"  score={score:.4f} [{dt:.0f}s]")

        torch.save({"epoch": epoch, "state_dict": model.state_dict()}, last_ckpt)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, best_ckpt)
            print("  * Best!")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print(f"Training done! Best epoch: {best_epoch} (score={best_score:.4f})")
    print("=" * 80)

    # Test best model
    print("\n[Loading best model for testing...]")
    ckpt = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()

    test = evaluate(model, test_loader, device, args.loss,
                    use_tta=cfg.USE_TTA, use_merge=args.use_merge)

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Model = MDS-ViTNet")
    print(f"LossMode = {args.loss}")
    print(f"UseMerge = {args.use_merge}")
    print("-" * 40)
    print("Saliency Metrics:")
    print(f"  CC  = {test['CC']:.4f}")
    print(f"  SIM = {test['SIM']:.4f}")
    print(f"  KLD = {test['KLD']:.4f}")
    print("-" * 40)
    print("Fixation Metrics:")
    print(f"  NSS       = {test['NSS']:.4f}")
    print(f"  AUC-Judd  = {test['AUC_Judd']:.4f}")
    print(f"  AUC-Borji = {test['AUC_Borji']:.4f}")
    print(f"  IG        = {test['IG']:.4f}")
    print("=" * 80)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("TrafficGaze MDS-ViTNet (192x320) - Fixation Metrics\n")
        f.write("=" * 50 + "\n")
        f.write(f"model: MDS-ViTNet\n")
        f.write(f"encoder: Swin-T\n")
        f.write(f"loss_mode: {args.loss}\n")
        f.write(f"use_merge: {args.use_merge}\n")
        f.write(f"fixation_percentile: {cfg.FIXATION_THRESHOLD_PERCENTILE}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write("-" * 50 + "\n")
        f.write("Saliency Metrics:\n")
        f.write(f"  CC: {test['CC']:.6f}\n")
        f.write(f"  SIM: {test['SIM']:.6f}\n")
        f.write(f"  KLD: {test['KLD']:.6f}\n")
        f.write("-" * 50 + "\n")
        f.write("Fixation Metrics:\n")
        f.write(f"  NSS: {test['NSS']:.6f}\n")
        f.write(f"  AUC_Judd: {test['AUC_Judd']:.6f}\n")
        f.write(f"  AUC_Borji: {test['AUC_Borji']:.6f}\n")
        f.write(f"  IG: {test['IG']:.6f}\n")

    print(f"\n[Saved] best={best_ckpt}")
    print(f"[Saved] last={last_ckpt}")
    print(f"[Saved] summary={summary_txt}")


if __name__ == "__main__":
    main()
