import os
import glob
import shutil
import math
import json
from collections import defaultdict

import numpy as np
import scipy.io as sio
import cv2
from ultralytics import YOLO
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

start = time.time()
# 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16
# CONFIG
# --------------------
FRAMES_DIR   = r'.../Traffic_Gaze/trafficframe/01'
FIX_MAT_PATH = r'.../fixdata/fixdata1.mat'
OUTPUT_ROOT  = r'.../TrafficGaze/Video1_salient_dataset'

# frame geometry (WxH)
FRAME_W, FRAME_H = 1280, 720

# YOLO settings
YOLO_WEIGHTS = 'yolo11x.pt'
IMGSZ        = 640
DEVICE       = 0
HALF         = True

# Detection filtering
CONF_THRES   = 0.25
IOU_THRES    = 0.5
MAX_DET      = 100

# Saliency rules
MIN_FIX_INSIDE_BBOX = 5
K_WINDOW             = 5
N_PROPAGATE          = 60

# Sampling
SAMPLE_EVERY = 5

# Classes
OUR_CLASSES = [
    'people',          # 0
    'car',             # 1
    'motorcycle',      # 2
    'traffic-light',   # 3
    'traffic-sign',    # 4
    'bus',             # 5
    'truck'            # 6
]
OUR_TO_ID = {n: i for i, n in enumerate(OUR_CLASSES)}

COCO_TO_OUR_NAME = {
    'person'        : 'people',
    'car'           : 'car',
    'motorcycle'    : 'motorcycle',
    'traffic light' : 'traffic-light',
    'stop sign'     : 'traffic-sign',
    'bus'           : 'bus',
    'truck'         : 'truck'
}

# Output folders
OUT_IMG_DIR = os.path.join(OUTPUT_ROOT, 'images')      # (kept)
OUT_LBL_DIR = os.path.join(OUTPUT_ROOT, 'labels')
OUT_VIS_DIR = os.path.join(OUTPUT_ROOT, 'images_vis')
OUT_SAL_DIR = os.path.join(OUTPUT_ROOT, 'saliency')    # NEW: saliency heatmaps
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)
os.makedirs(OUT_VIS_DIR, exist_ok=True)
os.makedirs(OUT_SAL_DIR, exist_ok=True)

# --------------------
# Utils
# --------------------
def load_fixations_mat(mat_path, total_frames=None):
    md = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    key = None
    for k in ['fixdata', 'fixdata1', 'fixdata01', 'fixdata_1']:
        if k in md:
            key = k
            break
    if key is None:
        for k in md:
            if isinstance(md[k], np.ndarray) and md[k].dtype == np.object_:
                key = k
                break
    if key is None:
        raise ValueError("Could not find fixdata cell array in MAT file.")

    cells = md[key]
    if cells.ndim != 1:
        cells = cells.reshape(-1)

    num_frames = len(cells) if total_frames is None else min(len(cells), total_frames)
    all_fix = []
    for i in range(num_frames):
        entry = cells[i]
        if entry is None:
            all_fix.append([])
            continue
        arr = np.array(entry, dtype=float)
        if arr.size == 0:
            all_fix.append([])
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 4)
        elif arr.shape[0] == 4 and arr.shape[1] != 4:
            arr = arr.T

        if arr.shape[1] != 4:
            all_fix.append([])
            continue

        xy = arr[:, 2:4]
        xy = xy[~np.isnan(xy).any(axis=1)]
        if xy.size > 0:
            xs = np.clip(xy[:, 0], 0, FRAME_W - 1)
            ys = np.clip(xy[:, 1], 0, FRAME_H - 1)
            xy = np.stack([xs, ys], axis=1)
            all_fix.append([(float(x), float(y)) for x, y in xy])
        else:
            all_fix.append([])

    return all_fix

def yolo_names_to_ours(names, cls_id):
    name = names.get(int(cls_id), None)
    if name is None:
        return None
    name = name.lower().strip()
    return COCO_TO_OUR_NAME.get(name, None)

def bbox_xyxy_to_yolo_txt(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / img_w, cy / img_h, w / img_w, h / img_h

def natural_sorted_frame_list(folder):
    exts = ('*.jpg', '*.png', '*.jpeg', '*.bmp')
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    if not files:
        raise FileNotFoundError(f"No frames found under: {folder}")
    def keyfunc(p):
        base = os.path.basename(p)
        nums = ''.join([c if c.isdigit() else ' ' for c in base]).split()
        return int(nums[-1]) if nums else base
    return sorted(files, key=keyfunc)

def _denorm_xywh_to_xyxy(cx_n, cy_n, w_n, h_n, W, H):
    cx, cy = cx_n * W, cy_n * H
    bw, bh = w_n * W, h_n * H
    x1 = int(round(cx - bw / 2)); y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2)); y2 = int(round(cy + bh / 2))
    x1 = max(0, min(x1, W - 1)); y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W - 1)); y2 = max(0, min(y2, H - 1))
    return x1, y1, x2, y2

# --- saliency heatmap builder (variance=200 px => sigma = sqrt(200)) ---
def build_saliency_map(points, H, W, var_px=200.0):
    if len(points) == 0:
        return np.zeros((H, W), dtype=np.uint8)
    sigma = math.sqrt(var_px)
    radius = int(3 * sigma)  # local window = +/- 3σ
    heat = np.zeros((H, W), dtype=np.float32)

    for (x, y) in points:
        cx = int(round(x)); cy = int(round(y))
        x1 = max(0, cx - radius); x2 = min(W - 1, cx + radius)
        y1 = max(0, cy - radius); y2 = min(H - 1, cy + radius)
        if x2 <= x1 or y2 <= y1:
            continue

        xs = np.arange(x1, x2 + 1, dtype=np.float32)
        ys = np.arange(y1, y2 + 1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        g = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2.0 * sigma * sigma))
        heat[y1:y2+1, x1:x2+1] += g

    # normalize to 0..255
    m = heat.max()
    if m > 0:
        heat = heat / m
    heat = (heat * 255.0).clip(0, 255).astype(np.uint8)
    return heat

# --------------------
# 1) Stream YOLO tracking
# --------------------
print("Loading frames list (for indexing only)...")
frame_paths_sorted = natural_sorted_frame_list(FRAMES_DIR)
num_frames_fs = len(frame_paths_sorted)
print(f"Found {num_frames_fs} frames.")

print("Starting YOLO streamed tracking...")
model = YOLO(YOLO_WEIGHTS)
results_gen = model.track(
    source=FRAMES_DIR,
    stream=True,
    imgsz=IMGSZ,
    device=DEVICE,
    half=HALF,
    persist=True,
    tracker='bytetrack.yaml',
    verbose=False,
    conf=CONF_THRES,
    iou=IOU_THRES,
    max_det=MAX_DET
)

per_frame = []
yolo_names = None
for r in results_gen:
    if yolo_names is None:
        yolo_names = r.names
    dets = []
    if r.boxes is not None and len(r.boxes) > 0:
        b = r.boxes
        xyxy = b.xyxy.cpu().numpy()
        cls  = b.cls.cpu().numpy().astype(int)
        ids  = b.id.cpu().numpy().astype(int) if b.id is not None else np.arange(len(xyxy))
        for bb, c, tid in zip(xyxy, cls, ids):
            our_name = yolo_names_to_ours(yolo_names, c)
            if our_name is None:
                continue
            dets.append({
                'track_id': int(tid),
                'bbox_xyxy': [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                'our_cls_name': our_name,
                'our_cls_id': int(OUR_TO_ID[our_name])
            })
    per_frame.append(dets)

T = min(len(per_frame), num_frames_fs)
per_frame = per_frame[:T]
frame_paths_sorted = frame_paths_sorted[:T]
print(f"YOLO produced {len(per_frame)} frames of results. Using T={T} frames.")

# --------------------
# Build frame_to_all_lines (all detections as YOLO txt lines)
# --------------------
frame_to_all_lines = {}  # stem -> ["cls cx cy w h", ...]
for t in range(T):
    img_path = frame_paths_sorted[t]
    stem = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    if img is None:
        frame_to_all_lines[stem] = []
        continue
    H_use, W_use = img.shape[:2]
    all_lines = []
    for det in per_frame[t]:
        cls_id = det['our_cls_id']
        x1, y1, x2, y2 = det['bbox_xyxy']
        x1 = max(0, min(W_use - 1, x1)); x2 = max(0, min(W_use - 1, x2))
        y1 = max(0, min(H_use - 1, y1)); y2 = max(0, min(H_use - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy, bw, bh = bbox_xyxy_to_yolo_txt([x1, y1, x2, y2], W_use, H_use)
        all_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    frame_to_all_lines[stem] = all_lines

# --------------------
# 2) Load fixations & count fix-inside-bbox for Rule (1)
# --------------------
print("Loading fixation points...")
fix_per_frame = load_fixations_mat(FIX_MAT_PATH, total_frames=T)

print("Counting fixations-in-bbox per frame...")
is_fixated = defaultdict(dict)  # is_fixated[t][track_id] = True/False
for t in range(T):
    points = fix_per_frame[t]
    if len(points) == 0 or len(per_frame[t]) == 0:
        continue
    pts_xy = np.array(points, dtype=float)
    xs = pts_xy[:, 0]; ys = pts_xy[:, 1]
    for det in per_frame[t]:
        x1, y1, x2, y2 = det['bbox_xyxy']
        inside = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
        cnt = int(inside.sum())
        if cnt >= MIN_FIX_INSIDE_BBOX:
            is_fixated[t][det['track_id']] = True

# --------------------
# 3) Rule (2) temporal consistency
# --------------------
print("Applying Rule (2) temporal consistency...")
track_to_frames = defaultdict(list)
for t in range(T):
    for det in per_frame[t]:
        track_to_frames[det['track_id']].append(t)

for tid, frames_list in track_to_frames.items():
    frames_arr = np.array(sorted(set(frames_list)), dtype=int)
    if frames_arr.size == 0:
        continue
    for t in frames_arr:
        lo = max(0, t - K_WINDOW)
        hi = min(T - 1, t + K_WINDOW)
        count_k = 0
        for tau in range(lo, hi + 1):
            if is_fixated.get(tau, {}).get(tid, False):
                count_k += 1
        if count_k >= K_WINDOW:
            is_fixated[t][tid] = True

# --------------------
# 4) Propagation within N frames
# --------------------
print("Applying propagation rule within N frames...")
for tid, frames_list in track_to_frames.items():
    labeled = sorted([t for t in frames_list if is_fixated.get(t, {}).get(tid, False)])
    if len(labeled) < 2:
        continue
    for i in range(len(labeled) - 1):
        t1, t2 = labeled[i], labeled[i + 1]
        if (t2 - t1) <= N_PROPAGATE:
            for tau in range(t1 + 1, t2):
                is_fixated[tau][tid] = True

# --------------------
# 5) Export sampled frames: write YOLO bboxes (all + attended) + POGs; save saliency image
# --------------------
print("Exporting sampled frames, labels, and saliency maps...")
exported = 0
for t in range(0, T, SAMPLE_EVERY):
    img_path = frame_paths_sorted[t]
    img = cv2.imread(img_path)  # BGR
    if img is None:
        continue

    H_use, W_use = img.shape[:2]
    stem   = os.path.splitext(os.path.basename(img_path))[0]
    out_img = os.path.join(OUT_IMG_DIR, f"{stem}.jpg")
    out_lbl = os.path.join(OUT_LBL_DIR, f"{stem}.txt")
    out_vis = os.path.join(OUT_VIS_DIR, f"{stem}.jpg")
    out_sal = os.path.join(OUT_SAL_DIR, f"{stem}.png")  # saliency heatmap

    # Save/copy scene image (for completeness / visualization base)
    if img_path.lower().endswith('.jpg'):
        shutil.copy2(img_path, out_img)
        vis = cv2.imread(out_img)
    else:
        cv2.imwrite(out_img, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        vis = img.copy()

    # Build attended lines for this frame
    attended_lines = []
    fix_t = is_fixated.get(t, {})
    if len(per_frame[t]) > 0 and len(fix_t) > 0:
        for det in per_frame[t]:
            tid = det['track_id']
            if not fix_t.get(tid, False):
                continue
            cls_id = det['our_cls_id']
            x1, y1, x2, y2 = det['bbox_xyxy']
            x1 = max(0, min(W_use - 1, x1)); x2 = max(0, min(W_use - 1, x2))
            y1 = max(0, min(H_use - 1, y1)); y2 = max(0, min(H_use - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            cx, cy, bw, bh = bbox_xyxy_to_yolo_txt([x1, y1, x2, y2], W_use, H_use)
            attended_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    # Draw all detections (blue) + attended (green) + fixation points (yellow)
    all_lines = frame_to_all_lines.get(stem, [])
    for la in all_lines:
        p = la.strip().split()
        if len(p) < 5: continue
        cls_all = int(p[0]); cx_n, cy_n, w_n, h_n = map(float, p[1:5])
        x1, y1, x2, y2 = _denorm_xywh_to_xyxy(cx_n, cy_n, w_n, h_n, W_use, H_use)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = OUR_CLASSES[cls_all] if 0 <= cls_all < len(OUR_CLASSES) else str(cls_all)
        cv2.putText(vis, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    for ls in attended_lines:
        p = ls.strip().split()
        if len(p) < 5: continue
        cls_sal = int(p[0]); cx_n, cy_n, w_n, h_n = map(float, p[1:5])
        x1, y1, x2, y2 = _denorm_xywh_to_xyxy(cx_n, cy_n, w_n, h_n, W_use, H_use)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 3)
        label = OUR_CLASSES[cls_sal] if 0 <= cls_sal < len(OUR_CLASSES) else str(cls_sal)
        cv2.putText(vis, f"{label} (attended)", (x1, min(H_use - 5, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2, cv2.LINE_AA)

    # fixation points overlay (for vis only)
    fix_points = fix_per_frame[t] if t < len(fix_per_frame) else []
    for (fx, fy) in fix_points:
        cv2.circle(vis, (int(fx), int(fy)), 4, (0, 255, 255), -1)
        cv2.circle(vis, (int(fx), int(fy)), 8, (0, 255, 255), 1)

    # ---- WRITE LABEL FILE: include BOTH YOLO and ATTENDED lines, plus POGs ----
    with open(out_lbl, 'w', encoding='utf-8') as f:
        f.write("# Format:\n")
        f.write("# Y <cls> <cx_n> <cy_n> <w_n> <h_n>  (YOLO11 detection)\n")
        f.write("# A <cls> <cx_n> <cy_n> <w_n> <h_n>  (Attended object)\n")
        # 1) YOLO ALL
        for la in all_lines:
            f.write("Y " + la + "\n")
        # 2) ATTENDED
        for ls in attended_lines:
            f.write("A " + ls + "\n")
        # 3) POG meta
        f.write(f"# IMG_W {W_use} IMG_H {H_use}\n")
        f.write(f"# NUM_FIX {len(fix_points)}\n")
        for (fx, fy) in fix_points:
            cx_n = float(fx) / float(W_use) if W_use > 0 else 0.0
            cy_n = float(fy) / float(H_use) if H_use > 0 else 0.0
            f.write(f"# FIX {int(round(fx))} {int(round(fy))} {cx_n:.6f} {cy_n:.6f}\n")

    # save visualization and saliency image
    cv2.imwrite(out_vis, vis)

    sal = build_saliency_map(fix_points, H_use, W_use, var_px=200.0)
    # Save grayscale saliency; optionally you can colorize:
    # sal_color = cv2.applyColorMap(sal, cv2.COLORMAP_JET)
    cv2.imwrite(out_sal, sal)

    exported += 1

print(f"Done. Exported {exported} sampled frames with labels (+YOLO, +ATTENDED, +POGs) and saliency maps -> {OUTPUT_ROOT}")
print(f"time is: {time.time()-start:.1f}s")

