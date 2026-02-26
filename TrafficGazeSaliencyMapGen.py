import os
import glob
import math
import time

import numpy as np
import scipy.io as sio
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

start = time.time()

# =============================================================================
# CONFIG
# =============================================================================
TRAFFIC_GAZE_ROOT = r".../TrafficGaze/Traffic_Gaze"
FRAMES_ROOT       = os.path.join(TRAFFIC_GAZE_ROOT, "trafficframe")     # contains 01, 02, ..., 16
FIX_ROOT          = r".../TrafficGaze/fixdata"         # contains fixdata1.mat, ..., fixdata16.mat
OUT_SAL_ROOT      = os.path.join(TRAFFIC_GAZE_ROOT, "saliencyframe")     # target output root

# Default frame geometry (used if we cannot read the first frame)
DEFAULT_W, DEFAULT_H = 1280, 720

# Saliency heatmap params
VAR_PX = 200.0  # variance in pixels (sigma = sqrt(VAR_PX))

# If you want to process only specific videos, set e.g.: VIDEO_IDS = ["01", "02"]
VIDEO_IDS = None  # None = auto-detect all numeric folders under FRAMES_ROOT

# =============================================================================
# Utils
# =============================================================================
def natural_sorted_frame_list(folder):
    exts = ("*.jpg", "*.png", "*.jpeg", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    if not files:
        raise FileNotFoundError(f"No frames found under: {folder}")

    def keyfunc(p):
        base = os.path.basename(p)
        nums = "".join([c if c.isdigit() else " " for c in base]).split()
        return int(nums[-1]) if nums else base

    return sorted(files, key=keyfunc)

def load_fixations_mat(mat_path, frame_w, frame_h, total_frames=None):
    md = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    key = None
    for k in ["fixdata", "fixdata1", "fixdata01", "fixdata_1"]:
        if k in md:
            key = k
            break
    if key is None:
        for k in md:
            if isinstance(md[k], np.ndarray) and md[k].dtype == np.object_:
                key = k
                break
    if key is None:
        raise ValueError(f"Could not find fixdata cell array in MAT file: {mat_path}")

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

        xy = arr[:, 2:4]  # (x, y)
        xy = xy[~np.isnan(xy).any(axis=1)]
        if xy.size > 0:
            xs = np.clip(xy[:, 0], 0, frame_w - 1)
            ys = np.clip(xy[:, 1], 0, frame_h - 1)
            xy = np.stack([xs, ys], axis=1)
            all_fix.append([(float(x), float(y)) for x, y in xy])
        else:
            all_fix.append([])

    return all_fix

def build_saliency_map(points, H, W, var_px=200.0):
    if len(points) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    sigma = math.sqrt(var_px)
    radius = int(3 * sigma)  # local window = +/- 3σ
    heat = np.zeros((H, W), dtype=np.float32)

    for (x, y) in points:
        cx = int(round(x))
        cy = int(round(y))

        x1 = max(0, cx - radius)
        x2 = min(W - 1, cx + radius)
        y1 = max(0, cy - radius)
        y2 = min(H - 1, cy + radius)
        if x2 <= x1 or y2 <= y1:
            continue

        xs = np.arange(x1, x2 + 1, dtype=np.float32)
        ys = np.arange(y1, y2 + 1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)

        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
        heat[y1 : y2 + 1, x1 : x2 + 1] += g

    m = float(heat.max())
    if m > 0:
        heat /= m
    heat = (heat * 255.0).clip(0, 255).astype(np.uint8)
    return heat

def list_video_ids(frames_root, explicit_ids=None):
    if explicit_ids is not None:
        return sorted(explicit_ids)
    vids = []
    for name in os.listdir(frames_root):
        full = os.path.join(frames_root, name)
        if os.path.isdir(full) and name.isdigit():
            vids.append(name.zfill(2))
    return sorted(set(vids))

def find_fix_mat(fix_root, vid_str):
    # trafficframe/01 -> fixdata1.mat (based on your example)
    vid_int = int(vid_str)
    cand = os.path.join(fix_root, f"fixdata{vid_int}.mat")
    if os.path.isfile(cand):
        return cand

    # fallback patterns (in case your naming differs)
    patterns = [
        os.path.join(fix_root, f"*{vid_int}*.mat"),
        os.path.join(fix_root, f"*{vid_str}*.mat"),
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            hits = sorted(hits)
            return hits[0]

    raise FileNotFoundError(f"Could not locate fixation MAT for video {vid_str} under: {fix_root}")

# =============================================================================
# MAIN: generate saliency map for EVERY frame of EVERY video
# =============================================================================
os.makedirs(OUT_SAL_ROOT, exist_ok=True)

video_ids = list_video_ids(FRAMES_ROOT, VIDEO_IDS)
if not video_ids:
    raise RuntimeError(f"No video folders found under: {FRAMES_ROOT}")

total_written = 0

for vid in video_ids:
    frames_dir = os.path.join(FRAMES_ROOT, vid)
    if not os.path.isdir(frames_dir):
        continue

    fix_mat_path = find_fix_mat(FIX_ROOT, vid)
    out_dir = os.path.join(OUT_SAL_ROOT, vid)  # keep per-video subfolder to avoid name collisions
    os.makedirs(out_dir, exist_ok=True)

    frame_paths = natural_sorted_frame_list(frames_dir)
    T = len(frame_paths)

    # Determine H/W from the first readable frame (fallback to defaults)
    W_use, H_use = DEFAULT_W, DEFAULT_H
    first_img = cv2.imread(frame_paths[0])
    if first_img is not None:
        H_use, W_use = first_img.shape[:2]

    print(f"[{vid}] Frames: {T} | Fix MAT: {os.path.basename(fix_mat_path)} | Size: {W_use}x{H_use}")

    fix_per_frame = load_fixations_mat(
        fix_mat_path,
        frame_w=W_use,
        frame_h=H_use,
        total_frames=T
    )

    written_vid = 0
    for t in range(T):
        stem = os.path.splitext(os.path.basename(frame_paths[t]))[0]
        out_sal_path = os.path.join(out_dir, f"{stem}.png")

        fix_points = fix_per_frame[t] if t < len(fix_per_frame) else []
        sal = build_saliency_map(fix_points, H_use, W_use, var_px=VAR_PX)

        cv2.imwrite(out_sal_path, sal)
        written_vid += 1

    total_written += written_vid
    print(f"[{vid}] Saved {written_vid} saliency maps -> {out_dir}")

print(f"Done. Total saliency maps saved: {total_written}")
print(f"time is: {time.time() - start:.1f}s")

