#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved surfing detection with activity classification.
Replaces velocity-only logic with trained activity classifier to reduce false positives.
"""

import os
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

try:
    import torch
except Exception:
    torch = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# -----------------------------
# Activity Classifier
# -----------------------------

class ActivityClassifier:
    """Activity classifier for surfing detection"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = self._select_device(device)
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _select_device(self, device: str):
        if device == 'auto':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _load_model(self, model_path: str):
        # Create model architecture
        model = models.efficientnet_b0(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
        # Load trained weights (handle PyTorch 2.6+ weights_only change)
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception:
            # Fallback for models saved with numpy objects
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, image_crop: np.ndarray, threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Predict if the crop shows active surfing
        
        Args:
            image_crop: BGR image crop containing person+surfboard
            threshold: Classification threshold (default 0.5)
            
        Returns:
            (is_active_surfing, confidence_score)
        """
        # Convert BGR to RGB
        if len(image_crop.shape) == 3:
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        input_tensor = self.transform(image_crop).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence = probabilities[0][1].item()  # Probability of active surfing
            is_active = confidence > threshold
        
        return is_active, confidence


# -----------------------------
# Utils (from original process.py)
# -----------------------------

def pick_device(user: Optional[str] = None) -> str:
    if user and user != 'auto':
        return user
    if torch is not None:
        try:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                return 'cuda'
        except Exception:
            pass
    return 'cpu'


def iou_xyxy(a, b):
    xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
    iw = max(0, xi2 - xi1); ih = max(0, yi2 - yi1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def center(box):
    return ((box[0]+box[2])/2.0, (box[1]+box[3])/2.0)


def l2(p, q):
    return math.hypot(p[0]-q[0], p[1]-q[1])


def sec_to_tc(sec: float) -> str:
    """Format seconds to mm:ss or hh:mm:ss; keep .2f if not an integer."""
    if sec >= 3600:
        h = int(sec // 3600); sec -= h*3600
        m = int(sec // 60); sec -= m*60
        return f"{h}:{m:02d}:{int(round(sec)):02d}" if abs(sec-round(sec))<1e-3 else f"{h}:{m:02d}:{sec:05.2f}"
    m = int(sec // 60); s = sec - m*60
    return f"{m}:{int(round(s)):02d}" if abs(s-round(s))<1e-3 else f"{m}:{s:05.2f}"


def merge_segments_final(segments: List[Tuple[float, float]], gap: float = 2.0) -> List[Tuple[float, float]]:
    """Merge segments that overlap or are within `gap` seconds apart."""
    if not segments:
        return []
    segs = sorted(segments, key=lambda x: x[0])
    merged: List[List[float]] = [[segs[0][0], segs[0][1]]]
    eps = 1e-3
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if s <= pe + gap + eps:  # overlap or close enough
            merged[-1][1] = max(pe, e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def parse_timecode(s: str) -> float:
    """Parse s, m:s or h:m:s into seconds (float allowed for seconds)."""
    s = s.strip()
    if s.count(':') == 2:
        h, m, sec = s.split(':'); return int(h)*3600 + int(m)*60 + float(sec)
    if s.count(':') == 1:
        m, sec = s.split(':'); return int(m)*60 + float(sec)
    return float(s)


def smooth_mask(mask: List[Tuple[float,int]], max_gap: float) -> List[Tuple[float,int]]:
    """Flip short 0-islands to 1 when surrounded by 1s and duration <= max_gap."""
    if not mask:
        return mask
    out = mask[:]
    n = len(out)
    i = 0
    while i < n:
        if out[i][1] == 0:
            j = i
            while j < n and out[j][1] == 0:
                j += 1
            prev_t = out[i-1][0] if i > 0 else None
            next_t = out[j][0] if j < n else None
            if prev_t is not None and next_t is not None and (next_t - prev_t) <= max_gap:
                for k in range(i, j):
                    out[k] = (out[k][0], 1)  # fill the gap
            i = j
        else:
            i += 1
    return out


# -----------------------------
# Core Detection with Activity Classifier
# -----------------------------

def detect_rides_batched(
    video_path: Path,
    model_path: str,
    device: str,
    conf: float,
    frame_stride: int,
    batch_size: int,
    iou_thresh: float,
    near_px: int,
    min_seg_sec: float,
    merge_gap_sec: float,
    preroll: float,
    postroll: float,
    preview: bool,
    imgsz: Optional[int] = None,
    # Activity classifier settings
    activity_model_path: Optional[str] = None,
    activity_threshold: float = 0.5,
    use_velocity_fallback: bool = True,
    vx_thresh: float = 60.0,
    # debug controls
    log_every: int = 0,
    save_debug: bool = False,
    debug_every: int = 300,
    debug_dir: Optional[Path] = None,
    # smoothing
    use_smoothing: bool = False,
    smooth_gap: Optional[float] = None,
    # test range
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> Tuple[List[Tuple[float,float]], List[Tuple[float,int]]]:
    
    # Lazy import so --self-test can run without ultralytics installed
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics not installed. Install with:\n  pip install ultralytics\n") from e

    model = YOLO(model_path)
    predict_device = 0 if device == 'cuda' else device
    
    # Load activity classifier if provided
    activity_classifier = None
    if activity_model_path and Path(activity_model_path).exists():
        try:
            activity_classifier = ActivityClassifier(activity_model_path, device)
            print(f"[INFO] Loaded activity classifier: {activity_model_path}")
        except Exception as e:
            print(f"[WARN] Failed to load activity classifier: {e}")
            if not use_velocity_fallback:
                raise

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine processing range
    if start_sec is None:
        start_sec = 0.0
    if end_sec is None:
        end_sec = (total_frames / fps)
    # Clamp
    start_sec = max(0.0, min(start_sec, total_frames / fps))
    end_sec = max(start_sec, min(end_sec, total_frames / fps))

    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    total_samples = max(1, (max(0, end_frame - start_frame + 1)) // max(1, frame_stride))
    pbar = tqdm(total=total_samples, desc="Detecting rides", unit="frames") if tqdm is not None else None

    # State for velocity fallback
    last_center = None
    last_time = None

    ride_mask: List[Tuple[float,int]] = []  # (t, 0/1)
    yes_total = 0
    segments: List[Tuple[float,float]] = []

    frames_batch: List[np.ndarray] = []
    times_batch: List[float] = []

    if save_debug and debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    sample_idx = 0  # counts sampled frames only
    fidx = start_frame

    while True:
        # Stop if we've reached the end of requested range
        if fidx > end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            # flush remaining batch
            if frames_batch:
                batch_len = len(frames_batch)
                last_center, last_time, batch_yes = process_batch(
                    model, frames_batch, times_batch, predict_device, conf, imgsz,
                    iou_thresh, near_px, vx_thresh,
                    ride_mask, preview, last_center, last_time,
                    log_every, save_debug, debug_every, debug_dir,
                    sample_start_idx=sample_idx-batch_len,
                    activity_classifier=activity_classifier,
                    activity_threshold=activity_threshold,
                    use_velocity_fallback=use_velocity_fallback
                )
                yes_total += batch_yes
                if pbar:
                    pbar.update(batch_len)
                    last_t = times_batch[-1] if times_batch else (fidx / fps)
                    pbar.set_postfix_str(f"time~{last_t:.1f}s YES={yes_total}")
            break

        if fidx % max(1, frame_stride) != 0:
            fidx += 1
            continue

        t = fidx / fps
        frames_batch.append(frame)
        times_batch.append(t)
        sample_idx += 1

        if len(frames_batch) >= max(1, batch_size):
            batch_len = len(frames_batch)
            last_center, last_time, batch_yes = process_batch(
                model, frames_batch, times_batch, predict_device, conf, imgsz,
                iou_thresh, near_px, vx_thresh,
                ride_mask, preview, last_center, last_time,
                log_every, save_debug, debug_every, debug_dir,
                sample_start_idx=sample_idx-batch_len,
                activity_classifier=activity_classifier,
                activity_threshold=activity_threshold,
                use_velocity_fallback=use_velocity_fallback
            )
            yes_total += batch_yes
            if pbar:
                pbar.update(batch_len)
                last_t = times_batch[-1] if times_batch else (fidx / fps)
                pbar.set_postfix_str(f"time~{last_t:.1f}s YES={yes_total}")
            frames_batch, times_batch = [], []

        fidx += 1

    cap.release()
    if pbar:
        pbar.close()

    # Optional smoothing BEFORE building segments
    if use_smoothing:
        gap = smooth_gap if (smooth_gap is not None) else merge_gap_sec
        ride_mask = smooth_mask(ride_mask, gap)

    # Combine ride_mask into segments (merge first, filter after)
    if not ride_mask:
        return [], []

    curr_start = None
    raw_segments: List[Tuple[float,float]] = []
    for (t, flag) in ride_mask:
        if flag and curr_start is None:
            curr_start = t
        elif (not flag) and (curr_start is not None):
            raw_segments.append((curr_start, t))
            curr_start = None
    if curr_start is not None:
        seg_end = (end_frame / fps)
        raw_segments.append((curr_start, seg_end))

    # MERGE FIRST
    raw_segments.sort()
    merged: List[List[float]] = []
    for s,e in raw_segments:
        if not merged:
            merged.append([s,e])
        else:
            ps,pe = merged[-1]
            if s - pe <= merge_gap_sec:
                merged[-1][1] = max(pe, e)
            else:
                merged.append([s,e])

    # Filter min length AFTER merging
    merged = [se for se in merged if (se[1] - se[0]) >= min_seg_sec]

    # Clamp with preroll/postroll (to overall video range)
    dur = total_frames / fps
    final: List[Tuple[float,float]] = []
    for s,e in merged:
        ss = max(0.0, s - preroll)
        ee = min(dur, e + postroll)
        if ee - ss >= 0.3:
            final.append((ss, ee))

    return final, ride_mask


def process_batch(model, frames_batch, times_batch, device, conf, imgsz,
                  iou_thresh, near_px, vx_thresh,
                  ride_mask, preview, last_center, last_time,
                  log_every, save_debug, debug_every, debug_dir, sample_start_idx=0,
                  activity_classifier=None, activity_threshold=0.5, use_velocity_fallback=True):
    """
    Enhanced batch processing with activity classification.
    Replaces velocity-only logic with trained activity classifier.
    """
    pred_kwargs = dict(conf=conf, verbose=False, device=device)
    if imgsz:
        pred_kwargs['imgsz'] = int(imgsz)

    results = model.predict(frames_batch, **pred_kwargs)

    lc = last_center
    lt = last_time
    batch_yes = 0

    for i, (frame, t, res) in enumerate(zip(frames_batch, times_batch, results)):
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0,4))
        cls = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros((0,), dtype=int)
        names = res.names
        person_boxes = [boxes[i] for i,c in enumerate(cls) if names[int(c)] == 'person']
        board_boxes  = [boxes[i] for i,c in enumerate(cls) if names[int(c)] == 'surfboard']

        # Pair by IoU/proximity (same logic as original)
        best_pair = None
        best_score = 0.0
        for pb in person_boxes:
            for sb in board_boxes:
                iou = iou_xyxy(pb, sb)
                cp, cs = center(pb), center(sb)
                near = max(0.0, (near_px - l2(cp, cs))) / max(near_px,1)
                score = max(iou, 0.6 * near)
                if score > best_score:
                    best_score = score
                    best_pair = (pb, sb, cp, cs, iou)

        flag = 0
        paired = best_pair is not None and (best_pair[4] >= iou_thresh or l2(best_pair[2], best_pair[3]) <= near_px)
        
        if paired:
            # NEW: Activity classification instead of velocity-only
            if activity_classifier is not None:
                try:
                    # Extract crop from person+surfboard region
                    pb, sb, cp, cs, iou = best_pair
                    
                    # Create bounding box that encompasses both person and surfboard
                    min_x = max(0, int(min(pb[0], sb[0]) - 20))
                    min_y = max(0, int(min(pb[1], sb[1]) - 20))
                    max_x = min(frame.shape[1], int(max(pb[2], sb[2]) + 20))
                    max_y = min(frame.shape[0], int(max(pb[3], sb[3]) + 20))
                    
                    crop = frame[min_y:max_y, min_x:max_x]
                    
                    if crop.size > 0:
                        is_active, confidence = activity_classifier.predict(crop, activity_threshold)
                        flag = 1 if is_active else 0
                        
                        # Update center for velocity fallback if needed
                        curr_center = ((cp[0] + cs[0]) / 2.0, (cp[1] + cs[1]) / 2.0)
                        lc, lt = curr_center, t
                    else:
                        # Fallback to velocity if crop extraction fails
                        if use_velocity_fallback:
                            flag = velocity_check(best_pair, lc, lt, t, vx_thresh)
                            if flag:
                                curr_center = ((best_pair[2][0] + best_pair[3][0]) / 2.0,
                                              (best_pair[2][1] + best_pair[3][1]) / 2.0)
                                lc, lt = curr_center, t
                        
                except Exception as e:
                    # Fallback to velocity on classifier error
                    if use_velocity_fallback:
                        flag = velocity_check(best_pair, lc, lt, t, vx_thresh)
                        if flag:
                            curr_center = ((best_pair[2][0] + best_pair[3][0]) / 2.0,
                                          (best_pair[2][1] + best_pair[3][1]) / 2.0)
                            lc, lt = curr_center, t
                    else:
                        print(f"[WARN] Activity classifier failed: {e}")
            else:
                # Fallback to original velocity-based approach
                flag = velocity_check(best_pair, lc, lt, t, vx_thresh)
                if flag:
                    curr_center = ((best_pair[2][0] + best_pair[3][0]) / 2.0,
                                  (best_pair[2][1] + best_pair[3][1]) / 2.0)
                    lc, lt = curr_center, t
            
            if flag == 1:
                batch_yes += 1
        
        ride_mask.append((t, flag))

        # Logging
        global_idx = sample_start_idx + i + 1
        if log_every and (global_idx % log_every == 0):
            status = 'YES' if flag == 1 else 'NO'
            method = 'CLASSIFIER' if activity_classifier and paired else 'VELOCITY'
            print(f"[DBG] sample#{global_idx} t={t:.2f}s persons={len(person_boxes)} boards={len(board_boxes)} paired={int(bool(paired))} method={method} -> {status}")

        # Debug snapshot
        if save_debug and (global_idx % max(1, debug_every) == 0):
            vis = frame.copy()
            for b in person_boxes:
                x1,y1,x2,y2 = map(int,b); cv2.rectangle(vis,(x1,y1),(x2,y2),(255,255,0),2)
            for b in board_boxes:
                x1,y1,x2,y2 = map(int,b); cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,255),2)
            label = 'RIDE' if flag == 1 else 'NO-RIDE'
            color = (0,255,0) if flag == 1 else (0,0,255)
            cv2.putText(vis, f"{label} t={t:.2f}s", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if paired:
                ccx = int((best_pair[2][0] + best_pair[3][0]) / 2.0)
                ccy = int((best_pair[2][1] + best_pair[3][1]) / 2.0)
                cv2.circle(vis, (ccx, ccy), 6, (0,255,0), -1)
            if debug_dir is not None:
                outp = debug_dir / f"sample_{global_idx:06d}_t{t:.2f}.jpg"
                cv2.imwrite(str(outp), vis)

    return lc, lt, batch_yes


def velocity_check(best_pair, last_center, last_time, current_time, vx_thresh):
    """Original velocity-based activity check"""
    curr_center = ((best_pair[2][0] + best_pair[3][0]) / 2.0,
                   (best_pair[2][1] + best_pair[3][1]) / 2.0)
    
    if last_center is not None and last_time is not None:
        dt = max(1e-3, current_time - last_time)
        vx = abs(curr_center[0] - last_center[0]) / dt
        return 1 if vx >= vx_thresh else 0
    else:
        return 1  # First detection, assume active


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description='Improved surfing detection with activity classification')
    ap.add_argument('--input', required=True, help='Input video')
    ap.add_argument('--outdir', default='clips', help='Output directory')
    ap.add_argument('--model', default='yolov8n.pt', help='YOLOv8 detection model')
    ap.add_argument('--device', default='auto', choices=['auto','cpu','cuda','mps'], help='Device')
    ap.add_argument('--conf', type=float, default=0.25, help='Detection confidence')
    ap.add_argument('--frame-stride', type=int, default=5, help='Sample every N frames')
    ap.add_argument('--batch-size', type=int, default=32, help='Frames per inference batch')
    ap.add_argument('--imgsz', type=int, default=512, help='Model input size (try 512/640/768 for 1080p)')
    ap.add_argument('--iou-thresh', type=float, default=0.05, help='IoU threshold for pairing')
    ap.add_argument('--near-px', type=int, default=80, help='Proximity threshold (px)')
    ap.add_argument('--vx-thresh', type=float, default=60.0, help='Horizontal speed threshold (px/s) for fallback')
    ap.add_argument('--min-seg-sec', type=float, default=0.8, help='Minimum segment duration (s)')
    ap.add_argument('--merge-gap-sec', type=float, default=1.0, help='Merge gap (s)')
    ap.add_argument('--preroll', type=float, default=1.5, help='Seconds before start')
    ap.add_argument('--postroll', type=float, default=1.0, help='Seconds after end')
    ap.add_argument('--preview', action='store_false', help='Show quick preview overlays (slower)')

    # Activity classifier settings
    ap.add_argument('--activity-model', default='models/activity_classifier.pth', help='Path to trained activity classifier')
    ap.add_argument('--activity-threshold', type=float, default=0.3, help='Activity classification threshold (lower = more sensitive to surfing)')
    ap.add_argument('--no-velocity-fallback', action='store_true', help='Disable velocity fallback if classifier fails')

    # Debug / progress controls
    ap.add_argument('--log-every', type=int, default=0, help='Print debug info every N sampled frames (0=off)')
    ap.add_argument('--save-debug', action='store_true', help='Save debug frames with boxes to outdir/debug')
    ap.add_argument('--debug-every', type=int, default=300, help='Save one debug frame every N sampled frames')

    # Mask smoothing
    ap.add_argument('--smooth', action='store_true', help='Fill short 0-gaps in the ride mask before segmenting')
    ap.add_argument('--smooth-gap', type=float, default=None, help='Max gap (s) to fill when --smooth is enabled (default: --merge-gap-sec)')

    # Final merge-pass knob
    ap.add_argument('--final-merge-gap-sec', type=float, default=2.0,
                    help='Final pass: merge segments that overlap or are within this many seconds')

    # Test window: process only start-end (e.g., 2:30-4:00)
    ap.add_argument('--test', type=str, default=None,
                    help='Quick test window "start-end" (supports s, m:s, or h:m:s). Example: --test "2:30-4:00"')

    # Self-test (no model required)
    ap.add_argument('--self-test', action='store_true', help='Write sample segments/mask without running YOLO')

    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")
    os.makedirs(args.outdir, exist_ok=True)

    # Self-test path to validate file output without heavy inference
    if args.self_test:
        segfile = Path(args.outdir) / f"{inp.stem}_segments.txt"
        with open(segfile, 'w', encoding='utf-8', newline='') as f:
            for s, e in [(10.0, 12.5), (90.0, 95.2), (123.4, 130.0)]:
                f.write(f"{sec_to_tc(s)}-{sec_to_tc(e)}\n")
        maskfile = Path(args.outdir) / f"{inp.stem}_ride_mask.tsv"
        with open(maskfile, 'w', encoding='utf-8') as f:
            for t in [9.9, 10.0, 10.1, 12.4, 12.5]:
                f.write(f"{round(t,3)}\t{1 if 10.0 <= t <= 12.5 else 0}\n")
        print(f"[SELF-TEST] Segments saved: {segfile}")
        print(f"[SELF-TEST] Ride mask saved: {maskfile}")
        return

    # Parse --test window if supplied
    test_start, test_end = None, None
    if args.test:
        try:
            s_str, e_str = [p for p in args.test.split('-') if p.strip()]
            test_start = parse_timecode(s_str)
            test_end = parse_timecode(e_str)
            if test_end <= test_start:
                raise ValueError('end must be greater than start')
        except Exception as ex:
            raise SystemExit(f"Invalid --test format. Use e.g. --test '2:30-4:00' or '150-240'. Error: {ex}")
        print(f"[TEST] Limiting processing to {sec_to_tc(test_start)} - {sec_to_tc(test_end)}")

    device = pick_device(args.device)
    print(f"[INFO] Using device: {device}")

    debug_dir = Path(args.outdir) / 'debug' if args.save_debug else None

    # Check for activity classifier
    activity_model_path = None
    if Path(args.activity_model).exists():
        activity_model_path = args.activity_model
        print(f"[INFO] Activity classifier found: {activity_model_path}")
        print(f"[INFO] Activity threshold: {args.activity_threshold} (lower = more sensitive)")
    else:
        print(f"[WARN] Activity classifier not found: {args.activity_model}")
        print(f"[WARN] Falling back to velocity-only detection")

    segments, ride_mask = detect_rides_batched(
        video_path=inp,
        model_path=args.model,
        device=device,
        conf=args.conf,
        frame_stride=max(1, args.frame_stride),
        batch_size=max(1, args.batch_size),
        iou_thresh=args.iou_thresh,
        near_px=args.near_px,
        min_seg_sec=args.min_seg_sec,
        merge_gap_sec=args.merge_gap_sec,
        preroll=args.preroll,
        postroll=args.postroll,
        preview=args.preview,
        imgsz=args.imgsz,
        log_every=max(0, args.log_every),
        save_debug=bool(args.save_debug),
        debug_every=max(1, args.debug_every),
        debug_dir=debug_dir,
        use_smoothing=bool(args.smooth),
        smooth_gap=args.smooth_gap,
        start_sec=test_start,
        end_sec=test_end,
        # Activity classifier parameters
        activity_model_path=activity_model_path,
        activity_threshold=args.activity_threshold,
        use_velocity_fallback=not args.no_velocity_fallback,
        vx_thresh=args.vx_thresh,
    )

    # Final pass: merge overlaps or segments within N seconds
    segments = merge_segments_final(segments, gap=args.final_merge_gap_sec)

    # Write segments
    segfile = Path(args.outdir) / f"{inp.stem}_segments.txt"
    with open(segfile, 'w', encoding='utf-8', newline='') as f:
        for s, e in segments:
            line = f"{sec_to_tc(s)}-{sec_to_tc(e)}\n"
            f.write(line)
    print(f"[INFO] Segments saved: {segfile} ({len(segments)} clips)")

    # Write ride mask (debug)
    maskfile = Path(args.outdir) / f"{inp.stem}_ride_mask.tsv"
    with open(maskfile, 'w', encoding='utf-8') as f:
        for t, flag in ride_mask:
            f.write(f"{round(t,3)}\t{flag}\n")
    print(f"[INFO] Ride mask saved: {maskfile} ({len(ride_mask)} sampled frames)")

    if len(segments) == 0:
        print("[HINT] No segments found. Try: --activity-threshold 0.2 (more sensitive), --vx-thresh 40 (looser), --near-px 100, --imgsz 768, --frame-stride 1, or --conf 0.20")


if __name__ == '__main__':
    main()