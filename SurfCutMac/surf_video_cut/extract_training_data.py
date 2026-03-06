#!/usr/bin/env python3
"""
Extract training data for activity classification from detection results.
Creates crops of person+surfboard pairs labeled as active_surfing vs sitting/waiting.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse


def parse_timecode(tc: str) -> float:
    """Parse mm:ss.ss or hh:mm:ss.ss to seconds"""
    parts = tc.strip().split(':')
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    elif len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    else:
        return float(parts[0])


def parse_segments_file(file_path: Path) -> List[Tuple[float, float]]:
    """Parse segments file into list of (start, end) tuples"""
    segments = []
    if not file_path.exists():
        return segments
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            time_range = parts[0]
            
            if '-' not in time_range:
                continue
                
            start_str, end_str = time_range.split('-', 1)
            start_sec = parse_timecode(start_str.strip())
            end_sec = parse_timecode(end_str.strip())
            segments.append((start_sec, end_sec))
    
    return segments


def is_in_segments(timestamp: float, segments: List[Tuple[float, float]], 
                   tolerance: float = 0.5) -> bool:
    """Check if timestamp falls within any segment (with tolerance)"""
    for start, end in segments:
        if start - tolerance <= timestamp <= end + tolerance:
            return True
    return False


def run_detection_with_debug(video_path: Path, output_dir: Path, 
                           device: str = 'auto') -> Path:
    """Run detection with debug frames enabled to extract crops"""
    print(f"Running detection with debug on {video_path.name}...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run process.py with debug enabled
    import subprocess
    cmd = [
        'python3', 'process.py',
        '--input', str(video_path),
        '--outdir', str(output_dir),
        '--device', device,
        '--imgsz', '640',
        '--frame-stride', '10',  # Reduced for better temporal resolution
        '--batch-size', '16',    # Smaller batch for better debug handling
        '--vx-thresh', '60.0',
        '--conf', '0.25',
        '--save-debug',          # Enable debug frame saving
        '--debug-every', '10',   # Save more debug frames
        '--log-every', '50'      # Enable logging
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running detection: {result.stderr}")
        raise RuntimeError(f"Detection failed for {video_path}")
    
    # Find the debug directory
    debug_dir = output_dir / 'debug'
    if not debug_dir.exists():
        raise RuntimeError(f"Debug directory not found: {debug_dir}")
    
    return debug_dir


def extract_crops_from_debug(debug_dir: Path, golden_segments: List[Tuple[float, float]],
                           output_dir: Path, video_name: str) -> Dict:
    """Extract person+surfboard crops from debug frames and label them"""
    
    # Create output directories for crops
    positive_dir = output_dir / 'positive' / video_name
    negative_dir = output_dir / 'negative' / video_name
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all debug frames
    debug_frames = sorted(debug_dir.glob("sample_*.jpg"))
    print(f"Found {len(debug_frames)} debug frames")
    
    positive_count = 0
    negative_count = 0
    metadata = []
    
    for frame_path in debug_frames:
        # Extract timestamp from filename: sample_000123_t45.67.jpg
        try:
            filename = frame_path.stem
            parts = filename.split('_t')
            if len(parts) != 2:
                continue
            timestamp = float(parts[1])
        except (ValueError, IndexError):
            print(f"Could not parse timestamp from {frame_path.name}")
            continue
        
        # Load frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        
        # Determine if this timestamp corresponds to active surfing
        is_positive = is_in_segments(timestamp, golden_segments)
        
        # For now, we'll save the entire frame since we don't have box coordinates
        # In a more sophisticated version, we'd extract the actual person+board crop
        # from the YOLO detection boxes
        
        if is_positive:
            output_path = positive_dir / f"surf_{positive_count:04d}_t{timestamp:.2f}.jpg"
            positive_count += 1
        else:
            output_path = negative_dir / f"wait_{negative_count:04d}_t{timestamp:.2f}.jpg"
            negative_count += 1
        
        # Save crop (for now, save entire frame)
        cv2.imwrite(str(output_path), frame)
        
        # Store metadata
        metadata.append({
            'original_frame': str(frame_path),
            'crop_path': str(output_path),
            'timestamp': timestamp,
            'is_active_surfing': is_positive,
            'video_name': video_name
        })
    
    print(f"Extracted {positive_count} positive and {negative_count} negative examples")
    
    return {
        'positive_count': positive_count,
        'negative_count': negative_count,
        'metadata': metadata
    }


def create_improved_training_extractor(video_path: Path, golden_segments: List[Tuple[float, float]],
                                     output_dir: Path, device: str = 'auto') -> Dict:
    """
    Improved training data extraction that processes video frame by frame
    and extracts actual person+surfboard crops using YOLO detection
    """
    
    # Import required modules
    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        raise RuntimeError("Required packages not installed. Run: pip install ultralytics torch") from e
    
    print(f"Processing {video_path.name} for training data extraction...")
    
    # Load YOLO model
    model = YOLO('yolov8n.pt')
    predict_device = 0 if device == 'cuda' else device
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directories
    video_name = video_path.stem
    positive_dir = output_dir / 'positive' / video_name
    negative_dir = output_dir / 'negative' / video_name
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)
    
    positive_count = 0
    negative_count = 0
    metadata = []
    
    frame_stride = 30  # Sample every 30 frames (1 per second at 30fps)
    frame_idx = 0
    
    print(f"Processing video with {total_frames} frames at {fps} fps...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames based on stride
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        
        timestamp = frame_idx / fps
        
        # Run YOLO detection on this frame
        results = model.predict([frame], conf=0.25, verbose=False, device=predict_device)
        
        if not results or not results[0].boxes:
            frame_idx += 1
            continue
        
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        names = res.names
        
        # Find person and surfboard detections
        person_boxes = [boxes[i] for i, c in enumerate(cls) if names[int(c)] == 'person']
        board_boxes = [boxes[i] for i, c in enumerate(cls) if names[int(c)] == 'surfboard']
        
        # Find best person+surfboard pair (using same logic as process.py)
        best_pair = None
        best_score = 0.0
        
        for pb in person_boxes:
            for sb in board_boxes:
                # Calculate IoU
                xi1, yi1 = max(pb[0], sb[0]), max(pb[1], sb[1])
                xi2, yi2 = min(pb[2], sb[2]), min(pb[3], sb[3])
                iw, ih = max(0, xi2 - xi1), max(0, yi2 - yi1)
                inter = iw * ih
                
                if inter <= 0:
                    iou = 0.0
                else:
                    area_pb = (pb[2] - pb[0]) * (pb[3] - pb[1])
                    area_sb = (sb[2] - sb[0]) * (sb[3] - sb[1])
                    iou = inter / (area_pb + area_sb - inter + 1e-6)
                
                # Calculate proximity
                cp = ((pb[0] + pb[2]) / 2.0, (pb[1] + pb[3]) / 2.0)
                cs = ((sb[0] + sb[2]) / 2.0, (sb[1] + sb[3]) / 2.0)
                dist = np.sqrt((cp[0] - cs[0])**2 + (cp[1] - cs[1])**2)
                near_px = 80
                near = max(0.0, (near_px - dist)) / max(near_px, 1)
                
                score = max(iou, 0.6 * near)
                if score > best_score:
                    best_score = score
                    best_pair = (pb, sb, cp, cs, iou)
        
        # If we found a good pair, extract crop
        if best_pair and (best_pair[4] >= 0.05 or np.sqrt((best_pair[2][0] - best_pair[3][0])**2 + 
                                                          (best_pair[2][1] - best_pair[3][1])**2) <= 80):
            
            pb, sb, cp, cs, iou = best_pair
            
            # Create bounding box that encompasses both person and surfboard
            min_x = min(pb[0], sb[0])
            min_y = min(pb[1], sb[1])
            max_x = max(pb[2], sb[2])
            max_y = max(pb[3], sb[3])
            
            # Add some padding
            padding = 20
            min_x = max(0, int(min_x - padding))
            min_y = max(0, int(min_y - padding))
            max_x = min(frame.shape[1], int(max_x + padding))
            max_y = min(frame.shape[0], int(max_y + padding))
            
            # Extract crop
            crop = frame[min_y:max_y, min_x:max_x]
            
            if crop.size > 0:
                # Determine label based on golden segments
                is_positive = is_in_segments(timestamp, golden_segments, tolerance=1.0)
                
                if is_positive:
                    output_path = positive_dir / f"surf_{positive_count:04d}_t{timestamp:.2f}.jpg"
                    positive_count += 1
                else:
                    output_path = negative_dir / f"wait_{negative_count:04d}_t{timestamp:.2f}.jpg"
                    negative_count += 1
                
                # Resize crop to standard size
                crop_resized = cv2.resize(crop, (224, 224))
                cv2.imwrite(str(output_path), crop_resized)
                
                # Store metadata
                metadata.append({
                    'crop_path': str(output_path),
                    'timestamp': timestamp,
                    'is_active_surfing': is_positive,
                    'video_name': video_name,
                    'bbox': [min_x, min_y, max_x, max_y],
                    'person_box': pb.tolist(),
                    'surfboard_box': sb.tolist(),
                    'iou': float(iou),
                    'score': float(best_score)
                })
        
        frame_idx += 1
        
        # Progress update
        if frame_idx % (frame_stride * 30) == 0:
            print(f"Processed {frame_idx}/{total_frames} frames, extracted {positive_count + negative_count} crops")
    
    cap.release()
    
    print(f"Extraction complete: {positive_count} positive, {negative_count} negative examples")
    
    return {
        'positive_count': positive_count,
        'negative_count': negative_count,
        'metadata': metadata,
        'video_name': video_name
    }


def main():
    parser = argparse.ArgumentParser(description='Extract training data for activity classification')
    parser.add_argument('--test-dir', default='create_test_case_2',
                       help='Directory containing test videos')
    parser.add_argument('--output-dir', default='training_data',
                       help='Directory for extracted training crops')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device for YOLO inference')
    parser.add_argument('--method', default='improved', choices=['debug', 'improved'],
                       help='Extraction method: debug (from saved debug frames) or improved (direct processing)')
    
    args = parser.parse_args()
    
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all test video directories
    video_dirs = [d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith('clips')]
    video_dirs.sort()
    
    print(f"Found {len(video_dirs)} test video directories")
    
    all_results = {}
    total_positive = 0
    total_negative = 0
    
    for video_dir in video_dirs:
        print(f"\n{'='*60}")
        print(f"Processing {video_dir.name}")
        print('='*60)
        
        # Find video file and golden reference
        video_files = list(video_dir.glob("*.mp4"))
        golden_files = list(video_dir.glob("*_golden.txt"))
        
        if not video_files or not golden_files:
            print(f"Missing video or golden file in {video_dir}")
            continue
        
        video_path = video_files[0]
        golden_path = golden_files[0]
        
        # Load golden segments
        golden_segments = parse_segments_file(golden_path)
        print(f"Golden segments: {len(golden_segments)}")
        
        try:
            if args.method == 'improved':
                # Use improved extraction method
                result = create_improved_training_extractor(
                    video_path, golden_segments, output_dir, args.device
                )
            else:
                # Use debug frame method (legacy)
                debug_dir = run_detection_with_debug(video_path, output_dir / 'temp' / video_dir.name, args.device)
                result = extract_crops_from_debug(debug_dir, golden_segments, output_dir, video_path.stem)
            
            all_results[video_dir.name] = result
            total_positive += result['positive_count']
            total_negative += result['negative_count']
            
        except Exception as e:
            print(f"Error processing {video_dir.name}: {e}")
            continue
    
    # Save metadata
    metadata_file = output_dir / 'training_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump({
            'method': args.method,
            'total_positive_examples': total_positive,
            'total_negative_examples': total_negative,
            'results_by_video': all_results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print('='*60)
    print(f"Total Positive Examples (Active Surfing): {total_positive}")
    print(f"Total Negative Examples (Sitting/Waiting): {total_negative}")
    print(f"Total Examples: {total_positive + total_negative}")
    if total_positive + total_negative > 0:
        print(f"Class Balance: {total_positive/(total_positive+total_negative)*100:.1f}% positive")
    else:
        print("No training examples extracted!")
    print(f"Metadata saved to: {metadata_file}")
    print(f"\nTraining data structure:")
    print(f"  {output_dir}/positive/  - Active surfing examples")
    print(f"  {output_dir}/negative/  - Sitting/waiting examples")


if __name__ == '__main__':
    main()