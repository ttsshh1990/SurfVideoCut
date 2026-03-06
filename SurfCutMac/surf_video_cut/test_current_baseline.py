#!/usr/bin/env python3
"""
Script to run current detection system on all test videos and analyze performance
against golden reference segments.
"""

import os
import subprocess
import json
from pathlib import Path
from typing import List, Tuple, Dict
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
    """Parse segments file into list of (start, end) tuples in seconds"""
    segments = []
    if not file_path.exists():
        return segments
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Handle format: start-end or start-end\tindex
            parts = line.split('\t')
            time_range = parts[0]
            
            if '-' not in time_range:
                continue
                
            start_str, end_str = time_range.split('-', 1)
            start_sec = parse_timecode(start_str.strip())
            end_sec = parse_timecode(end_str.strip())
            segments.append((start_sec, end_sec))
    
    return segments

def calculate_overlap(seg1: Tuple[float, float], seg2: Tuple[float, float]) -> float:
    """Calculate temporal overlap between two segments"""
    start1, end1 = seg1
    start2, end2 = seg2
    
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_end <= overlap_start:
        return 0.0
    
    overlap_duration = overlap_end - overlap_start
    union_duration = max(end1, end2) - min(start1, start2)
    
    return overlap_duration / union_duration if union_duration > 0 else 0.0

def calculate_metrics(detected: List[Tuple[float, float]], 
                     golden: List[Tuple[float, float]], 
                     overlap_threshold: float = 0.3) -> Dict:
    """Calculate precision, recall, and other metrics"""
    
    # Find matches based on overlap threshold
    matched_detected = set()
    matched_golden = set()
    total_overlap = 0.0
    temporal_extensions = []  # How much detected segments extend beyond golden
    
    for i, det_seg in enumerate(detected):
        best_overlap = 0.0
        best_golden_idx = -1
        
        for j, gold_seg in enumerate(golden):
            if j in matched_golden:
                continue
                
            overlap = calculate_overlap(det_seg, gold_seg)
            if overlap > best_overlap and overlap >= overlap_threshold:
                best_overlap = overlap
                best_golden_idx = j
        
        if best_golden_idx >= 0:
            matched_detected.add(i)
            matched_golden.add(best_golden_idx)
            total_overlap += best_overlap
            
            # Calculate temporal extension
            det_start, det_end = det_seg
            gold_start, gold_end = golden[best_golden_idx]
            extension = (det_end - det_start) - (gold_end - gold_start)
            temporal_extensions.append(extension)
    
    # Calculate metrics
    precision = len(matched_detected) / len(detected) if detected else 0.0
    recall = len(matched_golden) / len(golden) if golden else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_overlap = total_overlap / len(matched_detected) if matched_detected else 0.0
    
    # False positives and negatives
    false_positives = len(detected) - len(matched_detected)
    false_negatives = len(golden) - len(matched_golden)
    
    # Temporal analysis
    avg_extension = sum(temporal_extensions) / len(temporal_extensions) if temporal_extensions else 0.0
    max_extension = max(temporal_extensions) if temporal_extensions else 0.0
    violations_5s = sum(1 for ext in temporal_extensions if ext > 5.0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_overlap': avg_overlap,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'detected_count': len(detected),
        'golden_count': len(golden),
        'matched_count': len(matched_detected),
        'avg_temporal_extension': avg_extension,
        'max_temporal_extension': max_extension,
        'temporal_violations_5s': violations_5s,
        'temporal_extensions': temporal_extensions
    }

def run_detection(video_path: Path, output_dir: Path, device: str = 'auto') -> Path:
    """Run the current detection system on a video"""
    print(f"Running detection on {video_path.name}...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run process.py
    cmd = [
        'python3', 'process.py',
        '--input', str(video_path),
        '--outdir', str(output_dir),
        '--device', device,
        '--imgsz', '640',
        '--frame-stride', '5',
        '--batch-size', '32',
        '--vx-thresh', '60.0',
        '--conf', '0.25'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running detection: {result.stderr}")
        raise RuntimeError(f"Detection failed for {video_path}")
    
    # Find the generated segments file
    stem = video_path.stem
    segments_file = output_dir / f"{stem}_segments.txt"
    
    if not segments_file.exists():
        print(f"Warning: Segments file not found: {segments_file}")
        # Try to find any segments file in output dir
        candidates = list(output_dir.glob("*_segments.txt"))
        if candidates:
            segments_file = candidates[0]
            print(f"Using: {segments_file}")
        else:
            print("No segments file found!")
            return output_dir / "empty_segments.txt"
    
    return segments_file

def main():
    parser = argparse.ArgumentParser(description='Test current baseline detection system')
    parser.add_argument('--test-dir', default='create_test_case_2', 
                       help='Directory containing test videos')
    parser.add_argument('--output-dir', default='baseline_results',
                       help='Directory for detection outputs')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device for detection')
    parser.add_argument('--overlap-threshold', type=float, default=0.3,
                       help='Minimum overlap to consider a match')
    
    args = parser.parse_args()
    
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Find all test video directories
    video_dirs = [d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith('clips')]
    video_dirs.sort()
    
    print(f"Found {len(video_dirs)} test video directories")
    
    all_results = {}
    overall_metrics = {
        'total_detected': 0,
        'total_golden': 0,
        'total_matched': 0,
        'total_false_positives': 0,
        'total_false_negatives': 0,
        'all_extensions': []
    }
    
    for video_dir in video_dirs:
        print(f"\n{'='*60}")
        print(f"Processing {video_dir.name}")
        print('='*60)
        
        # Find video file and golden reference
        video_files = list(video_dir.glob("*.mp4"))
        golden_files = list(video_dir.glob("*_golden.txt"))
        
        if not video_files:
            print(f"No video file found in {video_dir}")
            continue
            
        if not golden_files:
            print(f"No golden reference found in {video_dir}")
            continue
        
        video_path = video_files[0]
        golden_path = golden_files[0]
        
        # Create output directory for this video
        video_output_dir = output_dir / video_dir.name
        
        try:
            # Run detection
            segments_file = run_detection(video_path, video_output_dir, args.device)
            
            # Load results
            detected_segments = parse_segments_file(segments_file)
            golden_segments = parse_segments_file(golden_path)
            
            print(f"Detected segments: {len(detected_segments)}")
            print(f"Golden segments: {len(golden_segments)}")
            
            # Calculate metrics
            metrics = calculate_metrics(detected_segments, golden_segments, args.overlap_threshold)
            
            # Store results
            all_results[video_dir.name] = {
                'video_path': str(video_path),
                'golden_path': str(golden_path),
                'segments_file': str(segments_file),
                'detected_segments': detected_segments,
                'golden_segments': golden_segments,
                'metrics': metrics
            }
            
            # Update overall metrics
            overall_metrics['total_detected'] += metrics['detected_count']
            overall_metrics['total_golden'] += metrics['golden_count']
            overall_metrics['total_matched'] += metrics['matched_count']
            overall_metrics['total_false_positives'] += metrics['false_positives']
            overall_metrics['total_false_negatives'] += metrics['false_negatives']
            overall_metrics['all_extensions'].extend(metrics['temporal_extensions'])
            
            # Print metrics for this video
            print(f"\nMetrics for {video_dir.name}:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1 Score: {metrics['f1_score']:.3f}")
            print(f"  False Positives: {metrics['false_positives']}")
            print(f"  False Negatives: {metrics['false_negatives']}")
            print(f"  Avg Temporal Extension: {metrics['avg_temporal_extension']:.2f}s")
            print(f"  Max Temporal Extension: {metrics['max_temporal_extension']:.2f}s")
            print(f"  Violations >5s: {metrics['temporal_violations_5s']}")
            
        except Exception as e:
            print(f"Error processing {video_dir.name}: {e}")
            continue
    
    # Calculate overall metrics
    if overall_metrics['total_detected'] > 0:
        overall_precision = overall_metrics['total_matched'] / overall_metrics['total_detected']
    else:
        overall_precision = 0.0
        
    if overall_metrics['total_golden'] > 0:
        overall_recall = overall_metrics['total_matched'] / overall_metrics['total_golden']
    else:
        overall_recall = 0.0
        
    if overall_precision + overall_recall > 0:
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
    else:
        overall_f1 = 0.0
    
    extensions = overall_metrics['all_extensions']
    avg_extension = sum(extensions) / len(extensions) if extensions else 0.0
    max_extension = max(extensions) if extensions else 0.0
    violations_5s = sum(1 for ext in extensions if ext > 5.0)
    
    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print('='*60)
    print(f"Total Golden Segments: {overall_metrics['total_golden']}")
    print(f"Total Detected Segments: {overall_metrics['total_detected']}")
    print(f"Total Matched Segments: {overall_metrics['total_matched']}")
    print(f"Total False Positives: {overall_metrics['total_false_positives']}")
    print(f"Total False Negatives: {overall_metrics['total_false_negatives']}")
    print(f"\nOverall Precision: {overall_precision:.3f}")
    print(f"Overall Recall: {overall_recall:.3f}")
    print(f"Overall F1 Score: {overall_f1:.3f}")
    print(f"\nTemporal Analysis:")
    print(f"  Average Extension: {avg_extension:.2f}s")
    print(f"  Maximum Extension: {max_extension:.2f}s") 
    print(f"  Violations >5s: {violations_5s}")
    
    # Save detailed results
    results_file = output_dir / 'baseline_results.json'
    with open(results_file, 'w') as f:
        # Convert tuples to lists for JSON serialization
        json_results = {}
        for key, value in all_results.items():
            json_value = value.copy()
            json_value['detected_segments'] = [list(seg) for seg in value['detected_segments']]
            json_value['golden_segments'] = [list(seg) for seg in value['golden_segments']]
            json_results[key] = json_value
        
        json.dump({
            'individual_results': json_results,
            'overall_metrics': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1_score': overall_f1,
                'total_golden': overall_metrics['total_golden'],
                'total_detected': overall_metrics['total_detected'],
                'total_matched': overall_metrics['total_matched'],
                'total_false_positives': overall_metrics['total_false_positives'],
                'total_false_negatives': overall_metrics['total_false_negatives'],
                'avg_temporal_extension': avg_extension,
                'max_temporal_extension': max_extension,
                'temporal_violations_5s': violations_5s
            }
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")

if __name__ == '__main__':
    main()