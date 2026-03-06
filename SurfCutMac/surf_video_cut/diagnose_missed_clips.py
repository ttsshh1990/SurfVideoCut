#!/usr/bin/env python3
"""
Diagnose why specific golden clips are being missed by the improved detection system.
"""

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


def find_missed_clips(detected: List[Tuple[float, float]], 
                     golden: List[Tuple[float, float]], 
                     overlap_threshold: float = 0.3) -> List[Tuple[int, Tuple[float, float]]]:
    """Find golden clips that were not detected"""
    matched_golden = set()
    
    for i, det_seg in enumerate(detected):
        for j, gold_seg in enumerate(golden):
            if j in matched_golden:
                continue
            overlap = calculate_overlap(det_seg, gold_seg)
            if overlap >= overlap_threshold:
                matched_golden.add(j)
                break
    
    # Return unmatched golden clips with their indices
    missed = []
    for j, gold_seg in enumerate(golden):
        if j not in matched_golden:
            missed.append((j, gold_seg))
    
    return missed


def sec_to_tc(sec: float) -> str:
    """Format seconds to mm:ss"""
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m}:{s:05.2f}"


def main():
    parser = argparse.ArgumentParser(description='Diagnose missed golden clips')
    parser.add_argument('--results-file', default='improved_results/improved_results.json',
                       help='Path to improved results JSON')
    parser.add_argument('--test-dir', default='create_test_case_2',
                       help='Directory containing test videos')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Activity threshold to analyze')
    
    args = parser.parse_args()
    
    # Load results
    results_file = Path(args.results_file)
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    threshold_key = str(args.threshold)
    if threshold_key not in results_data['threshold_results']:
        available = list(results_data['threshold_results'].keys())
        raise ValueError(f"Threshold {args.threshold} not found. Available: {available}")
    
    threshold_results = results_data['threshold_results'][threshold_key]['individual_results']
    
    print(f"Analyzing missed clips for threshold {args.threshold}")
    print(f"{'='*60}")
    
    total_missed = 0
    
    for video_name, video_results in threshold_results.items():
        detected_segments = [tuple(seg) for seg in video_results['detected_segments']]
        golden_segments = [tuple(seg) for seg in video_results['golden_segments']]
        
        missed_clips = find_missed_clips(detected_segments, golden_segments)
        
        if missed_clips:
            print(f"\n{video_name}: {len(missed_clips)} missed clips")
            print(f"  Total golden clips: {len(golden_segments)}")
            print(f"  Detected clips: {len(detected_segments)}")
            
            for clip_idx, (start, end) in missed_clips:
                duration = end - start
                print(f"    Missed #{clip_idx+1}: {sec_to_tc(start)}-{sec_to_tc(end)} (duration: {duration:.2f}s)")
            
            # Show detected clips for comparison
            print(f"  Detected clips:")
            for i, (start, end) in enumerate(detected_segments):
                duration = end - start
                print(f"    Detected #{i+1}: {sec_to_tc(start)}-{sec_to_tc(end)} (duration: {duration:.2f}s)")
            
            total_missed += len(missed_clips)
        else:
            print(f"\n{video_name}: Perfect recall! All {len(golden_segments)} clips detected")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total missed clips: {total_missed}")
    
    if total_missed > 0:
        print(f"\nNext steps to investigate:")
        print(f"1. Run improved detection with --log-every 1 on time windows around missed clips")
        print(f"2. Check if YOLO detects person+surfboard pairs during those times")
        print(f"3. Examine activity classifier predictions for those time periods")
        print(f"4. Consider adjusting detection parameters (IoU, proximity, confidence)")
        
        # Generate test commands for missed clips
        print(f"\nDiagnostic commands to run:")
        for video_name, video_results in threshold_results.items():
            detected_segments = [tuple(seg) for seg in video_results['detected_segments']]
            golden_segments = [tuple(seg) for seg in video_results['golden_segments']]
            
            missed_clips = find_missed_clips(detected_segments, golden_segments)
            
            if missed_clips:
                video_dir = Path(args.test_dir) / video_name
                video_files = list(video_dir.glob("*.mp4"))
                if video_files:
                    video_path = video_files[0]
                    
                    for clip_idx, (start, end) in missed_clips:
                        # Expand window by ±30 seconds for context
                        window_start = max(0, start - 30)
                        window_end = end + 30
                        
                        print(f"\n# Debug missed clip #{clip_idx+1} in {video_name}")
                        print(f"python3 process_improved.py \\")
                        print(f"  --input {video_path} \\")
                        print(f"  --outdir debug_missed_{video_name}_clip{clip_idx+1} \\") 
                        print(f"  --device mps \\")
                        print(f"  --activity-threshold {args.threshold} \\")
                        print(f"  --log-every 5 \\")
                        print(f"  --save-debug \\")
                        print(f"  --test \"{sec_to_tc(window_start)}-{sec_to_tc(window_end)}\"")


if __name__ == '__main__':
    main()