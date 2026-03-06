#!/usr/bin/env python3
"""
Test the optimized surfing detection system with aggressive parameters for 100% recall.
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
    
    matched_detected = set()
    matched_golden = set()
    total_overlap = 0.0
    temporal_extensions = []
    
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
            
            det_start, det_end = det_seg
            gold_start, gold_end = golden[best_golden_idx]
            extension = (det_end - det_start) - (gold_end - gold_start)
            temporal_extensions.append(extension)
    
    precision = len(matched_detected) / len(detected) if detected else 0.0
    recall = len(matched_golden) / len(golden) if golden else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_overlap = total_overlap / len(matched_detected) if matched_detected else 0.0
    
    false_positives = len(detected) - len(matched_detected)
    false_negatives = len(golden) - len(matched_golden)
    
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


def run_optimized_detection(video_path: Path, output_dir: Path, 
                          activity_threshold: float = 0.1, device: str = 'mps') -> Path:
    """Run the optimized detection system with aggressive parameters for high recall"""
    print(f"Running optimized detection on {video_path.name} (threshold={activity_threshold})...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Optimized parameters for maximum recall
    cmd = [
        'python3', 'process_improved.py',
        '--input', str(video_path),
        '--outdir', str(output_dir),
        '--device', device,
        '--conf', '0.15',           # Lower confidence for more detections
        '--imgsz', '768',           # Higher resolution for better detection
        '--frame-stride', '3',      # Higher temporal resolution  
        '--batch-size', '16',       # Smaller batch for better processing
        '--near-px', '120',         # Increased proximity threshold
        '--iou-thresh', '0.01',     # Lower IoU threshold for pairing
        '--activity-threshold', str(activity_threshold),
        '--min-seg-sec', '0.5',     # Allow shorter segments
        '--merge-gap-sec', '2.0',   # More aggressive merging
        '--preroll', '2.0',         # More preroll for safety
        '--postroll', '1.5',        # More postroll for safety
        '--smooth',                 # Enable smoothing
        '--smooth-gap', '1.5'       # Fill small gaps
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running optimized detection: {result.stderr}")
        raise RuntimeError(f"Optimized detection failed for {video_path}")
    
    stem = video_path.stem
    segments_file = output_dir / f"{stem}_segments.txt"
    
    if not segments_file.exists():
        candidates = list(output_dir.glob("*_segments.txt"))
        if candidates:
            segments_file = candidates[0]
        else:
            return output_dir / "empty_segments.txt"
    
    return segments_file


def main():
    parser = argparse.ArgumentParser(description='Test optimized surfing detection system')
    parser.add_argument('--test-dir', default='create_test_case_2',
                       help='Directory containing test videos')
    parser.add_argument('--output-dir', default='optimized_results',
                       help='Directory for detection outputs')
    parser.add_argument('--device', default='mps', choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device for detection')
    parser.add_argument('--thresholds', nargs='+', type=float, 
                       default=[0.05, 0.1, 0.15],
                       help='Activity classification thresholds to test')
    parser.add_argument('--baseline-results', default='baseline_results/baseline_results.json',
                       help='Path to baseline results for comparison')
    
    args = parser.parse_args()
    
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_dirs = [d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith('clips')]
    video_dirs.sort()
    
    print(f"Found {len(video_dirs)} test video directories")
    print(f"Testing optimized system with activity thresholds: {args.thresholds}")
    
    threshold_results = {}
    
    for threshold in args.thresholds:
        print(f"\n{'='*60}")
        print(f"Testing optimized system with activity threshold: {threshold}")
        print('='*60)
        
        threshold_results_individual = {}
        total_metrics = {
            'total_detected': 0,
            'total_golden': 0,
            'total_matched': 0,
            'total_false_positives': 0,
            'total_false_negatives': 0,
            'all_extensions': []
        }
        
        for video_dir in video_dirs:
            video_files = list(video_dir.glob("*.mp4"))
            golden_files = list(video_dir.glob("*_golden.txt"))
            
            if not video_files or not golden_files:
                continue
            
            video_path = video_files[0]
            golden_path = golden_files[0]
            
            threshold_output_dir = output_dir / f"threshold_{threshold}" / video_dir.name
            
            try:
                segments_file = run_optimized_detection(video_path, threshold_output_dir, threshold, args.device)
                
                detected_segments = parse_segments_file(segments_file)
                golden_segments = parse_segments_file(golden_path)
                
                metrics = calculate_metrics(detected_segments, golden_segments, 0.3)
                
                threshold_results_individual[video_dir.name] = {
                    'detected_segments': detected_segments,
                    'golden_segments': golden_segments,
                    'metrics': metrics
                }
                
                total_metrics['total_detected'] += metrics['detected_count']
                total_metrics['total_golden'] += metrics['golden_count']
                total_metrics['total_matched'] += metrics['matched_count']
                total_metrics['total_false_positives'] += metrics['false_positives']
                total_metrics['total_false_negatives'] += metrics['false_negatives']
                total_metrics['all_extensions'].extend(metrics['temporal_extensions'])
                
                recall_status = "✓ PERFECT" if metrics['recall'] == 1.0 else f"✗ {metrics['recall']:.1%}"
                print(f"  {video_dir.name}: Precision={metrics['precision']:.3f}, Recall={recall_status}, FP={metrics['false_positives']}, FN={metrics['false_negatives']}")
                
            except Exception as e:
                print(f"Error processing {video_dir.name}: {e}")
                continue
        
        # Calculate overall metrics
        overall_precision = total_metrics['total_matched'] / total_metrics['total_detected'] if total_metrics['total_detected'] > 0 else 0.0
        overall_recall = total_metrics['total_matched'] / total_metrics['total_golden'] if total_metrics['total_golden'] > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        extensions = total_metrics['all_extensions']
        avg_extension = sum(extensions) / len(extensions) if extensions else 0.0
        max_extension = max(extensions) if extensions else 0.0
        violations_5s = sum(1 for ext in extensions if ext > 5.0)
        
        threshold_results[threshold] = {
            'individual_results': threshold_results_individual,
            'overall_metrics': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1_score': overall_f1,
                'total_detected': total_metrics['total_detected'],
                'total_golden': total_metrics['total_golden'],
                'total_matched': total_metrics['total_matched'],
                'total_false_positives': total_metrics['total_false_positives'],
                'total_false_negatives': total_metrics['total_false_negatives'],
                'avg_temporal_extension': avg_extension,
                'max_temporal_extension': max_extension,
                'temporal_violations_5s': violations_5s
            }
        }
        
        recall_status = "🎯 100% RECALL!" if overall_recall >= 1.0 else f"❌ {overall_recall:.1%} recall"
        print(f"\nOverall Results for threshold {threshold}: {recall_status}")
        print(f"  Precision: {overall_precision:.3f}")
        print(f"  F1 Score: {overall_f1:.3f}")
        print(f"  False Positives: {total_metrics['total_false_positives']}")
        print(f"  False Negatives: {total_metrics['total_false_negatives']}")
        print(f"  Temporal Violations >5s: {violations_5s}")
    
    # Load baseline results for comparison
    baseline_metrics = None
    if Path(args.baseline_results).exists():
        with open(args.baseline_results, 'r') as f:
            baseline_data = json.load(f)
            baseline_metrics = baseline_data['overall_metrics']
    
    # Find best configuration
    best_threshold = None
    best_score = 0.0
    
    print(f"\n{'='*80}")
    print("OPTIMIZED SYSTEM COMPARISON SUMMARY")
    print('='*80)
    print(f"{'Threshold':<12} {'Precision':<10} {'Recall':<8} {'F1':<8} {'FP':<6} {'FN':<6} {'Violations>5s':<12}")
    print('-' * 80)
    
    for threshold in sorted(args.thresholds):
        metrics = threshold_results[threshold]['overall_metrics']
        
        # Prioritize 100% recall, then precision
        if metrics['recall'] >= 1.0:
            score = 1000 + metrics['precision']  # 100% recall gets high score
        else:
            score = metrics['recall'] * 500  # Penalize missing clips heavily
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
        
        recall_display = f"{metrics['recall']:.3f}"
        if metrics['recall'] >= 1.0:
            recall_display = "1.000★"
        
        print(f"{threshold:<12.2f} {metrics['precision']:<10.3f} {recall_display:<8} {metrics['f1_score']:<8.3f} "
              f"{metrics['total_false_positives']:<6} {metrics['total_false_negatives']:<6} {metrics['temporal_violations_5s']:<12}")
    
    print('-' * 80)
    
    if baseline_metrics:
        print(f"{'BASELINE':<12} {baseline_metrics['precision']:<10.3f} {baseline_metrics['recall']:<8.3f} {baseline_metrics['f1_score']:<8.3f} "
              f"{baseline_metrics['total_false_positives']:<6} {baseline_metrics['total_false_negatives']:<6} {baseline_metrics['temporal_violations_5s']:<12}")
    
    if best_threshold is not None:
        best_metrics = threshold_results[best_threshold]['overall_metrics']
        
        print(f"\n🏆 BEST CONFIGURATION: Activity threshold {best_threshold}")
        print(f"   Precision: {best_metrics['precision']:.3f}")
        print(f"   Recall: {best_metrics['recall']:.3f}")
        print(f"   F1 Score: {best_metrics['f1_score']:.3f}")
        print(f"   False Positives: {best_metrics['total_false_positives']}")
        print(f"   False Negatives: {best_metrics['total_false_negatives']}")
        print(f"   Temporal Violations >5s: {best_metrics['temporal_violations_5s']}")
        
        if baseline_metrics:
            fp_reduction = ((baseline_metrics['total_false_positives'] - best_metrics['total_false_positives']) 
                          / baseline_metrics['total_false_positives'] * 100)
            print(f"\n📈 IMPROVEMENT vs BASELINE:")
            print(f"   Precision: {best_metrics['precision']/baseline_metrics['precision']:.1f}x better")
            print(f"   Recall: {best_metrics['recall']/baseline_metrics['recall']:.1f}x better")
            print(f"   False Positives: {fp_reduction:.0f}% reduction")
            print(f"   False Negatives: {baseline_metrics['total_false_negatives'] - best_metrics['total_false_negatives']} fewer")
        
        if best_metrics['recall'] >= 1.0:
            print(f"\n✅ SUCCESS: 100% recall achieved!")
            print(f"   All {best_metrics['total_golden']} golden surf clips detected")
        else:
            print(f"\n⚠️  Still missing {best_metrics['total_false_negatives']} golden clips")
    
    # Save results
    results_file = output_dir / 'optimized_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'optimized_system_results': threshold_results,
            'best_threshold': best_threshold,
            'best_score': best_score,
            'baseline_comparison': baseline_metrics
        }, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == '__main__':
    main()