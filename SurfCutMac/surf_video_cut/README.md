# Improved Surfing Video Detection System

## 🏄‍♂️ Overview

This system automatically detects and extracts surfing moments from videos using an **advanced activity classification approach** that dramatically reduces false positives compared to velocity-only detection.

## 🚀 Key Improvements Over Baseline

| Metric | Baseline System | **Improved System** | **Improvement** |
|--------|----------------|-------------------|----------------|
| **Precision** | 17.3% | **71.9%** | **🎯 4.16x better** |
| **Recall** | 73.1% | **88.5%** | **📈 21% better** |
| **False Positives** | 91 | **9** | **⬇️ 90% reduction** |
| **False Negatives** | 7 | **3** | **⬇️ 57% reduction** |
| **F1 Score** | 27.9% | **79.3%** | **🏆 2.84x better** |

## 🛠 System Architecture

### Two-Stage Detection Pipeline
1. **Stage 1: YOLO Detection** - Detects person + surfboard pairs (unchanged)
2. **Stage 2: Activity Classification** - **NEW:** Distinguishes active surfing from sitting/waiting

### Key Innovation: Activity Classifier
- **Problem Solved**: Eliminates false positives from surfers sitting on boards
- **Technology**: EfficientNet-B0 based classifier trained on 681 examples
- **Input**: 224x224 crops of person+surfboard regions
- **Output**: Binary classification (active surfing vs sitting/waiting)

## 📁 Project Structure

```
surf_video_cut/
├── process.py                    # Original baseline system
├── process_improved.py           # NEW: Improved system with activity classifier
├── train_activity_classifier.py  # NEW: Train the activity classifier
├── extract_training_data.py      # NEW: Extract training data from videos
├── test_improved_system.py       # NEW: Benchmark improved system
├── models/                       # NEW: Trained models
│   ├── activity_classifier.pth   # Activity classification model
│   └── training_metadata.json    # Training statistics
├── training_data/                # NEW: Training dataset
│   ├── positive/                 # Active surfing examples  
│   └── negative/                 # Sitting/waiting examples
└── create_test_case_2/           # Test videos with golden references
    ├── clips1/...clips5/         # 5 test video sets
    └── *_golden.txt              # Ground truth annotations
```

## 🚀 Quick Start

### 1. Run Improved Detection on Your Video
```bash
python3 process_improved.py \
  --input your_surf_video.mp4 \
  --outdir results \
  --device mps \
  --activity-threshold 0.1
```

### 2. Key Parameters
- `--activity-threshold 0.1` - Lower = more sensitive to surfing (recommended: 0.1-0.15)
- `--device mps` - Use Apple Silicon acceleration (or `cuda` for NVIDIA)
- `--conf 0.25` - YOLO detection confidence
- `--imgsz 640` - Input resolution (higher = better detection, slower)

### 3. Output Files
```
results/
├── your_surf_video_segments.txt  # Detected surf clips (mm:ss-mm:ss format)
├── your_surf_video_ride_mask.tsv # Frame-by-frame detections (debug)
└── debug/                        # Debug frames (if --save-debug enabled)
```

## 🔧 Advanced Usage

### High Recall Mode (Catch More Clips)
```bash
python3 process_improved.py \
  --input video.mp4 \
  --outdir results \
  --activity-threshold 0.05 \
  --conf 0.2 \
  --near-px 100 \
  --smooth \
  --device mps
```

### High Precision Mode (Fewer False Positives)
```bash
python3 process_improved.py \
  --input video.mp4 \
  --outdir results \
  --activity-threshold 0.2 \
  --conf 0.3 \
  --min-seg-sec 1.2 \
  --device mps
```

### Debug Mode (Troubleshoot Detection Issues)
```bash
python3 process_improved.py \
  --input video.mp4 \
  --outdir debug_results \
  --log-every 10 \
  --save-debug \
  --debug-every 30 \
  --test "2:00-4:00" \
  --device mps
```

## 🎯 Performance Optimization

### Device Selection
- **Apple Silicon (M1/M2/M3)**: `--device mps` (recommended)
- **NVIDIA GPU**: `--device cuda`  
- **CPU Only**: `--device cpu` (slowest)

### Speed vs Accuracy Trade-offs
```bash
# Fastest (good for quick tests)
--imgsz 512 --frame-stride 10 --batch-size 64

# Balanced (recommended)  
--imgsz 640 --frame-stride 5 --batch-size 32

# Highest Quality (slow but thorough)
--imgsz 768 --frame-stride 3 --batch-size 16
```

## 🔄 Continuous Improvement

**Yes! The classifier can be continuously improved with new videos:**

1. **Add new training videos** to `new_videos/` directory
2. **Create golden annotations** (*_golden.txt files)
3. **Extract new training data**:
   ```bash
   python3 extract_training_data.py --test-dir new_videos --output-dir new_training
   ```
4. **Retrain the classifier**:
   ```bash
   python3 train_activity_classifier.py --data-dir combined_data --epochs 15
   ```

See [`CONTINUOUS_IMPROVEMENT_GUIDE.md`](CONTINUOUS_IMPROVEMENT_GUIDE.md) for detailed instructions.

## 🚀 Batch Processing with run_all.py

The [`run_all.py`](run_all.py) script has been **updated to use the improved detection system by default**:

```bash
# Process all SD card Track_* folders automatically
python3 run_all.py \
  --sd-root /Volumes/SD \
  --outdir results \
  --ingest-all \
  --activity-threshold 0.1 \
  --device mps

# Process single video with improved detection
python3 run_all.py \
  --input video.mp4 \
  --outdir results \
  --activity-threshold 0.15 \
  --device mps \
  --jobs 8
```

**New Activity Classifier Parameters:**
- `--activity-threshold 0.3` - Classification sensitivity (lower = more surfing detected)
- `--device auto` - Processing device (auto/mps/cuda/cpu)
- `--activity-model models/activity_classifier.pth` - Classifier model path
- `--no-velocity-fallback` - Disable velocity fallback on classifier failure

## 🧪 Testing and Validation

### Benchmark Against Test Data
```bash
# Test current system performance
python3 test_improved_system.py --device mps --thresholds 0.1

# Compare with baseline
python3 test_current_baseline.py --device mps
```

### Diagnose Missed Clips
```bash
# Find out why specific clips were missed
python3 diagnose_missed_clips.py --threshold 0.1

# Debug specific time windows
python3 process_improved.py --test "1:23-1:30" --log-every 1 --save-debug
```

## ⚙️ Configuration Reference

### Core Detection Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--conf` | 0.25 | YOLO detection confidence (lower = more detections) |
| `--imgsz` | 640 | Input resolution (512/640/768) |
| `--frame-stride` | 5 | Sample every N frames |
| `--iou-thresh` | 0.05 | IoU threshold for person+board pairing |
| `--near-px` | 80 | Max distance for person+board pairing (pixels) |

### Activity Classification Parameters  
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--activity-threshold` | 0.3 | Classification threshold (lower = more sensitive) |
| `--activity-model` | models/activity_classifier.pth | Path to trained classifier |
| `--no-velocity-fallback` | False | Disable velocity fallback if classifier fails |

### Temporal Processing Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-seg-sec` | 0.8 | Minimum segment duration |
| `--merge-gap-sec` | 1.0 | Merge segments within N seconds |
| `--preroll` | 1.5 | Seconds before detected start |
| `--postroll` | 1.0 | Seconds after detected end |
| `--smooth` | False | Fill short gaps in ride mask |

## 🐛 Troubleshooting

### Common Issues

**"Activity classifier not found"**
```bash
# Make sure the model exists
ls -la models/activity_classifier.pth

# If missing, retrain:
python3 train_activity_classifier.py --data-dir training_data --device mps
```

**"No segments detected"**
```bash
# Try more sensitive settings:
python3 process_improved.py --activity-threshold 0.05 --conf 0.2 --near-px 120
```

**"Too many false positives"**
```bash
# Try more conservative settings:
python3 process_improved.py --activity-threshold 0.2 --min-seg-sec 1.2
```

**"YOLO not detecting surfboards"**
```bash
# Check debug output:
python3 process_improved.py --log-every 10 --test "1:00-2:00"
# Look for "boards=0" in output - indicates YOLO detection issues
```

### Performance Debugging
```bash
# Check detection quality in specific time window
python3 process_improved.py \
  --input video.mp4 \
  --test "2:30-3:00" \
  --log-every 5 \
  --save-debug \
  --outdir debug_output
```

## 📊 System Validation

The improved system was validated on 5 test videos (26 total golden surf clips):

### Per-Video Performance
| Video | Golden Clips | Detected | Precision | Recall | Status |
|-------|--------------|----------|-----------|--------|---------|
| clips1 | 6 | 6 | 100% | **✅ 100%** | Perfect |
| clips2 | 4 | 7 | 57.1% | **✅ 100%** | Perfect recall |  
| clips3 | 7 | 10 | 50.0% | 71.4% | 2 missed |
| clips4 | 7 | 8 | 87.5% | **✅ 100%** | Perfect recall |
| clips5 | 2 | 1 | 100% | 50.0% | 1 missed |

**Overall: 88.5% recall with 90% reduction in false positives vs baseline**

## 🏆 Success Metrics Achieved

✅ **Primary Goal**: Reduce false positives from surfers sitting on boards  
✅ **90% False Positive Reduction**: From 91 → 9 false positives  
✅ **Improved Recall**: 73.1% → 88.5% (+21%)  
✅ **Temporal Precision**: Most clips within ±2s of golden segments  
✅ **Real-time Performance**: ~6-8 FPS on Apple Silicon  
✅ **Easy Integration**: Drop-in replacement for original system  

## 📚 Additional Resources

- [`surfing_detection_improvement_plan.md`](surfing_detection_improvement_plan.md) - Technical architecture details
- [`CONTINUOUS_IMPROVEMENT_GUIDE.md`](CONTINUOUS_IMPROVEMENT_GUIDE.md) - How to retrain and improve the system
- [`models/training_metadata.json`](models/training_metadata.json) - Current model performance statistics

## 💡 Future Enhancements

- **Higher Recall**: Train on more diverse surfing examples  
- **Temporal Modeling**: Use 3-5 frame sequences for better context
- **Real-time Processing**: Optimize for live video streams
- **Multi-Surfer Support**: Detect and track multiple surfers simultaneously
- **Wave Quality Assessment**: Integrate wave quality scoring

---

**🏄‍♂️ Happy surfing and automatic clip detection!**