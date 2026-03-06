# Surfing Detection Improvement Plan

## Current System Analysis

### System Overview
The current system uses a generic YOLOv8 model (`yolov8n.pt`) to detect:
1. **Person** objects 
2. **Surfboard** objects
3. **Pairing Logic**: Uses IoU and proximity to pair person+surfboard
4. **Activity Classification**: Uses horizontal velocity (`vx_thresh=60px/s`) to determine if actively surfing

### Key Parameters in Current System
- **Detection**: `conf=0.25`, `imgsz=640`, `frame_stride=5`
- **Pairing**: `iou_thresh=0.05`, `near_px=80`  
- **Activity**: `vx_thresh=60.0` (horizontal speed threshold)
- **Temporal**: `min_seg_sec=0.8`, `merge_gap_sec=1.0`

### Primary Problem Identified
**False Positives**: Surfers sitting on surfboards (waiting for waves, resting) are incorrectly classified as "riding" when they have sufficient horizontal movement due to:
- Wave motion moving surfer+board laterally
- Paddling movements creating horizontal displacement
- Camera movement or zoom changes

## Test Data Structure

### Available Test Videos (5 sets)
1. `clips1/Track_2025_08_10_090259_test_part01.mp4` - 6 golden segments
2. `clips2/Track_2025_08_10_090259_test_part02.mp4` - 4 golden segments  
3. `clips3/Track_2025_08_10_090259_test_part03.mp4` - 7 golden segments
4. `clips4/Track_2025_08_10_090259_test_part04.mp4` - 7 golden segments
5. `clips5/Track_2025_08_10_090259_test_part05.mp4` - 2 golden segments

### Golden Reference Format
Time segments in `mm:ss.ss-mm:ss.ss` format representing true surfing moments.

## Requirements & Constraints

### Performance Requirements
1. **100% Recall**: Must detect ALL golden reference surf clips
2. **Temporal Precision**: Results can extend ±5 seconds beyond golden segments (max)
3. **False Positive Reduction**: Focus on eliminating "sitting on surfboard" detections

### Technical Constraints  
- Must integrate with existing pipeline ([`process.py`](process.py:1))
- Apple Silicon (MPS) optimization preferred
- Maintain batch processing capabilities

## Proposed Solution Approaches

### Option 1: Enhanced Activity Classification
**Approach**: Improve the velocity-based classification with additional features:
- **Posture Detection**: Add pose estimation to distinguish sitting vs standing/crouching
- **Wave Context**: Analyze wave patterns and surfer position relative to wave face
- **Temporal Smoothing**: Require sustained riding motion over multiple frames

**Pros**: 
- Minimal changes to existing pipeline
- Leverages current detection accuracy
- Fast inference

**Cons**: 
- Still relies on generic YOLO model
- Complex rule engineering required

### Option 2: Custom YOLO Model Training
**Approach**: Train specialized model for surfing activity classification:
- **Multi-class Detection**: person, surfboard, person-riding, person-sitting
- **Activity-aware Training**: Annotations distinguish between sitting and riding states
- **Temporal Features**: Incorporate frame sequences for better context

**Pros**: 
- End-to-end learned solution
- Better generalization potential
- Reduced false positives at source

**Cons**: 
- Requires extensive annotation effort
- Longer development timeline
- Need quality training data

### Option 3: Two-Stage Architecture  
**Approach**: Current detection + specialized activity classifier:
- **Stage 1**: Use current YOLO for person/surfboard detection
- **Stage 2**: CNN/Vision Transformer to classify detected pairs as riding/sitting
- **Integration**: Replace velocity-only logic with learned classifier

**Pros**:
- Modular design - can improve stages independently  
- Leverage existing detection quality
- Focused training on activity classification only

**Cons**:
- Increased inference complexity
- Two models to maintain

## Recommended Approach: Option 3 (Two-Stage)

### Rationale
1. **Pragmatic**: Leverages existing detection pipeline
2. **Focused**: Addresses core problem (sitting vs riding) specifically  
3. **Manageable**: Smaller dataset requirements for activity classification
4. **Maintainable**: Clear separation of concerns

### Implementation Plan

#### Phase 1: Baseline Analysis & Data Preparation
1. **Run Current System** on all 5 test videos 
2. **Compare Results** with golden reference segments
3. **Analyze False Positives** - identify sitting vs riding patterns
4. **Extract Training Crops** from person+surfboard detection boxes
5. **Label Activity Classes** (riding=1, sitting/waiting=0)

#### Phase 2: Activity Classifier Development
1. **Architecture Selection**: 
   - ResNet/EfficientNet for single-frame classification
   - 3D-CNN or LSTM for temporal sequence modeling
   - Vision Transformer for attention-based classification
   
2. **Training Strategy**:
   - Use crops from detection boxes as input
   - Binary classification: active_surfing vs not_active_surfing
   - Data augmentation for robustness
   - Temporal consistency training

3. **Integration**:
   - Replace velocity check in [`process_batch()`](process.py:339) 
   - Add activity classifier inference
   - Maintain MPS optimization

#### Phase 3: Validation & Optimization
1. **Performance Metrics**:
   - Recall on golden segments (target: 100%)
   - Precision improvement (false positive reduction)
   - Temporal accuracy (±5s constraint)
   
2. **Hyperparameter Tuning**:
   - Classification threshold optimization
   - Temporal smoothing parameters  
   - Detection confidence adjustments

### Data Requirements

#### Training Data Needed
- **Positive Examples**: ~500-1000 crops of active surfing moments
- **Negative Examples**: ~500-1000 crops of sitting/waiting/paddling
- **Temporal Sequences**: 3-5 frame sequences for motion context
- **Variety**: Different lighting, wave conditions, surfer positions

#### Annotation Strategy
1. Extract detection crops from current system on training videos
2. Manual labeling of activity state (riding vs not-riding)  
3. Temporal consistency checks across frame sequences
4. Quality validation with multiple annotators

## Technical Architecture

### Model Integration Point
Current detection flow:
```python
# In process_batch() around line 376
flag = 0
paired = best_pair is not None and (conditions)
if paired:
    # Current: velocity-only check
    v_ok = vx >= vx_thresh  # REPLACE THIS
    flag = 1 if v_ok else 0
```

Proposed integration:
```python
# Replace velocity check with activity classifier
if paired:
    crop = extract_crop(frame, best_pair)  # Extract person+board region
    activity_score = activity_classifier.predict(crop)  # New model
    flag = 1 if activity_score > threshold else 0
```

### Model Architecture Options

#### Option A: Single Frame Classifier
- **Input**: 224x224 RGB crop of person+surfboard region
- **Architecture**: EfficientNet-B0 or ResNet18
- **Output**: Binary classification (riding/not_riding)
- **Pros**: Simple, fast inference
- **Cons**: No temporal context

#### Option B: Temporal Sequence Classifier  
- **Input**: 5-frame sequence of 224x224 crops
- **Architecture**: 3D-CNN or ResNet + LSTM
- **Output**: Binary classification with temporal smoothing
- **Pros**: Better accuracy, temporal consistency
- **Cons**: More complex, higher memory usage

## Success Metrics

### Primary Objectives
1. **Recall**: 100% of golden segments detected (no false negatives)
2. **Precision**: Reduce false positives by 70%+ (mainly sitting surfers)
3. **Temporal Accuracy**: No detected segments >5s longer than golden reference

### Validation Protocol
1. **Cross-validation** on 5 test video sets
2. **Temporal overlap analysis** between detected and golden segments  
3. **False positive categorization** and root cause analysis
4. **Performance benchmarking** vs current system

## Next Steps

1. **Switch to Code Mode** to implement baseline testing
2. **Run Current Detection** on all test videos
3. **Analyze Results** and quantify current performance gaps
4. **Design Activity Classifier** based on observed patterns
5. **Implement Training Pipeline** for activity classification model