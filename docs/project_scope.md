# Project Scope & Priorities

## Primary Goal
Build an end-to-end pipeline for detecting and segmenting corrosion defects in pipeline images with severity classification.

## Phase Priorities

### Phase 1: Classification (Days 1-2)
- **Must Have:**
  - Binary classifier (corrosion vs no-corrosion)
  - Baseline metrics (accuracy, F1)
  - Error analysis
  
- **Nice to Have:**
  - Multi-class severity classification
  - Grad-CAM visualization

### Phase 2: Segmentation (Days 2-3)
- **Must Have:**
  - Pixel-level defect localization
  - IoU, Dice metrics
  - Mask visualization
  
- **Nice to Have:**
  - Bounding box extraction
  - Multiple defect detection per image

### Phase 3: Feature Extraction & Severity (Day 3-4)
- **Must Have:**
  - Defect area calculation
  - Severity score/class prediction
  - Feature importance analysis

### Phase 4: Robustness & Demo (Days 4-5)
- **Must Have:**
  - Working demo interface
  - Documentation
  - Presentation materials

## Success Criteria
- [ ] Classification accuracy > 85%
- [ ] Segmentation IoU > 0.6
- [ ] End-to-end pipeline runs in < 5 seconds per image
- [ ] Demo works on fresh environment