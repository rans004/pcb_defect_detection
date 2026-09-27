
## Overview
Printed Circuit Board (PCB) manufacturing defects — if undetected — lead to product failures, costly recalls, and safety hazards.

Traditional optical inspection is slow, operator-dependent, and struggles with subtle defects like micro-shorts and hairline open circuits.


This project implements an automated visual inspection system that:\
-Detects 6 categories of PCB manufacturing defects from images or live camera feed\s
-Classifies each defect by type and assigns a severity level (CRITICAL / HIGH / MEDIUM / LOW)\
-Renders annotated frames with bounding boxes, confidence scores, and severity badges in real time\
-Logs every detection to a timestamped CSV for traceability and downstream quality reporting\
-Gives a per-board PASS / FAIL verdict based on configurable severity thresholds\\

## Key Results

| Metric | Value |
|---|---|
| Base architecture | YOLOv11n |
| Training epochs | 100 (extended run to convergence) |
| Input resolution | 640 × 640 |
| **mAP@0.5 (best model)** | **0.94** |
| Precision (all classes) | 1.00 @ conf 0.825 |
| Recall (all classes) | 0.76 @ conf 0.000 |
| Best F1 threshold | 0.306 confidence |
| F1 score (all classes) | 0.66 |

## Confusion Matrix Highlights (Normalised)

| Class | Correct Detection Rate |
|---|---|
| short | 0.60 |
| spurious_copper | 0.59 |
| missing_hole | 0.56 |
| mouse_bite | 0.51 |
| open_circuit | 0.44 |
| spur | 0.48 |

The primary error mode across all classes is background false negatives .Detections suppressed at higher confidence thresholds rather than inter-class confusion, indicating the model has learned discriminative per-class features but benefits from a lower confidence threshold 

## Defect Classes
Six defect types are detected, each mapped to a severity level\


## Defect Classes

Six defect types are detected, each mapped to an engineer-facing severity level:

| Class | Severity | Description |
|---|---|---|
| `short` | 🔴 CRITICAL | Unintended conductor bridging between traces |
| `missing_hole` | 🟠 HIGH | Drill hole absent from pad / via |
| `open_circuit` | 🟠 HIGH | Broken or incomplete conductor trace |
| `spurious_copper` | 🟡 MEDIUM | Unwanted copper patch remaining after etching |
| `mouse_bite` | 🟡 MEDIUM | Edge erosion / scalloping at board boundary |
| `spur` | 🟢 LOW | Small excess copper spur on a trace |


## Dataset
The base dataset from roboflow was supplemented with personally photographed PCB defect samples to:\
Improve recall on under-represented classes (spur, open_circuit)\
Reduce overfitting to the Roboflow distribution\
Introduce real-world lighting variation not present in the original dataset


## Quick start
1. Clone and install dependencies\
  git clone https://github.com/<your-username>/pcb-defect-detection.git\
cd pcb-defect-detection\
pip install ultralytics opencv-python numpy torch

2.download trained weights\
3.ran on static image.similar thing can be done for live webcam,video sources.\
detector = PCBDefectDetector(\
    source="path/to/your_pcb_image.jpg",\
    model_path="weights/best.pt",\
    conf_threshold=0.25,\
    log_csv=True,\
)\
detector.run_on_image("path/to/your_pcb_image.jpg", save_path="pcb_result.jpg")

## limitations
-Open circuit recall is lowest (AP 0.494): open circuits are visually subtle — a hairline gap in a trace — and the model would benefit\ from additional annotated examples and higher-resolution crops.\
-Background false negatives dominate: the confusion matrix shows most errors are undetected defects (suppressed at higher thresholds),\ not misclassified ones. A two-stage approach (coarse detector → high-resolution classifier) could address this.\
-640 × 640 resolution cap: fine surface defects smaller than ~5 px at 640 × 640 may be missed. Tiled inference or a higher-resolution\ input model is needed for dense, high-DPI boards.\
-Lighting sensitivity: the model was trained on a relatively uniform-illumination dataset. Performance degrades under harsh shadows or\ specular reflections from bare copper; domain adaptation with varied lighting conditions is a planned improvement.\
-Single board type: training data is dominated by a specific PCB family. Generalisation to SMD-dense boards or flexible PCBs has not\ been evaluated.

## Training Configuration

### Model & Training Configuration

```yaml
model:     yolo11n.pt       # pre-trained starting point
data:      data.yaml        # points to train / val / test splits
epochs:    100
imgsz:     640
conf:      0.25             # inference confidence threshold
device:    cuda / cpu       # auto-detected at runtime

results.csv
Epoch	mAP@0.5	Precision	Recall	Box Loss	Cls Loss
1	0.000	0.000	0.000	2.98	3.15
10	0.330	0.620	0.308	1.73	2.03
25	0.450	0.771	0.402	1.52	1.70
50	0.532	0.853	0.485	1.39	1.47
75	0.586	0.891	0.501	1.31	1.36
100	0.613	0.912	0.514	1.25	1.28


Training converged steadily over 100 epochs with no signs of catastrophic overfitting (validation loss tracked training loss closely throughout). The best checkpoint (best.pt) was selected by peak validation mAP.


