# Close-Range Photogrammetry - Control Point Detection

A desktop application for automatic detection and subpixel-precise localization of control point targets in high-resolution close-range photogrammetry images.

## Overview

This system detects ring-shaped control point markers (concentric black rings on white background with crosshair lines) commonly used in close-range photogrammetry. It processes high-resolution photographs (~8500x5000 px), locates each target's center with subpixel accuracy, and exports structured coordinate data for downstream space resection calculations.

## Features

### Detection Pipeline
- **Adaptive preprocessing** -- CLAHE histogram equalization, bilateral filtering, and adaptive thresholding, automatically tuned to target size
- **Multi-stage contour filtering** -- area, circularity, aspect ratio, and convexity filters with ring-structure verification to reject false positives (numbers, crosshairs, noise)
- **Two-level subpixel localization**
  - Level 1: Ellipse fitting on outer/inner ring contours (~0.1 px accuracy)
  - Level 2: Grayscale-weighted centroid refinement (~0.02 px accuracy)

### GUI Application
- **Three-panel layout** -- file browser, image viewer, parameter & point info panel
- **Large image support** -- smooth mouse-wheel zoom and middle-button pan on multi-megapixel images
- **Visual overlay** -- detected targets rendered as green circle + crosshair + ID labels; selected point highlighted in yellow
- **Semi-automatic correction** -- correct ID, delete point, or manually add point by clicking on image
- **Keyboard nudging** -- arrow keys move selected point by 1 px (Shift + arrow for 0.1 px fine adjustment)
- **Batch processing** -- detect all images in a folder via background thread with progress bar
- **JSON export** -- structured output with pixel coordinates, confidence, ellipse parameters, and eccentricity

## Requirements

- Python 3.10+
- OpenCV (`opencv-python >= 4.8`)
- NumPy (`numpy >= 1.24`)
- PyQt5 (`PyQt5 >= 5.15`)

## Installation

```bash
git clone https://github.com/LanZHongHe/CloseRangePhotogrammetry.git
cd CloseRangePhotogrammetry

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

### Workflow

1. **Open folder** -- click "Open Folder..." and select a directory containing images (.jpg, .png, .tif, .bmp)
2. **Load image** -- click a filename in the left panel to display it in the viewer
3. **Adjust parameters** -- tune Target Size, Circularity Min, and Area Tolerance in the right panel
4. **Detect** -- click "Detect Current" for a single image or "Detect All" for batch processing
5. **Review & correct** -- click markers to inspect details, correct IDs, delete false positives, or add missed targets
6. **Export** -- click "Export JSON..." to save all detection results

### Output Format

The exported JSON contains per-image detection results:

```json
{
  "image": "photo/DSC_0035.JPG",
  "image_size": [8256, 5504],
  "detection_time": "2026-04-14T16:14:33",
  "targets": [
    {
      "id": "181",
      "pixel_x": 8062.7158,
      "pixel_y": 4129.8107,
      "confidence": 0.6986,
      "source": "auto",
      "subpixel_method": "centroid",
      "ellipse": {
        "semi_major": 74.34,
        "semi_minor": 68.87,
        "angle_deg": 118.85
      },
      "eccentricity": 0.3766
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `pixel_x`, `pixel_y` | Subpixel-precise center coordinates |
| `confidence` | Detection confidence (0.7 + 0.3 x circularity) |
| `source` | `"auto"` (detected) or `"manual"` (added by user) |
| `ellipse` | Fitted ellipse parameters (handles affine deformation from oblique views) |
| `eccentricity` | Ellipse eccentricity -- 0 = circle, approaching 1 = highly elongated |

## Project Structure

```
CloseRangePhotogrammetry/
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
├── gui/
│   └── main_window.py         # PyQt5 GUI (MainWindow, ImageViewer, DetectionWorker)
├── src/
│   ├── preprocessing.py       # CLAHE, bilateral filter, adaptive threshold
│   ├── detection.py           # Contour-based coarse ring detection
│   ├── subpixel.py            # Ellipse fitting + centroid refinement
│   ├── id_recognition.py      # Sequential ID assignment
│   ├── data_model.py          # TargetPoint, EllipseInfo, DetectionResult dataclasses
│   └── io_utils.py            # JSON serialization
├── photo/                     # Sample images
└── ControlPoint_Detection_Design.md  # Detailed design document (Chinese)
```

## How It Works

```
Input Image
    │
    ▼
┌─────────────────┐
│  Preprocessing   │  Grayscale → CLAHE → Bilateral Filter → Adaptive Threshold
└────────┬────────┘
         ▼
┌─────────────────┐
│ Contour Detection│  findContours(RETR_TREE) → geometric filters → ring verification
└────────┬────────┘
         ▼
┌─────────────────┐
│  Subpixel Fit    │  Ellipse fitting (L1) → Centroid refinement (L2)
└────────┬────────┘
         ▼
┌─────────────────┐
│  ID Assignment   │  Sort top-to-bottom, left-to-right → sequential numbering
└────────┬────────┘
         ▼
   JSON Output
```

## Acknowledgements

This project was developed as part of close-range photogrammetry coursework. The control point detection design follows photogrammetric best practices for coded target recognition and subpixel measurement.
