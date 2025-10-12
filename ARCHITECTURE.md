
# Breast Ultrasound AI Analysis - Architecture

## Overview

This application provides AI-powered analysis of breast ultrasound images with binary classification (Benign/Malignant) and tumor segmentation. The architecture has been simplified to remove demo modes and medical recommendations.

## Current Architecture

### Entry Point
- **`main.py`**: Main application entry point, initializes GUI

### 1. Core ML Classes (Simplified)

#### `classification_model.py`
- **`ClassificationModel`**: Binary classification (Benign/Malignant only)
  - `load_model()`: Load Keras/TensorFlow model (.h5)
  - `predict()`: Returns classification results or None if model unavailable
  - `preprocess_image()`: Handle any size input, convert to 224x224


#### `segmentation_model.py` 
- **`UNetModel`**: PyTorch U-Net architecture for segmentation
- **`SegmentationModel`**: Wrapper for tumor segmentation
  - `load_model()`: Load PyTorch model (.pth)
  - `predict()`: Segment tumor regions, return binary mask
  - `preprocess_image()`: Handle any size input, convert to 256x256

#### `ml_backend.py`
- **`MLBackend`**: Simplified ML orchestrator
  - `analyze_image()`: Run complete analysis, returns None on failure
  - `initialize()`: Load and setup both models
  - `get_model_status()`: Check model loading status

### 2. Image Processing Classes

#### `image_processor.py`
- **`ImageProcessor`**: Essential image operations only
  - `load_image()`: Load image from file (any format)
  - `resize_image()`: Resize to target dimensions
  - `normalize_image()`: Scale pixels to 0-1 range
  - `convert_to_rgb()`: Handle different formats (BGR, grayscale, RGBA)
  - `validate_image()`: Check if image is valid

- **`ResultProcessor`**: Simplified result processing
  - `calculate_tumor_metrics()`: **Simplified**: Returns only pixel count
  - `detect_tumor_presence()`: Determine if tumor exists
  - `generate_overlay()`: Create visualization with red tumor highlight
  - `format_results()`: Combine analysis outputs


- **`FileManager`**: File operations
  - `save_analysis_report()`: Export results to text file (no recommendations)

### 3. GUI Classes

#### `gui_classes.py` (Active GUI)
- **`MainWindow`**: Main application controller
  - Complete tkinter interface with threading
  - Menu system and controls
  - Background analysis threading
  - Model loading interface

- **`ImageViewer`**: Image display component
  - `display_image()`: Show ultrasound images
  - `display_overlay()`: Show segmentation results
  - `zoom_in()`/`zoom_out()`/`reset_view()`: Image controls
  - `toggle_overlay()`: Show/hide tumor highlighting

- **`ResultsPanel`**: Analysis results display (simplified)
  - `display_results()`: Show classification and segmentation results
  - Classification probabilities and confidence
  - Segmentation results with pixel count only


### 4. Data Management

#### `data_manager.py`
- **`DataManager`**: Patient data and analysis storage
- **`ConfigManager`**: Application configuration
- **`FileManager`**: File operations and validation

## Current File Structure

```
new_gui/
├── main.py                    # Entry point (active)
├── gui_classes.py            # Main GUI implementation (active)
├── ml_backend.py             # ML orchestrator (active)
├── classification_model.py    # Binary classification (active)
├── segmentation_model.py     # Tumor segmentation (active)
├── image_processor.py        # Image processing utilities (active)
├── data_manager.py           # Data and config management (active)
├── requirements.txt          # Dependencies
├── breast_cancer_model_no_finetune.h5  # Classification model
├── unet_resnet34_best.pth    # Segmentation model
├── README.md                 # Documentation
├── ARCHITECTURE.md           # This file
├── main_gui.py              # Unused alternative GUI
└── gui.py                   # Unused simplified GUI
```

### Active Dependency Chain
```
main.py
└── gui_classes.py
    ├── ml_backend.py
    │   ├── classification_model.py
    │   ├── segmentation_model.py
    │   └── image_processor.py (ResultProcessor)
    ├── image_processor.py (ImageProcessor)
    └── data_manager.py (via imports)
```

## Key Features (Current)

### Simplified Analysis
- **Binary Classification**: Benign vs Malignant only
- **Pixel-based Measurements**: Tumor area in pixels only


### Image Support
- **Input**: Images of any dimensions (100x100 to 4K+)
- **Processing**: Automatic resize to model requirements
- **Formats**: JPG, PNG, BMP, TIFF, Grayscale, RGB, RGBA

### AI Analysis
- **Classification**: TensorFlow/Keras model for binary classification
- **Segmentation**: PyTorch U-Net for tumor detection
- **Results**: Clean analysis output with confidence scores


### GUI Features
- **Clean Interface**: Modern tkinter design
- **Image Viewer**: Zoom, overlay controls
- **Results Display**: Simplified output (classification + pixel count)
- **Threading**: Non-blocking UI during analysis
- **Model Status**: Clear indication of model availability

## Usage

### Quick Start
```bash
python main.py
```

### With Models
1. Ensure models are in directory:
   - `breast_cancer_model_no_finetune.h5` (Classification)
   - `unet_resnet34_best.pth` (Segmentation)
2. Load ultrasound image
3. Click "Analyze" 
4. View results and save if needed

### Example Results Output
```json
{
  "classification": {
    "prediction": "malignant",
    "confidence": 0.87,
    "probabilities": {
      "benign": 0.13,
      "malignant": 0.87
    }
  },
  "segmentation": {
    "tumor_detected": true,
    "metrics": {
      "area_pixels": 1247
    }
  },
  "processing_time": 2.3
}
```

### Code Integration Example
```python
from ml_backend import MLBackend

# Initialize
backend = MLBackend(
    classification_model_path="breast_cancer_model_no_finetune.h5",
    segmentation_model_path="unet_resnet34_best.pth"
)
backend.initialize()

# Analyze
results = backend.analyze_image("ultrasound.jpg")

if results:
    print(f"Classification: {results['classification']['prediction']}")
    print(f"Confidence: {results['classification']['confidence']:.2f}")
    print(f"Tumor area: {results['segmentation']['metrics']['area_pixels']} pixels")
else:
    print("Analysis failed - check model availability")
```


###  Features  
- Binary classification only (Benign/Malignant)
- Pixel count measurements only
- error handling (returns None on failure)
- Streamlined result display


