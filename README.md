# AI Image Cropper

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Intelligent image cropping tool with multiple detection methods including You Only Look Once (YOLO), DEtection TRansformer (DETR), Real-Time DEtection TRansformer (RT-DETR), and traditional computer vision algorithms. Available as both an interactive Gradio web interface and a command-line tool.

## Features

### Detection Methods

- **YOLO** - Fast and accurate deep learning (recommended)
- **RT-DETR** - Real-time DETR with faster inference and similar accuracy
- **DETR** - State-of-the-art transformer-based detection
- **Contour** - Fast, works well with clear backgrounds
- **Saliency** - Identifies visually interesting regions
- **Edge** - Canny edge detection
- **GrabCut** - Foreground/background segmentation

### Capabilities

- 🎯 **Object Detection**: Detect specific objects (person, car, couch, etc.)
- 📐 **Custom Aspect Ratios**: Set target aspect ratios (16:9, 4:3, 1:1, custom)
- 🔲 **Smart Padding**: Add breathing room around detected objects
- 🎨 **Batch Processing**: Crop all detected objects individually
- 🖼️ **Multiple Formats**: JPEG, PNG, WebP support
- 🌐 **Web UI (User Interface)**: User-friendly Gradio interface
- ⌨️ **CLI (Command-Line Interface)**: Full command-line interface for automation

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone project and setup
git clone https://github.com/garystafford/ai-image-cropper.git
cd ai-image-cropper

# Install dependencies (creates .venv automatically)
uv sync

uv sync --dev

source .venv/bin/activate

# Run the web interface (launches automatically)
uv run crop-ui

# Or use the CLI directly
uv run crop-cli --help

# Single object detection and cropping with visualization
uv run crop-cli sample_images/sample_image_00001.jpg --method yolo --visualize

# Batch object detection and cropping
uv run crop-cli sample_images/sample_image_00001.jpg --method yolo --batch-crop
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

### Why uv?

- ⚡ **10-100x faster** than traditional package managers
- 🔒 **Deterministic builds** with automatic lock file generation
- 🎯 **All-in-one tool** - replaces multiple Python tools (pip-tools, pipx, poetry, pyenv, virtualenv)
- 📦 **Better dependency resolution** with clear error messages
- 🚀 **Modern Python tooling** written in Rust for maximum performance

### 1. Install uv

**macOS/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone the Repository

```bash
git clone https://github.com/garystafford/image-cropper.git
cd image-cropper
```

### 3. Install Project and Dependencies

```bash
# Create virtual environment and install dependencies (one command!)
uv sync

# For development with testing tools
uv sync --all-extras
```

**Note**: On first run, YOLO and RT-DETR will automatically download their model files (~200-300MB for YOLO v12 X-Large, ~200MB for RT-DETR).

### 4. Using Entry Points (Recommended)

After installation, use the convenient entry points:

```bash
# Launch web interface
uv run crop-ui

# Use CLI tool
uv run crop-cli image.jpg --method yolo --visualize
```

Or activate the virtual environment for direct access:

```bash
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Then use directly
crop-ui
crop-cli image.jpg --method yolo
```

## Usage

### Web Interface (Gradio)

#### Start the Application

```bash
# Using entry point (recommended)
uv run crop-ui

# Or directly with Python
uv run python app.py
```

This will start the Gradio server at `http://127.0.0.1:7860`

#### Using the Interface

1. **Upload Image**: Drag and drop or click to upload (JPEG, PNG, WebP)
2. **Choose Detection Method**: Select from YOLO, RT-DETR, DETR, Contour, Saliency, Edge, or GrabCut
3. **Configure Parameters**:
   - **Object to Detect**: Specify object name (e.g., "couch", "person") or leave empty
   - **Confidence Threshold**: Adjust detection sensitivity (0.1-1.0)
   - **Target Aspect Ratio**: Set custom ratio (16:9, 4:3, etc.) or leave empty
   - **Keep Original Aspect Ratio**: Toggle to maintain original proportions
   - **Padding**: Add space around object (0-50%)
4. **Process**: Click "🚀 Process Image" or wait for auto-processing
5. **View Results**:
   - Left preview shows detection with green bounding box
   - Right panel shows final cropped image
   - Processing information displayed below
6. **Batch Crop** (optional): Click "🖼️ Batch Crop All Objects" to save all detected objects

### Command-Line Interface

#### Basic Usage

```bash
# Using entry point (recommended)
uv run crop-cli image.jpg --visualize --crop-output output.jpg

# Or directly with Python
python cropper.py image.jpg --visualize --crop-output output.jpg
```

#### Single Object Detection

```bash
# Detect and crop a couch with custom aspect ratio
python cropper.py living_room.jpg --method yolo --object couch --aspect-ratio 16:9 --crop-output couch.jpg

# Detect person with RT-DETR (faster than DETR)
python cropper.py photo.jpg --method rt-detr --object person --confidence 0.5 --padding 10 --crop-output person.jpg

# Detect person with DETR, add padding
python cropper.py photo.jpg --method detr --object person --confidence 0.8 --padding 10 --crop-output person.jpg

# Use contour detection with visualization
python cropper.py product.jpg --method contour --threshold 200 --padding 5 --visualize
```

#### Batch Processing

Batch processing automatically crops all detected objects and saves them separately:

```bash
# Detect and crop all people in a family photo
python cropper.py family.jpg --method yolo --batch-crop --batch-output-dir ./people

# Batch crop with RT-DETR for faster processing
python cropper.py room.jpg --method rt-detr --batch-crop --confidence 0.5

# Batch crop with custom aspect ratio and padding (DETR)
python cropper.py room.jpg --method detr --batch-crop --aspect-ratio 4:3 --padding 15

# Batch crop all objects (no specific object filter)
python cropper.py scene.jpg --method yolo --batch-crop --confidence 0.7
```

#### CLI Options

```
positional arguments:
  image_path            Path to the input image

options:
  --method              Detection method: contour, saliency, edge, grabcut, detr, rt-detr, yolo
  --object              Target object(s) to detect (can specify multiple times)
  --confidence          Confidence threshold for deep learning methods (0-1, default: 0.7)
  --keep-aspect         Maintain original aspect ratio
  --aspect-ratio        Custom aspect ratio (e.g., 16:9, 4:3, 1.5, 2.35:1)
  --padding             Padding percentage around object (default: 5)
  --threshold           Threshold value for contour detection (default: 240)
  --visualize           Display detection visualization window
  --crop-output         Save cropped image to specified path
  --batch-crop          Crop all detected objects individually (YOLO/RT-DETR/DETR only)
  --batch-output-dir    Output directory for batch crop (default: cropped_images)
  --image-quality       JPEG quality for saved images (1-100, default: 95)
  --debug               Save debug images during processing
```

## Tips

- **YOLO** is the fastest and most accurate for common objects
- **RT-DETR** offers a balance between speed and accuracy, faster than DETR with similar results
- **DETR** provides detailed object detection but is slower than YOLO and RT-DETR
- For best results, use padding of 5-10%
- Batch mode works only with YOLO, RT-DETR, and DETR methods
- Common detectable objects: person, car, couch, chair, dog, cat, bottle, laptop, bicycle, etc.
- Use `--visualize` in CLI to preview detection before cropping

## Troubleshooting

### Image Format Error

Ensure your image is JPEG (.jpg, .jpeg), PNG (.png), or WebP (.webp)

### Model Download on First Run

YOLO and RT-DETR will download their model files on first use (may take 2-5 minutes for YOLO v12 X-Large, 2-5 minutes for RT-DETR)

### DETR/RT-DETR Memory Usage

DETR and RT-DETR require more memory than YOLO. RT-DETR is more efficient than DETR. For large images, consider using YOLO or RT-DETR instead of DETR.

### No Objects Detected

- Lower the confidence threshold
- Try a different detection method
- Verify the object name is in the model's vocabulary

## Project Structure

```
image-cropper/
├── app.py                    # Gradio web interface
├── cropper.py                # Core processing engine + CLI
├── config.py                 # Configuration constants
├── pyproject.toml            # uv project configuration
├── py.typed                  # Type checking marker
├── .python-version           # Python version (3.13)
├── uv.lock                   # Dependency lock file
├── LICENSE                   # MIT License
├── CHANGELOG.md              # Version history
├── CONTRIBUTING.md           # Contribution guidelines
├── README.md                 # This file
├── .venv/                    # Virtual environment (created by uv)
├── models/                   # YOLO model files (auto-downloaded)
├── cropped_images/           # Default batch crop output
└── sample_images/            # Sample images for testing
```

## Requirements

- **Python**: 3.13+
- **Package Manager**: [uv](https://docs.astral.sh/uv/)
- **Key Dependencies**:
  - gradio >= 5.0.0
  - opencv-python >= 4.8.0
  - ultralytics >= 8.0.0 (YOLO)
  - transformers >= 4.30.0 (DETR and RT-DETR)
  - torch >= 2.0.0
  - numpy >= 1.24.0
  - pillow >= 10.0.0

See [`pyproject.toml`](pyproject.toml) for complete dependency list.

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and release notes.

## License

This project is open source and available under the [MIT License](LICENSE).

Copyright (c) 2025 Gary A. Stafford
