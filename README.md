# DIT-HAP Verification Organization

Automated image processing pipeline for DIT-HAP (Deletion-Induced Transcriptional Homeostasis and Asynchronous Progression) verification experiments in fission yeast (*Schizosaccharomyces pombe*).

## Overview

This pipeline processes high-throughput yeast colony microscopy images using OpenCV-based computer vision techniques to:
- Detect and analyze yeast colonies
- Crop and align plate images
- Organize verification data with gene metadata
- Generate systematic reports

## Quick Start

### 1. Environment Setup

```bash
# Clone and navigate to project
git clone <repository-url>
cd DIT_HAP_verification_organization

# Activate required environment
mamba activate opencv

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import cv2, numpy, pandas, tqdm, openpyxl, loguru; print('✅ Ready')"
```

### 2. Run Pipeline

```bash
# Process all images
python scripts/batch_crop_image.py

# Rename files with gene metadata
python scripts/rename_image_names.py

# Generate verification tables and PDFs
python scripts/organize_tables.py
```

## Project Structure

```
DIT_HAP_verification_organization/
├── src/                      # Core processing modules
│   ├── image_processing.py   # OpenCV-based colony detection
│   ├── utils.py             # Data management and metadata
│   ├── rename_functions.py  # File renaming utilities
│   ├── table_organizer.py   # Table generation
│   └── pdf_generator.py     # PDF report creation
├── scripts/                  # Execution scripts
│   ├── batch_crop_image.py  # Main image processing
│   ├── rename_image_names.py # File renaming
│   └── organize_tables.py   # Table and PDF generation
├── resource/                 # Required data files
└── results/                  # Output directory
```

## Key Features

- **Advanced Colony Detection**: Hough circle transforms with adaptive thresholding
- **Synchronized Processing**: Coordinated tetrad and replica plate dimensions
- **Gene Integration**: PomBase metadata for systematic file organization
- **Multi-format Export**: CSV, Excel, and JSON output capabilities
- **Research-Focused**: Flexible parameters for experimental optimization

## Dependencies

- Python 3.8+
- OpenCV 4.x
- NumPy, Pandas, tqdm
- openpyxl, loguru
- scikit-image

*Full development guidelines and detailed documentation are available in [CLAUDE.md](CLAUDE.md).*