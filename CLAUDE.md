# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DIT-HAP verification organization pipeline for processing yeast colony microscopy images using OpenCV-based computer vision. The system detects colonies, crops plates, and organizes verification data for fission yeast genetic experiments.

## Core Architecture

### Processing Pipeline

1. **Image Processing (`src/image_processing.py`)**: OpenCV-based colony detection with Hough circle transforms, adaptive thresholding, and CLAHE enhancement. Main functions: `process_tetrad_images()` and `process_replica_images()`.

2. **Data Management (`src/utils.py`)**: Gene metadata handling with `verificationMetadata` class and round-based folder configuration with `roundConfig`.

3. **File Organization (`src/rename_functions.py`)**: Systematic renaming using gene metadata from PomBase integration.

4. **Table Generation (`src/table_organizer.py`)**: Quality assessment and verification table creation with multi-format export.

5. **PDF Reports (`src/pdf_generator.py`)**: Formatted PDF generation for verification documentation.

### Data Flow

Raw images → Plate detection (Hough circles) → Colony detection (adaptive thresholding) → Centroid alignment → Cropped output → Metadata integration → Tables/PDFs

## Essential Commands

### Environment Setup (Critical)

```bash
# Required: Activate opencv environment
mamba activate opencv
# Verify Python path: /data/a/yangyusheng/.local/share/mamba/envs/opencv/bin/python

# Install dependencies
pip install -r requirements.txt

# Test environment
python -c "import cv2, numpy, pandas, tqdm, openpyxl, loguru, skimage; print('Environment OK')"
```

### Main Workflows

```bash
# 1. Process all images (primary workflow)
python scripts/batch_crop_image.py

# 2. Rename with gene metadata
python scripts/rename_image_names.py

# 3. Generate tables and PDFs
python scripts/organize_tables.py
```

### New Script Development

```bash
# Always start from template
cp TEMPLATE.py scripts/new_script.py
# Modify following the structure with %% section separators
```

## Key Configuration

### Image Processing Parameters

```python
@dataclass
class ImageProcessingConfig:
    target_radius: int = 490  # Plate radius in pixels
    min_colony_size: int = 50  # Tetrad: 50, Replica: 500
    circularity_threshold: float = 0.7  # Tetrad: 0.7, Replica: 0.45
    adaptive_block_size: int = 30  # Tetrad: 30, Replica: 200
    contrast_alpha: float = 1.0  # Tetrad: 1.0, Replica: 1.6
    visualize_colonies: bool = True
```

### Directory Structure

```
/hugedata/YangshengYang/DIT_HAP_verification/
├── data/processed_data/DIT_HAP_deletion/  # Input images (renamed)
└── data/cropped_images/DIT_HAP_deletion/  # Output images
    └── {round_name}/
        ├── 3d/  # 3-day growth
        └── 5d/  # 5-day growth
```

## Critical Requirements

- **Environment**: Must use "opencv" mamba environment
- **Working Directory**: Run all scripts from repository root
- **Resource Files**: `resource/` directory must contain Excel files with gene metadata
- **Data Paths**: Hardcoded paths in `/hugedata/` - must be updated for different environments

## Development Guidelines

- Use `@dataclass` for configuration (avoid complex config objects)
- Modern Python type hints (PEP 585+): `dict[str, int]` not `Dict[str, int]`
- All scripts must add `src` to path: `sys.path.append(str(Path(__file__).parent.parent / "src"))`
- Research-focused design: prioritize experimental flexibility over production robustness
- Error handling with `@logger.catch` decorator

## Common Issues

1. **Wrong Python environment** - Verify with `which python`
2. **Missing resource files** - Check `resource/` directory exists with Excel files
3. **Hardcoded paths** - Update data paths in `BatchConfig` for different environments
4. **Memory issues** - Process large image batches in smaller chunks