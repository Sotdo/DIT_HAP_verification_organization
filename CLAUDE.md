# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **DIT_HAP_verification_organization** project, which focuses on processing and organizing DIT-HAP (Deletion-Induced Transcriptional Homeostasis and Asynchronous Progression) verification data for fission yeast (*Schizosaccharomyces pombe*) experiments. The project handles image processing pipeline for tetrad and replica plates, gene verification data management, and systematic file organization for high-throughput screening experiments.

## Core Architecture

### Main Components

- **`src/image_processing.py`**: Advanced OpenCV-based image processing pipeline for colony detection and plate cropping
  - Template-style architecture with comprehensive type hints and dataclasses
  - Colony detection with adaptive thresholding and circularity analysis
  - Optimized Hough circle detection with configurable parameters
  - Tetrad and replica plate processing workflows with synchronized dimensions
  - Centroid adjustment algorithms for image alignment with outlier detection
  - Enhanced CLAHE parameters for improved contrast and colony detection
  - Progress tracking with tqdm for batch processing operations
  - Performance optimizations: memory efficiency, error handling, and graceful degradation
  - One-line function documentation following template standards

- **`src/utils.py`**: Data management classes and configuration
  - `verificationMetadata`: Handles gene verification metadata from Excel files
  - `roundConfig`: Manages round-specific folder structures and paths
  - Integration with PomBase gene databases and essentiality data

- **`src/rename_functions.py`**: File renaming utilities
  - Systematic image naming conventions for experiments
  - Integration with gene metadata for meaningful file names
  - Support for different experimental rounds and timepoints

### Directory Structure

```
DIT_HAP_verification_organization/
├── src/                      # Core processing modules
│   ├── image_processing.py   # OpenCV-based colony detection and plate processing
│   ├── utils.py             # Data classes and configuration management
│   ├── rename_functions.py  # File renaming utilities
│   └── __init__.py
├── scripts/                  # Execution scripts
│   ├── rename_image_names.py # Main renaming workflow
│   └── batch_crop_image.py  # Enhanced batch image cropping interface with template-style architecture
├── TEMPLATE.py               # Template script for new development
├── resource/                 # Data and reference files
│   ├── Hayles_2013_OB_merged_categories_sysIDupdated.xlsx
│   ├── all_for_verification_genes_by_round.xlsx
│   └── gene_IDs_names_products/
└── requirements.txt          # Python dependencies
```

## Common Development Tasks

### Setting up the Environment

The project uses a dedicated mamba environment named "opencv" for development:

```bash
# Activate the opencv environment
mamba activate opencv

# Verify Python path
which python  # Should show: /data/a/yangyusheng/.local/share/mamba/envs/opencv/bin/python

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

**Important**: Always activate the "opencv" environment before running any scripts or performing development work.

**Environment Activation Methods**:

**Method 1: Using mamba (preferred)**:
```bash
mamba activate opencv
```

**Method 2: Using PATH export** (if mamba activation doesn't work):
```bash
export PATH="/data/a/yangyusheng/.local/share/mamba/envs/opencv/bin:$PATH"
```

**Python Path Verification**:
```bash
# Verify you're using the correct Python environment
which python
# Should output: /data/a/yangyusheng/.local/share/mamba/envs/opencv/bin/python

# Test dependencies
python -c "import cv2, numpy, pandas, tqdm, openpyxl; print('All dependencies available')"
```

**Running Scripts with Correct Environment**:
```bash
# Method 1: Activate environment first, then run
mamba activate opencv
python scripts/rename_image_names.py

# Method 2: Use PATH export in single command
export PATH="/data/a/yangyusheng/.local/share/mamba/envs/opencv/bin:$PATH" && python scripts/rename_image_names.py

# Method 3: Use full Python path
/data/a/yangyusheng/.local/share/mamba/envs/opencv/bin/python scripts/rename_image_names.py
```

### Research Code Principles and Architecture

This project follows **research code principles** rather than production engineering:

#### Simplicity and Modern Python:
- **Dataclasses**: Use `@dataclass` for simple configuration management, not complex inheritance hierarchies
- **Type Hints**: Include type annotations for clarity without over-engineering
- **Clear Functions**: One purpose per function, minimal abstraction layers
- **Direct Parameters**: Pass parameters directly rather than complex configuration objects
- **Modern Features**: Use pathlib, f-strings, and other modern Python features

#### Code Organization:
- **Avoid Redundancy**: Don't duplicate configuration between `batch_crop_image.py` and `image_processing.py`
- **Simple Imports**: Standard library → Third-party → Project-specific with clear path handling
- **Minimal Dependencies**: Only import what's needed for the core functionality
- **Research-Focused**: Code should be easy to modify and experiment with, not locked into rigid patterns

#### Recent Simplifications:
- **Removed Over-Engineering**: Eliminated complex configuration class hierarchies
- **Direct Function Calls**: Simplified `batch_crop_image.py` to directly call processing functions
- **Streamlined image_processing.py**: Focus on core OpenCV functionality without excessive abstraction
- **Clean Configuration**: Single dataclass with clear parameter names

### Creating New Research Scripts

For research code, prioritize simplicity and modularity over rigid templates:

#### Simple Research Script Structure:

```python
"""
Brief description of what this research script does.
Usage: python scripts/research_script.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.resolve() / "src"))

# Import research modules
from utils import roundConfig
from image_processing import process_tetrad_images

# Simple configuration with dataclass
@dataclass
class Config:
    """Simple configuration for research experiment."""
    target_radius: int = 490
    min_colony_size: int = 50
    visualize_colonies: bool = True

def main():
    """Main function for research experiment."""
    config = Config()
    print("Starting research experiment...")

    # Your research logic here
    tetrad_size, radius = process_tetrad_images(
        input_dir='data/input',
        output_dir='results/output',
        **config.__dict__
    )

    print(f"Completed with size: {tetrad_size}")

if __name__ == '__main__':
    main()
```

#### Key Research Code Principles:
- **Minimal Structure**: Just imports, simple config, main function
- **Direct Parameters**: Pass parameters directly to functions
- **Easy Modification**: Parameters should be easy to change for experiments
- **Clear Purpose**: Each script should do one research task well
- **No Over-Engineering**: Avoid complex hierarchies and abstractions
- **Modern Python**: Use dataclasses, pathlib, type hints for clarity
- **Section Separators**: Use `# %% ------------------------------------ SECTION ------------------------------------ #` only if script gets long

The goal is code that's easy to understand, modify, and experiment with - not production-grade rigidity.

### Running Image Processing Pipeline

The main image processing workflow is handled through `src/image_processing.py`:

```python
# Process tetrad images
from src.image_processing import process_tetrad_images
tetrad_output_size, tetrad_radius = process_tetrad_images(
    input_dir='data/tetrad',
    output_dir='results/tetrad_cropped',
    target_radius=490,  # Optional: auto-calculated if None
    plate_to_tetrad_height_range=(40, 85),
    plate_to_tetrad_width_range=(10, 95),
    final_tetrad_height_percent=30,
    final_tetrad_width_percent=75,
    min_colony_size=50,
    circularity_threshold=0.7,
    visualize_colonies=True
)

# Process replica images
from src.image_processing import process_replica_images
process_replica_images(
    input_dir='data/replica',
    output_dir='results/replica_cropped',
    final_output_size_px=tetrad_output_size,
    tetrad_crop_radius=tetrad_radius,
    min_colony_size=25,  # Smaller colonies on replica plates
    circularity_threshold=0.6
)
```

### Batch File Renaming

To rename experimental images using systematic conventions:

```bash
# Run the renaming script from project root
python scripts/rename_image_names.py
```

**Important**: Always run scripts from the project root directory to ensure correct relative paths for resource files. The `src/` modules expect to be run from the project root where `resource/` directory is accessible.

This script:
- Processes all experimental rounds automatically
- Renames images with format: `{gene_num}_{gene_name}_{day_or_marker}_{colony_id}_{date}`
- Integrates with gene metadata from PomBase
- Requires resource files to be present in `resource/` directory

### Configuration Management

The project uses dataclasses for configuration:

```python
from src.utils import verificationMetadata, roundConfig

# Load verification metadata
verification_meta = verificationMetadata()

# Configure round-specific processing
round_config = roundConfig(
    round_folder_name="1st_round",
    raw_data_folder_path=Path("/path/to/raw/data"),
    output_folder_path=Path("/path/to/processed/data")
)
```

## Key Dependencies

- **OpenCV (`opencv-python`)**: Core image processing and computer vision
- **NumPy**: Numerical operations and array handling
- **Pandas**: Data manipulation for gene metadata
- **tqdm**: Progress bars for batch processing
- **openpyxl**: Excel file reading for verification data

## Data Processing Workflow

1. **Image Acquisition**: Raw microscopy images stored in round-specific directories
2. **Plate Detection**: Circle detection to identify individual plates on images
3. **Colony Detection**: Adaptive thresholding and circularity analysis to find colonies
4. **Centroid Calculation**: Determine colony centroids for plate alignment
5. **Image Cropping**: Extract consistently sized regions around colony clusters
6. **File Organization**: Systematic renaming with gene metadata integration

## Code Style and Development Standards

- **Use `TEMPLATE.py`** as the starting point for all new scripts
- **Import organization**: Standard library → third-party → project-specific modules
- **Type hints**: Required for all function parameters and return values
- **Documentation**: Comprehensive docstrings with purpose, usage examples, and input/output descriptions
- **Configuration**: Use `@dataclass` for configuration management
- **Path handling**: Use `pathlib.Path` for all file system operations

## Environment Troubleshooting

### Common Issues and Solutions

1. **Module Import Errors**: Ensure you're running from the project root directory
   ```bash
   # Run from DIT_HAP_verification_organization/ directory, not from src/ or scripts/
   pwd  # Should show: /data/c/yangyusheng_optimized/DIT_HAP_verification_organization
   ```

2. **Resource File Not Found**: Check that resource files exist in the correct location
   ```bash
   ls resource/all_for_verification_genes_by_round.xlsx
   ls resource/Hayles_2013_OB_merged_categories_sysIDupdated.xlsx
   ```

3. **Wrong Python Environment**: Verify you're using the opencv environment
   ```bash
   which python
   # Should output: /data/a/yangyusheng/.local/share/mamba/envs/opencv/bin/python
   ```

4. **Missing Dependencies**: Install required packages if needed
   ```bash
   /data/a/yangyusheng/.local/share/mamba/envs/opencv/bin/pip install -r requirements.txt
   ```

### Environment Verification Test

To verify your environment is correctly configured, you can run:
```python
# Quick environment test
import sys
sys.path.append('src')
import cv2, numpy as pandas, tqdm, openpyxl
print("Environment configured correctly!")
```

# ========================= KeyWord =========================

## Important Notes

- The project expects specific directory structures for raw and processed data
- Image processing parameters (colony size, circularity thresholds) may need adjustment for different experimental conditions
- Gene metadata is sourced from PomBase and essentiality studies
- The centroid adjustment algorithm helps maintain consistency across multiple plates and rounds
- Always follow the template structure when creating new scripts to maintain code consistency
- Always activate the "opencv" mamba environment before running scripts