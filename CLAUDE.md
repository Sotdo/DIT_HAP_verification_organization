# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **DIT_HAP_verification_organization** project, which focuses on processing and organizing DIT-HAP (Deletion-Induced Transcriptional Homeostasis and Asynchronous Progression) verification data for fission yeast (*Schizosaccharomyces pombe*) experiments. The project handles image processing pipeline for tetrad and replica plates, gene verification data management, and systematic file organization for high-throughput screening experiments.

## Core Architecture

### Main Components

- **`src/image_processing.py`**: Core computer vision pipeline using OpenCV for colony detection and plate cropping
  - Colony detection with adaptive thresholding and circularity analysis
  - Plate circle detection using Hough circle transform
  - Tetrad and replica plate processing workflows
  - Centroid adjustment algorithms for image alignment

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
│   └── batch_crop_image.py  # Batch image cropping interface
├── TEMPLATE.py               # Template script for new development
├── resource/                 # Data and reference files
│   ├── Hayles_2013_OB_merged_categories_sysIDupdated.xlsx
│   ├── all_for_verification_genes_by_round.xlsx
│   └── gene_IDs_names_products/
└── requirements.txt          # Python dependencies
```

## Common Development Tasks

### Setting up the Environment

```bash
# Install dependencies
pip install -r requirements.txt
```

### Creating New Scripts

When adding new functionality, use the provided `TEMPLATE.py` as a starting point. The template follows these conventions:

- **Documentation**: Comprehensive docstring with purpose, usage, and format descriptions
- **Import Organization**: Standard library first, then third-party, then project-specific modules
- **Constants**: UPPERCASE naming for configuration values
- **Dataclasses**: Use `@dataclass` for configuration management
- **Type Hints**: Include type annotations for all function parameters and return values
- **Main Block**: Use `if __name__ == "__main__":` pattern with a separate `main()` function

Template structure:
```python
"""
[Detailed script documentation]
"""

# Standard library imports
import sys
from pathlib import Path
from dataclasses import dataclass

# Third-party imports
import pandas as pd

# Project-specific imports (with proper path handling)
# SCRIPT_DIR = Path(__file__).parent.resolve()
# TARGET_path = str((SCRIPT_DIR / "../../src").resolve())
# sys.path.append(TARGET_path)
# from utils import custom_function

# Constants (UPPERCASE)
EXAMPLE_THRESHOLD = 100
DEFAULT_VALUE = 0.5

# Configuration dataclasses
@dataclass
class Config:
    parameter1: int = 3

# Core functions with type hints and docstrings
def function1(param1: str, param2: int) -> pd.DataFrame:
    """One line docs of function1."""
    pass

def main():
    """Main function to execute the script logic."""
    pass

if __name__ == "__main__":
    main()
```

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
# Run the renaming script
python scripts/rename_image_names.py
```

This script:
- Processes all experimental rounds automatically
- Renames images with format: `{gene_num}_{gene_name}_{day_or_marker}_{colony_id}_{date}`
- Integrates with gene metadata from PomBase

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

## Important Notes

- The project expects specific directory structures for raw and processed data
- Image processing parameters (colony size, circularity thresholds) may need adjustment for different experimental conditions
- Gene metadata is sourced from PomBase and essentiality studies
- The centroid adjustment algorithm helps maintain consistency across multiple plates and rounds
- Always follow the template structure when creating new scripts to maintain code consistency