# DIT-HAP Verification Organization

A comprehensive image processing and data management pipeline for DIT-HAP (Deletion-Induced Transcriptional Homeostasis and Asynchronous Progression) verification experiments in fission yeast (*Schizosaccharomyces pombe*).

## Overview

This project provides automated tools for processing high-throughput yeast colony images, detecting colonies, and organizing verification data for systematic genetic analysis. The pipeline handles both tetrad and replica plate processing with synchronized dimensions and systematic file naming conventions.

## Key Features

- **Automated Colony Detection**: OpenCV-based detection with adaptive thresholding and circularity analysis
- **Plate Processing**: Hough circle detection for accurate plate identification and cropping
- **Systematic File Organization**: Gene-aware renaming with PomBase metadata integration
- **Batch Processing**: High-throughput capabilities with progress tracking
- **Research-Focused Design**: Easy parameter modification for experimental optimization

## Quick Start

### Prerequisites

- Python 3.8+
- Mamba/Conda environment manager
- Access to raw DIT-HAP verification data

### Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd DIT_HAP_verification_organization

# Create and activate the opencv environment
mamba create -n opencv python=3.9
mamba activate opencv

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Process tetrad images
python scripts/batch_crop_image.py

# Rename images with systematic conventions
python scripts/rename_image_names.py
```

## Project Structure

```
DIT_HAP_verification_organization/
├── src/                      # Core processing modules
│   ├── image_processing.py   # OpenCV-based colony detection and plate processing
│   ├── utils.py             # Data classes and configuration management
│   ├── rename_functions.py  # File renaming utilities
│   └── __init__.py
├── scripts/                  # Execution scripts
│   ├── rename_image_names.py # Main renaming workflow
│   └── batch_crop_image.py  # Enhanced batch image cropping interface
├── TEMPLATE.py               # Template script for new development
├── resource/                 # Data and reference files
│   ├── Hayles_2013_OB_merged_categories_sysIDupdated.xlsx
│   ├── all_for_verification_genes_by_round.xlsx
│   └── gene_IDs_names_products/
├── requirements.txt          # Python dependencies
├── CLAUDE.md                # Development guidelines for Claude Code
└── README.md               # This file
```

## Core Components

### Image Processing Pipeline (`src/image_processing.py`)

Advanced OpenCV-based pipeline for colony detection and plate processing:

- **Colony Detection**: Adaptive thresholding with configurable circularity thresholds
- **Plate Identification**: Hough circle transform with parameter optimization
- **Image Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Centroid Calculation**: Outlier detection and plate alignment algorithms
- **Batch Processing**: Memory-efficient processing with tqdm progress tracking

#### Key Parameters

```python
# Tetrad plate processing
tetrad_output_size, tetrad_radius = process_tetrad_images(
    input_dir='data/tetrad',
    output_dir='results/tetrad_cropped',
    target_radius=490,           # Plate radius in pixels
    min_colony_size=50,          # Minimum colony size
    circularity_threshold=0.7,   # Circularity filter
    visualize_colonies=True      # Show detection results
)

# Replica plate processing (uses tetrad output for synchronization)
process_replica_images(
    input_dir='data/replica',
    output_dir='results/replica_cropped',
    final_output_size_px=tetrad_output_size,
    tetrad_crop_radius=tetrad_radius,
    min_colony_size=25,          # Smaller colonies on replicas
    circularity_threshold=0.6   # More lenient for replicas
)
```

### Data Management (`src/utils.py`)

Configuration and metadata management classes:

- **`verificationMetadata`**: Handles gene verification data from Excel files
- **`roundConfig`**: Manages round-specific folder structures and paths
- **PomBase Integration**: Gene names, systematic IDs, and essentiality data

### File Renaming (`src/rename_functions.py`)

Systematic image naming with gene metadata:

```python
# Output format: {gene_num}_{gene_name}_{day_or_marker}_{colony_id}_{date}
# Example: 123_adh1_3d_colonyA_20251128
```

## Dependencies

- **OpenCV (`opencv-python`)**: Core image processing and computer vision
- **NumPy**: Numerical operations and array handling
- **Pandas**: Data manipulation for gene metadata
- **tqdm**: Progress bars for batch processing
- **openpyxl**: Excel file reading for verification data
- **loguru**: Advanced logging with colorization and formatting

## Data Processing Workflow

1. **Image Acquisition**: Raw microscopy images stored in round-specific directories
2. **Plate Detection**: Circle detection to identify individual plates on images
3. **Colony Detection**: Adaptive thresholding and circularity analysis to find colonies
4. **Centroid Calculation**: Determine colony centroids for plate alignment
5. **Image Cropping**: Extract consistently sized regions around colony clusters
6. **File Organization**: Systematic renaming with gene metadata integration

## Usage Examples

### Custom Image Processing Script

```python
"""
Custom script for processing specific experimental conditions.
"""
import sys
from pathlib import Path
from dataclasses import dataclass

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from image_processing import process_tetrad_images
from utils import roundConfig

@dataclass
class CustomConfig:
    """Configuration for custom experiment."""
    target_radius: int = 490
    min_colony_size: int = 50
    circularity_threshold: float = 0.7

def main():
    config = CustomConfig()

    tetrad_size, radius = process_tetrad_images(
        input_dir='data/custom_tetrad',
        output_dir='results/custom_output',
        target_radius=config.target_radius,
        min_colony_size=config.min_colony_size,
        circularity_threshold=config.circularity_threshold,
        visualize_colonies=True
    )

    print(f"Processing complete. Output size: {tetrad_size}px")

if __name__ == '__main__':
    main()
```

### Batch Processing with Custom Parameters

```python
# Modify processing parameters for different experimental conditions
from image_processing import ImageProcessingConfig

# Custom configuration for challenging images
custom_config = ImageProcessingConfig(
    min_colony_size=30,          # Smaller colonies
    circularity_threshold=0.5,   # More lenient filtering
    adaptive_block_size=50,      # Larger adaptive threshold blocks
    contrast_alpha=1.2          # Enhanced contrast
)
```

## Configuration Management

### Round Configuration

```python
from utils import roundConfig

# Configure round-specific processing
round_config = roundConfig(
    round_folder_name="1st_round",
    raw_data_folder_path=Path("/path/to/raw/data"),
    output_folder_path=Path("/path/to/processed/data")
)

# Access folder structure
print(round_config.all_sub_folders["3d"]["input"])   # Raw 3-day images
print(round_config.all_sub_folders["3d"]["output"])  # Processed 3-day images
```

### Verification Metadata

```python
from utils import verificationMetadata

# Load gene verification data
verification_meta = verificationMetadata()

# Access gene mappings
print(verification_meta.num2gene[123])  # Get gene name by number
print(verification_meta.verification_genes.head())  # View verification data
```

## Troubleshooting

### Common Issues

1. **Module Import Errors**
   ```bash
   # Ensure running from project root
   pwd  # Should show: /path/to/DIT_HAP_verification_organization

   # Check environment
   which python  # Should show opencv environment path
   ```

2. **Resource File Not Found**
   ```bash
   # Verify resource files exist
   ls resource/all_for_verification_genes_by_round.xlsx
   ls resource/Hayles_2013_OB_merged_categories_sysIDupdated.xlsx
   ```

3. **Image Processing Parameters**
   - Adjust `min_colony_size` for different colony sizes
   - Modify `circularity_threshold` for colony shape variations
   - Tune `adaptive_block_size` for different contrast conditions

### Environment Verification

```bash
# Test environment setup
python -c "
import sys
sys.path.append('src')
import cv2, numpy as pandas, tqdm, openpyxl, loguru
print('All dependencies available!')
"
```

## Development Guidelines

- **Use `TEMPLATE.py`** as starting point for new scripts
- **Research-focused design**: Prioritize flexibility over rigid architecture
- **Modern Python practices**: Use dataclasses, pathlib, type hints
- **Clear documentation**: One-line function docs with usage examples
- **Parameter tuning**: Make experimental parameters easily configurable

See [`CLAUDE.md`](CLAUDE.md) for comprehensive development guidelines and best practices.

## License

[Add license information here]

## Citation

[Add citation information here]

## Contact

[Add contact information here]