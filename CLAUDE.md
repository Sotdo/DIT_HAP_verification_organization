# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **DIT_HAP_verification_organization** project, which focuses on processing and organizing DIT-HAP (Deletion-Induced Transcriptional Homeostasis and Asynchronous Progression) verification data for fission yeast (*Schizosaccharomyces pombe*) experiments. The project provides an automated image processing pipeline for high-throughput yeast colony detection, plate processing, gene verification data management, and systematic file organization.

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

- **`src/utils.py`**: Data management classes and configuration
  - `verificationMetadata`: Handles gene verification metadata from Excel files
  - `roundConfig`: Manages round-specific folder structures and paths
  - Integration with PomBase gene databases and essentiality data

- **`src/rename_functions.py`**: File renaming utilities
  - Systematic image naming conventions for experiments
  - Integration with gene metadata for meaningful file names
  - Support for different experimental rounds and timepoints

- **`src/table_organizer.py`**: Table structure organization and data processing
  - Quality assessment for cropped images
  - Verification table creation with comprehensive metadata
  - Multi-format export capabilities (CSV, Excel, JSON)
  - Integration with gene verification data

- **`src/pdf_generator.py`**: PDF document generation
  - Formatted PDF creation for verification reports
  - Gene-specific PDF generation with customizable layouts
  - Integration with table organization results

### Directory Structure

```
DIT_HAP_verification_organization/
├── src/                      # Core processing modules
│   ├── image_processing.py   # OpenCV-based colony detection and plate processing
│   ├── utils.py             # Data classes and configuration management
│   ├── rename_functions.py  # File renaming utilities
│   ├── table_organizer.py   # Table structure organization and PDF generation
│   ├── pdf_generator.py     # PDF generation functionality
│   └── __init__.py
├── scripts/                  # Execution scripts
│   ├── rename_image_names.py # Main renaming workflow
│   ├── batch_crop_image.py  # Enhanced batch image cropping interface
│   └── organize_tables.py   # Table organization and PDF generation workflow
├── resource/                 # Data and reference files
│   ├── Hayles_2013_OB_merged_categories_sysIDupdated.xlsx
│   ├── all_for_verification_genes_by_round.xlsx
│   └── gene_IDs_names_products/
├── results/                  # Output directory
│   └── merged_pdfs/
├── TEMPLATE.py               # Template script for new development
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── PROMPT.md                # Function request framework
└── README_Task3.md          # Task-specific documentation
```

## Essential Commands and Workflows

### Environment Setup

**Critical: Always use the opencv environment for development**

```bash
# Activate the opencv environment (REQUIRED)
mamba activate opencv

# Verify correct Python environment
which python
# Should output: /data/a/yangyusheng/.local/share/mamba/envs/opencv/bin/python

# Test core dependencies
python -c "import cv2, numpy, pandas, tqdm, openpyxl, loguru; print('Environment OK')"

# Install dependencies if needed (only run from opencv environment)
pip install -r requirements.txt
```

**Alternative Environment Methods** (if mamba activation fails):
```bash
# Method 2: Direct PATH export
export PATH="/data/a/yangyusheng/.local/share/mamba/envs/opencv/bin:$PATH"

# Method 3: Full Python path for commands
/data/a/yangyusheng/.local/share/mamba/envs/opencv/bin/python <script>
```

### Main Processing Workflows

#### 1. Image Processing Pipeline
```bash
# Process tetrad and replica plate images (primary workflow)
python scripts/batch_crop_image.py

# Custom image processing with specific parameters
python scripts/batch_crop_image.py --custom-config
```

#### 2. File Renaming System
```bash
# Rename all experimental images with systematic conventions
python scripts/rename_image_names.py
```

#### 3. Table Organization and PDF Generation
```bash
# Create verification tables and generate PDF reports
python scripts/organize_tables.py
```

### Research Script Development

**Creating New Research Scripts:**
```bash
# Always start from TEMPLATE.py for new scripts
cp TEMPLATE.py scripts/my_new_script.py
# Then modify the new script following research code principles
```

**Custom Image Processing Example:**
```python
"""
Custom research script for specific experimental conditions.
Usage: python scripts/custom_analysis.py
"""
import sys
from pathlib import Path
from dataclasses import dataclass

# Add src to path (REQUIRED for all scripts)
sys.path.append(str(Path(__file__).parent.parent / "src"))

from image_processing import process_tetrad_images, ImageProcessingConfig

@dataclass
class CustomConfig:
    """Configuration for custom experiment."""
    target_radius: int = 490
    min_colony_size: int = 50
    circularity_threshold: float = 0.7
    visualize_colonies: bool = True

def main():
    config = CustomConfig()

    # Process with custom parameters
    tetrad_size, radius = process_tetrad_images(
        input_dir='data/experimental_tetrad',
        output_dir='results/custom_output',
        target_radius=config.target_radius,
        min_colony_size=config.min_colony_size,
        circularity_threshold=config.circularity_threshold,
        visualize_colonies=config.visualize_colonies
    )

    print(f"Processing complete. Output size: {tetrad_size}px")

if __name__ == '__main__':
    main()
```

## Core Configuration and Usage

### Image Processing Parameters

**Tetrad Plate Processing:**
```python
from src.image_processing import process_tetrad_images

tetrad_output_size, tetrad_radius = process_tetrad_images(
    input_dir='data/tetrad',
    output_dir='results/tetrad_cropped',
    target_radius=490,              # Plate radius in pixels
    min_colony_size=50,             # Minimum colony size for tetrads
    circularity_threshold=0.7,      # Circularity filter for tetrads
    plate_to_tetrad_height_range=(40, 85),
    plate_to_tetrad_width_range=(10, 95),
    final_tetrad_height_percent=30,
    final_tetrad_width_percent=75,
    visualize_colonies=True
)
```

**Replica Plate Processing:**
```python
from src.image_processing import process_replica_images

process_replica_images(
    input_dir='data/replica',
    output_dir='results/replica_cropped',
    final_output_size_px=tetrad_output_size,  # Sync with tetrad output
    tetrad_crop_radius=tetrad_radius,          # Sync with tetrad radius
    min_colony_size=25,                        # Smaller colonies on replicas
    circularity_threshold=0.6                  # More lenient for replicas
)
```

### Data Management Classes

**Gene Metadata and Verification:**
```python
from src.utils import verificationMetadata, roundConfig

# Load verification metadata
verification_meta = verificationMetadata()

# Access gene information
print(verification_meta.num2gene[123])  # Get gene name by number
print(verification_meta.verification_genes.head())  # View verification data

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

### Table Organization and PDF Generation

**Table Processing Configuration:**
```python
from src.table_organizer import TableConfig, process_all_rounds
from src.pdf_generator import PDFGeneratorConfig, generate_round_pdfs

# Table organization
table_config = TableConfig(
    base_path=Path("/path/to/cropped/images"),
    output_path=Path("../results")
)

# Process all rounds
process_all_rounds(table_config)

# PDF generation
pdf_config = PDFGeneratorConfig(
    output_path=Path("../results/merged_pdfs"),
    include_quality_metrics=True
)

# Generate PDFs for specific rounds
generate_round_pdfs(pdf_config, rounds=["1st_round", "2nd_round"])
```

## Development Principles and Standards

### Research Code Philosophy

This project follows **research code principles** rather than production engineering:

- **Simplicity over Rigidity**: Prioritize experimental flexibility over rigid architecture
- **Direct Parameter Passing**: Avoid complex configuration objects, use direct parameters
- **Easy Modification**: Parameters should be easily configurable for experimental optimization
- **Modern Python Practices**: Use dataclasses, pathlib, type hints for clarity
- **Clear Purpose**: Each function serves one clear purpose with minimal abstraction

### Type Hint Guidelines

Use modern Python type hints consistently:
```python
# Preferred style (PEP 585+)
def process_data(
    data: dict[str, int],      # Not Dict[str, int]
    items: list[str],          # Not List[str]
    enabled: bool = True
) -> Optional[str]:           # Not Optional[str] for return types
    return result if condition else None
```

### Code Organization Standards

- **Import Order**: Standard library → Third-party → Project-specific modules
- **Path Handling**: Use `pathlib.Path` for all file system operations
- **Error Handling**: Comprehensive error handling with graceful degradation
- **Documentation**: One-line function documentation with usage examples
- **Configuration**: Use `@dataclass` for simple configuration management

### Template Usage

**ALWAYS use `TEMPLATE.py`** as the starting point for new scripts:
- Standardized structure with sections separated by `# %%`
- Modern Python import organization
- Dataclass-based configuration
- Clear separation of concerns

## Environment Troubleshooting

### Common Issues and Solutions

1. **Module Import Errors**:
   ```bash
   # Ensure running from project root directory
   pwd  # Should show: /data/c/yangyusheng_optimized/DIT_HAP_verification_organization

   # Verify src directory exists
   ls src/
   ```

2. **Wrong Python Environment**:
   ```bash
   # Check current Python
   which python
   # Should show: /data/a/yangyusheng/.local/share/mamba/envs/opencv/bin/python

   # Activate correct environment
   mamba activate opencv
   ```

3. **Resource File Not Found**:
   ```bash
   # Verify resource files exist
   ls resource/all_for_verification_genes_by_round.xlsx
   ls resource/Hayles_2013_OB_merged_categories_sysIDupdated.xlsx
   ls resource/gene_IDs_names_products/
   ```

4. **Missing Dependencies**:
   ```bash
   # Install from opencv environment only
   mamba activate opencv
   pip install -r requirements.txt
   ```

5. **OpenCV Installation Issues**:
   ```bash
   # Test OpenCV installation
   python -c "import cv2; print('OpenCV version:', cv2.__version__)"
   ```

### Environment Verification

**Complete Environment Test:**
```python
# Run this to verify full environment setup
import sys
sys.path.append('src')

try:
    import cv2
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    import openpyxl
    from loguru import logger
    print("✅ All dependencies available!")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
```

## Data Processing Pipeline

### Complete Workflow

1. **Image Acquisition**: Raw microscopy images stored in round-specific directories
2. **Plate Detection**: Circle detection to identify individual plates using Hough transform
3. **Colony Detection**: Adaptive thresholding and circularity analysis to find colonies
4. **Centroid Calculation**: Determine colony centroids with outlier detection for plate alignment
5. **Image Cropping**: Extract consistently sized regions around colony clusters
6. **File Organization**: Systematic renaming with gene metadata integration
7. **Table Creation**: Build comprehensive verification tables with quality metrics
8. **PDF Generation**: Create formatted reports and documentation

### Key Algorithms

- **Hough Circle Detection**: Robust plate identification with parameter optimization
- **Adaptive Thresholding**: Colony detection optimized for varying contrast conditions
- **Centroid Alignment**: Outlier detection ensures consistent plate positioning across rounds
- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization for improved detection
- **Synchronized Processing**: Tetrad and replica plates processed with coordinated dimensions

## Critical Requirements and Constraints

### Environment Requirements

- **Python 3.8+** in the "opencv" mamba environment (MANDATORY)
- **OpenCV 4.x** for image processing operations
- **Sufficient Memory**: Large image batches require adequate RAM
- **Disk Space**: Cropped images and processed data require significant storage

### Data Requirements

- **Resource Files**: Must be present in `resource/` directory for gene metadata
- **Directory Structure**: Specific folder organization expected for raw and processed data
- **Image Formats**: Supports standard microscopy image formats (PNG, JPG, TIFF)
- **Naming Conventions**: Systematic filenames required for proper metadata extraction

### Processing Constraints

- **Parameter Tuning**: Colony detection parameters may need adjustment for different experimental conditions
- **Quality Validation**: Image quality assessment integrated into processing pipeline
- **Batch Processing**: Memory-efficient processing required for large datasets
- **Reproducibility**: Consistent parameters and random seeds for scientific reproducibility

# ========================= KeyWord =========================

## Important Development Notes

- **Environment is Critical**: Always verify you're in the "opencv" environment before any development
- **Run from Project Root**: All scripts expect to be executed from the repository root directory
- **Resource Dependencies**: Gene metadata files in `resource/` directory are required for most operations
- **Research-Focused**: Code designed for experimental flexibility, not production rigidity
- **Parameter Sensitivity**: Image processing parameters are experiment-dependent and may require tuning
- **Template Adherence**: Always use `TEMPLATE.py` as the foundation for new scripts
- **Modern Python**: Leverage dataclasses, pathlib, type hints for clean, maintainable code