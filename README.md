# DIT-HAP Verification Organization

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive, research-focused image processing and data management pipeline for DIT-HAP (Deletion-Induced Transcriptional Homeostasis and Asynchronous Progression) verification experiments in fission yeast (*Schizosaccharomyces pombe*).

## 🎯 Overview

This project provides automated tools for processing high-throughput yeast colony images, detecting colonies with advanced computer vision techniques, and organizing verification data for systematic genetic analysis. The pipeline is designed with **research flexibility** in mind, allowing easy parameter modification for experimental optimization while maintaining reproducible workflows.

### Key Capabilities

- **🔬 Advanced Colony Detection**: OpenCV-based detection with adaptive thresholding, CLAHE enhancement, and circularity analysis
- **🎯 Precise Plate Processing**: Hough circle detection with optimized parameters for accurate plate identification and cropping
- **📁 Systematic File Organization**: Gene-aware renaming with PomBase metadata integration and round-based categorization
- **⚡ High-Throughput Processing**: Memory-efficient batch processing with comprehensive progress tracking
- **📊 Data Integration**: Comprehensive table organization and PDF generation for verification reports
- **🔧 Research-Focused Design**: Easy parameter modification and template-based development for experimental flexibility

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (required for modern type hints and dataclasses)
- **Mamba/Conda** environment manager (recommended for dependency management)
- **Git** for version control
- Access to raw DIT-HAP verification microscopy data

### Environment Setup (Required)

> **⚠️ Critical**: The project requires the specific `opencv` mamba environment for proper functionality.

```bash
# Clone the repository
git clone <repository-url>
cd DIT_HAP_verification_organization

# Create and activate the required opencv environment
mamba create -n opencv python=3.9 -y
mamba activate opencv

# Verify environment (IMPORTANT)
which python
# Expected: /data/a/yangyusheng/.local/share/mamba/envs/opencv/bin/python

# Install dependencies from the correct environment
pip install -r requirements.txt

# Test complete environment setup
python -c "import cv2, numpy, pandas, tqdm, openpyxl, loguru; print('✅ Environment ready!')"
```

### Alternative Environment Methods

If mamba activation fails, use these alternatives:

```bash
# Method 2: Direct PATH export
export PATH="/data/a/yangyusheng/.local/share/mamba/envs/opencv/bin:$PATH"

# Method 3: Full Python path for commands
/data/a/yangyusheng/.local/share/mamba/envs/opencv/bin/python <script>
```

### First Run Verification

```bash
# Verify project structure and dependencies
python -c "
import sys
sys.path.append('src')
try:
    import cv2, numpy as np, pandas as pd, tqdm, openpyxl, loguru
    print('✅ All dependencies available!')
    print(f'OpenCV: {cv2.__version__}')
    print(f'NumPy: {np.__version__}')
    print(f'Pandas: {pd.__version__}')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    print('Run: pip install -r requirements.txt')
"
```

## 📁 Project Structure

```
DIT_HAP_verification_organization/
├── 📂 src/                           # Core processing modules
│   ├── 🔬 image_processing.py        # OpenCV-based colony detection and plate processing
│   ├── 📊 utils.py                   # Data classes, configuration, and metadata management
│   ├── 🏷️  rename_functions.py        # File renaming utilities with gene metadata
│   ├── 📋 table_organizer.py         # Table structure organization and quality assessment
│   ├── 📄 pdf_generator.py            # PDF document generation for verification reports
│   └── 🐍 __init__.py                 # Package initialization
├── 📂 scripts/                        # Execution scripts and workflows
│   ├── 🖼️  batch_crop_image.py       # Enhanced batch image cropping interface
│   ├── 🏷️  rename_image_names.py     # Main renaming workflow
│   ├── 📊 organize_tables.py          # Table organization and PDF generation workflow
│   └── 🔬 TEMPLATE.py                 # Template script for new development
├── 📂 resource/                       # Data and reference files (REQUIRED)
│   ├── 📋 Hayles_2013_OB_merged_categories_sysIDupdated.xlsx
│   ├── 📋 all_for_verification_genes_by_round.xlsx
│   └── 📂 gene_IDs_names_products/   # Gene metadata and systematic ID mappings
├── 📂 results/                        # Output directory for processed data
│   └── 📂 merged_pdfs/               # Generated PDF reports
├── 📋 requirements.txt                 # Python dependencies
├── 📖 CLAUDE.md                       # Development guidelines for Claude Code integration
├── 📖 README.md                       # This file
├── 📖 PROMPT.md                       # Function request framework
├── 📖 README_Task3.md                 # Task-specific documentation
└── 📄 LICENSE                         # License information
```

## 🔧 Core Components

### 1. Image Processing Pipeline (`src/image_processing.py`)

Advanced OpenCV-based pipeline with template-style architecture and comprehensive type hints.

#### Key Features
- **Adaptive Colony Detection**: Configurable thresholding with circularity analysis
- **Hough Circle Detection**: Optimized plate identification with parameter tuning
- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization
- **Centroid Alignment**: Outlier detection for consistent plate positioning
- **Synchronized Processing**: Coordinated tetrad and replica plate dimensions
- **Progress Tracking**: tqdm integration for batch processing monitoring

#### Usage Examples

```python
from src.image_processing import process_tetrad_images, process_replica_images

# Tetrad plate processing (primary workflow)
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
    visualize_colonies=True         # Show detection results
)

# Replica plate processing (synchronized with tetrads)
process_replica_images(
    input_dir='data/replica',
    output_dir='results/replica_cropped',
    final_output_size_px=tetrad_output_size,  # Sync with tetrad output
    tetrad_crop_radius=tetrad_radius,          # Sync with tetrad radius
    min_colony_size=25,                        # Smaller colonies on replicas
    circularity_threshold=0.6                  # More lenient for replicas
)
```

### 2. Data Management (`src/utils.py`)

Configuration and metadata management classes with PomBase integration.

#### Key Classes

```python
from src.utils import verificationMetadata, roundConfig

# Gene verification metadata management
verification_meta = verificationMetadata()

# Access gene information
print(verification_meta.num2gene[123])        # Get gene name by number
print(verification_meta.gene2num["adh1"])      # Get number by gene name
print(verification_meta.verification_genes.head())  # View verification data

# Round-specific configuration
round_config = roundConfig(
    round_folder_name="1st_round",
    raw_data_folder_path=Path("/path/to/raw/data"),
    output_folder_path=Path("/path/to/processed/data")
)

# Access organized folder structure
print(round_config.all_sub_folders["3d"]["input"])   # Raw 3-day images
print(round_config.all_sub_folders["5d"]["output"])  # Processed 5-day images
```

### 3. Table Organization (`src/table_organizer.py`)

Comprehensive table structure organization with quality assessment and multi-format export.

#### Features
- **Quality Assessment**: Automated evaluation of cropped images
- **Verification Tables**: Creation with comprehensive metadata integration
- **Multi-format Export**: CSV, Excel, and JSON output capabilities
- **Gene Integration**: Seamless connection with verification data

```python
from src.table_organizer import TableConfig, process_all_rounds

# Table processing configuration
table_config = TableConfig(
    base_path=Path("/path/to/cropped/images"),
    output_path=Path("../results"),
    include_quality_metrics=True,
    export_formats=["csv", "excel", "json"]
)

# Process all experimental rounds
process_all_rounds(table_config)
```

### 4. PDF Generation (`src/pdf_generator.py`)

Formatted PDF document creation for verification reports with customizable layouts.

```python
from src.pdf_generator import PDFGeneratorConfig, generate_round_pdfs

# PDF generation configuration
pdf_config = PDFGeneratorConfig(
    output_path=Path("../results/merged_pdfs"),
    include_quality_metrics=True,
    custom_layout="research"
)

# Generate PDFs for specific rounds
generate_round_pdfs(pdf_config, rounds=["1st_round", "2nd_round"])
```

## 🔄 Complete Data Processing Workflow

### 1. Image Acquisition & Organization
- Raw microscopy images stored in round-specific directories
- Systematic file naming with experimental metadata
- Time-based organization (3d, 5d, etc.) for different growth conditions

### 2. Plate Detection & Analysis
- Hough circle transform for accurate plate identification
- Parameter optimization for different experimental conditions
- Quality assessment of plate detection results

### 3. Colony Detection Processing
- Adaptive thresholding optimized for varying contrast conditions
- Circularity analysis with configurable thresholds
- CLAHE enhancement for improved colony visibility

### 4. Centroid Calculation & Alignment
- Colony centroid determination with outlier detection
- Plate alignment algorithms for consistent positioning
- Synchronization between tetrad and replica plates

### 5. Image Cropping & Enhancement
- Consistent region extraction around colony clusters
- Coordinated dimensions between tetrad and replica plates
- Memory-efficient processing for large image batches

### 6. File Organization & Metadata Integration
- Systematic renaming with gene metadata from PomBase
- Round-based categorization and folder structure
- Quality metrics integration and assessment

### 7. Table Creation & Documentation
- Comprehensive verification tables with quality metrics
- Multi-format export capabilities (CSV, Excel, JSON)
- PDF generation for research documentation

## 🛠️ Usage Examples

### Basic Processing Workflow

```bash
# Activate the required environment
mamba activate opencv

# Process tetrad images (primary workflow)
python scripts/batch_crop_image.py

# Rename images with systematic conventions
python scripts/rename_image_names.py

# Create verification tables and generate PDF reports
python scripts/organize_tables.py
```

### Custom Research Script

Always start from `TEMPLATE.py` for new development:

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

from image_processing import process_tetrad_images
from utils import roundConfig

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

### Batch Processing with Custom Parameters

```python
from src.image_processing import ImageProcessingConfig

# Custom configuration for challenging experimental conditions
custom_config = ImageProcessingConfig(
    min_colony_size=30,          # Smaller colonies
    circularity_threshold=0.5,   # More lenient filtering
    adaptive_block_size=50,      # Larger adaptive threshold blocks
    contrast_alpha=1.2,          # Enhanced contrast
    clahe_clip_limit=3.0         # Increased CLAHE enhancement
)
```

## 📋 Dependencies

### Core Requirements
- **OpenCV (`opencv-python`)**: Computer vision and image processing
- **NumPy**: Numerical operations and array handling
- **Pandas**: Data manipulation and analysis
- **tqdm**: Progress bars for batch processing
- **openpyxl**: Excel file reading and writing
- **loguru**: Advanced logging with colorization

### Installation

```bash
# Install from the correct opencv environment
mamba activate opencv
pip install -r requirements.txt
```

### Individual Component Installation

```bash
# Core image processing
pip install opencv-python numpy

# Data management
pip install pandas openpyxl tqdm

# Logging and utilities
pip install loguru pathlib
```

## 🔍 Troubleshooting

### Environment Issues

#### 1. Module Import Errors
```bash
# Ensure running from project root directory
pwd
# Expected: /path/to/DIT_HAP_verification_organization

# Verify src directory exists
ls src/

# Check Python environment
which python
# Expected: /data/a/yangyusheng/.local/share/mamba/envs/opencv/bin/python
```

#### 2. Missing Dependencies
```bash
# Test environment setup
python -c "
import sys
sys.path.append('src')
try:
    import cv2
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    import openpyxl
    from loguru import logger
    print('✅ All dependencies available!')
    print(f'OpenCV version: {cv2.__version__}')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    print('Run: pip install -r requirements.txt')
"
```

#### 3. Resource File Not Found
```bash
# Verify required resource files exist
ls resource/all_for_verification_genes_by_round.xlsx
ls resource/Hayles_2013_OB_merged_categories_sysIDupdated.xlsx
ls -la resource/gene_IDs_names_products/
```

### Image Processing Issues

#### 1. Colony Detection Problems
```python
# Adjust parameters for different experimental conditions
process_tetrad_images(
    # ... other parameters ...
    min_colony_size=30,          # Decrease for smaller colonies
    circularity_threshold=0.5,   # Decrease for more lenient filtering
    adaptive_block_size=51,      # Increase for better adaptation
    visualize_colonies=True      # Enable to debug detection
)
```

#### 2. Plate Detection Issues
```python
# Tune Hough circle parameters
process_tetrad_images(
    # ... other parameters ...
    hough_dp=1,                  # Accumulator resolution
    hough_min_dist=100,          # Minimum distance between circles
    hough_param1=50,             # Upper threshold for Canny edge detector
    hough_param2=30,             # Threshold for circle center detection
    visualize_colonies=True      # Enable to debug detection
)
```

#### 3. Memory Issues with Large Batches
```python
# Process in smaller batches
import os
from pathlib import Path

input_dir = Path("data/large_dataset")
batch_size = 50

all_files = list(input_dir.glob("*.jpg"))
for i in range(0, len(all_files), batch_size):
    batch_files = all_files[i:i+batch_size]
    # Process batch
    print(f"Processing batch {i//batch_size + 1}/{(len(all_files)-1)//batch_size + 1}")
```

### Performance Optimization

#### 1. Faster Processing
```python
# Optimize for speed (may reduce accuracy)
process_tetrad_images(
    # ... other parameters ...
    adaptive_block_size=25,      # Smaller blocks for faster processing
    clahe_enabled=False,         # Disable CLAHE for speed
    gaussian_blur=3,             # Smaller blur kernel
    skip_quality_assessment=True # Skip quality checks
)
```

#### 2. Higher Quality Processing
```python
# Optimize for accuracy (slower processing)
process_tetrad_images(
    # ... other parameters ...
    clahe_clip_limit=3.0,        # Increased contrast enhancement
    adaptive_block_size=51,      # Larger blocks for better adaptation
    gaussian_blur=7,             # Larger blur kernel for noise reduction
    quality_check=True           # Enable quality assessment
)
```

## 📖 Development Guidelines

### Research-Code Philosophy

This project follows **research code principles** rather than production engineering:

- **🔧 Flexibility First**: Prioritize experimental adaptability over rigid architecture
- **⚡ Direct Parameters**: Use direct parameter passing instead of complex configuration objects
- **🎯 Easy Modification**: Make experimental parameters easily accessible and tunable
- **📚 Modern Python**: Leverage dataclasses, pathlib, and type hints for clarity
- **📝 Clear Purpose**: Each function serves one clear purpose with minimal abstraction

### Type Hint Guidelines

Use modern Python type hints consistently (PEP 585+):

```python
# Preferred modern style
def process_data(
    data: dict[str, int],      # Not Dict[str, int]
    items: list[str],          # Not List[str]
    enabled: bool = True,
    config: ProcessConfig | None = None
) -> Optional[str]:           # Optional for return types only
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

```bash
# Create new research script
cp TEMPLATE.py scripts/my_research_script.py

# Modify following the template structure:
# - %% sections for organization
# - Modern import organization
# - Dataclass-based configuration
# - Clear separation of concerns
```

## 📊 File Naming Conventions

### Systematic Naming Format

```
{gene_number}_{gene_name}_{timepoint}_{colony_id}_{date}
```

**Examples:**
- `123_adh1_3d_colonyA_20251128` - Gene 123 (adh1), 3-day growth, colony A
- `456_cdc2_5d_colonyB_20251128` - Gene 456 (cdc2), 5-day growth, colony B

### Directory Structure

```
results/
├── 1st_round/
│   ├── 3d/
│   │   ├── 123_adh1_3d_colonyA_20251128.jpg
│   │   └── 456_cdc2_3d_colonyA_20251128.jpg
│   └── 5d/
│       ├── 123_adh1_5d_colonyB_20251128.jpg
│       └── 456_cdc2_5d_colonyB_20251128.jpg
└── 2nd_round/
    ├── 3d/
    └── 5d/
```

## 🔬 Experimental Parameters

### Recommended Starting Parameters

| Parameter | Tetrad Plates | Replica Plates | Notes |
|-----------|---------------|----------------|-------|
| `target_radius` | 490px | - | Physical plate radius |
| `min_colony_size` | 50 | 25 | Smaller for replicas |
| `circularity_threshold` | 0.7 | 0.6 | More lenient for replicas |
| `adaptive_block_size` | 51 | 51 | Odd numbers only |
| `clahe_clip_limit` | 2.0 | 2.0 | Contrast enhancement |

### Parameter Tuning Guide

```python
# For poor contrast images
process_tetrad_images(
    clahe_clip_limit=3.0,        # Increase contrast enhancement
    adaptive_block_size=51,      # Larger adaptation regions
    contrast_alpha=1.2           # Boost overall contrast
)

# For very small colonies
process_tetrad_images(
    min_colony_size=20,          # Smaller minimum size
    circularity_threshold=0.4,   # More lenient shape filter
    gaussian_blur=3              # Reduce noise filtering
)

# For high-noise images
process_tetrad_images(
    gaussian_blur=9,             # Increase noise reduction
    adaptive_block_size=71,      # Larger adaptation regions
    clahe_enabled=True           # Keep contrast enhancement
)
```

## 📈 Performance Considerations

### Memory Management

- **Batch Processing**: Process images in batches to manage memory usage
- **Efficient Arrays**: Use NumPy arrays for image data manipulation
- **Memory Profiling**: Monitor memory usage with large datasets
- **Cleanup**: Explicit cleanup of large temporary objects

### Processing Speed

- **Vectorized Operations**: Use NumPy vectorization where possible
- **Avoid Loops**: Minimize Python loops in image processing
- **Optimized Algorithms**: Use OpenCV's optimized functions
- **Parallel Processing**: Consider multiprocessing for independent operations

## 🤝 Contributing

### Development Workflow

1. **Environment Setup**: Always use the `opencv` environment
2. **Template Usage**: Start new features from `TEMPLATE.py`
3. **Testing**: Test with sample data before full processing
4. **Documentation**: Update function documentation and examples
5. **Type Hints**: Use modern Python type hints consistently

### Code Review Guidelines

- **Research Focus**: Prioritize experimental flexibility
- **Clear Interfaces**: Simple, direct function interfaces
- **Documentation**: One-line docs with usage examples
- **Error Handling**: Graceful degradation for research scenarios

## 📄 License

[Add your license information here]

## 📚 Citation

[Add citation information here]

## 📞 Contact

[Add contact information here]

---

**Note**: This project is designed for research use with a focus on experimental flexibility and ease of modification. For production deployment, consider additional error handling, logging, and performance optimizations.