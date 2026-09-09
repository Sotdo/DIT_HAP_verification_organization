# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DIT-HAP (Deletion-Induced Transcriptional Homeostasis and Asynchronous Progression) verification pipeline for automated analysis of fission yeast (*Schizosaccharomyces pombe*) colony microscopy images. The system uses OpenCV-based computer vision to detect colonies, crop plates, perform genotype analysis, and organize verification data for high-throughput genetic experiments.

## Core Architecture

### Module Structure

#### 1. **Image Processing Module** ([src/tetrad_crop.py](src/tetrad_crop.py))
OpenCV-based plate and colony detection pipeline:
- **Plate Detection**: Hough circle transform with bilateral filtering and CLAHE enhancement
- **Colony Detection**: Adaptive thresholding with morphological operations
- **Quality Filtering**: Circularity, solidity, and size-based filtering
- **Key Functions**:
  - `find_circle_plates()`: Detect circular plates using Hough transform
  - `process_tetrad_images()`: Process tetrad plates with time-course data
  - `process_time_course_tetrad_images()`: Batch processing with metadata integration

#### 2. **Genotype Analysis Module** ([src/tetrad_genotype.py](src/tetrad_genotype.py))
Advanced colony genotyping and grid analysis:
- **Grid Detection**: 4×12 colony grid identification with spatial validation
- **Centroid Alignment**: Hungarian algorithm for optimal centroid matching
- **Time-Series Analysis**: Multi-day growth tracking (3d, 4d, 5d, 6d)
- **Marker Integration**: HYG/NAT/LEU/ADE marker plate validation
- **Key Functions**:
  - `genotyping_pipeline()`: Complete genotype analysis workflow
  - `detect_colonies_from_grayscale()`: Multi-method colony detection
  - `identify_grid_structure()`: Grid topology reconstruction

#### 3. **Data Management Module** ([src/utils.py](src/utils.py))
Metadata and configuration management:
- **Classes**:
  - `verificationMetadata`: Gene metadata integration from PomBase
  - `roundConfig`: Round-based folder structure configuration
- **Features**:
  - Gene ID to name mapping
  - Essentiality and phenotype annotation
  - Automatic folder creation and path management

#### 4. **File Organization Module** ([src/rename_functions.py](src/rename_functions.py))
Systematic file renaming with gene metadata:
- Naming format: `{gene_num}_{gene_name}_{timepoint}_{replicate}_{date}.cropped.png`
- PomBase systematic ID integration
- Batch renaming with validation

#### 5. **Table Generation Module** ([src/table_organizer.py](src/table_organizer.py))
Quality assessment and verification table creation:
- Multi-format export (CSV, Excel, JSON)
- Cropping quality metrics
- Round-wise summarization
- Metadata integration

#### 6. **PDF Report Module** ([src/pdf_generator.py](src/pdf_generator.py))
Formatted PDF documentation generation:
- Multi-page plate visualization
- Metadata annotation
- Round-based report compilation

### Processing Pipeline

```
Raw Microscopy Images
    ↓
Plate Detection (Hough Circle Transform)
    ↓
Colony Detection (Adaptive Thresholding + Morphology)
    ↓
Quality Filtering (Circularity, Solidity, Size)
    ↓
Centroid Alignment & Grid Reconstruction
    ↓
Image Cropping & Standardization
    ↓
Genotype Analysis (Time-Series + Markers)
    ↓
Metadata Integration (Gene Names, Phenotypes)
    ↓
Output Generation (Tables, PDFs, Visualizations)
```

## Essential Commands

### Environment Setup

```bash
# Required: Activate opencv environment
mamba activate opencv
# Verify Python: /data/a/yangyusheng/.local/share/mamba/envs/opencv/bin/python

# Install dependencies
pip install -r requirements.txt

# Test environment
python -c "import cv2, numpy, pandas, tqdm, openpyxl, loguru, skimage, matplotlib, seaborn; print('✅ Environment OK')"
```

### Main Workflows

```bash
# 1. Image Processing: Crop tetrad and replica plates
python scripts/batch_crop_image.py

# 2. File Organization: Rename with gene metadata
python scripts/rename_image_names.py

# 3. Genotype Analysis: Batch genotyping with time-series
python scripts/batch_genotyping.py

# 4. Report Generation: Create tables and PDFs
python scripts/organize_tables.py

# 5. Visualization: Colony size analysis
python scripts/example_colony_size_visualization.py
```

### Development Workflow

```bash
# Always run from repository root
cd /data/c/yangyusheng_optimized/DIT_HAP_verification_organization

# Use %% section separators for Jupyter-style code cells
# Follow existing script structure in scripts/ directory
```

## Key Configuration

### Image Processing Parameters

```python
@dataclass
class ImageProcessingConfig:
    # Plate parameters
    target_radius: int = 515  # Plate radius in pixels
    
    # Cropping region
    height_range: tuple[int, int] = (47, 79)
    width_range: tuple[int, int] = (10, 90)
    
    # Colony detection
    min_colony_size: int = 5      # Tetrad: 5,  Replica: 50
    max_colony_size: int = 2000   # Tetrad: 2000, Replica: 4000
    circularity_threshold: float = 0.7  # Tetrad: 0.7, Replica: 0.45
    solidity_threshold: float = 0.9     # Tetrad: 0.9, Replica: 0.7
    
    # Adaptive thresholding
    adaptive_block_size: int = 30  # Tetrad: 30, Replica: 200 (must be odd)
    adaptive_thresh_c: int = 2
    
    # Contrast enhancement
    contrast_alpha: float = 1.0  # Tetrad: 1.0, Replica: 1.6
    
    # Output
    final_height: int = 300
    final_width: int = 750
    visualize_colonies: bool = True
```

### Genotype Analysis Parameters

```python
@dataclass
class Configuration:
    # Grid parameters
    expected_rows: int = 4
    expected_cols: int = 12
    average_x_spacing: float = 57.0  # pixels
    average_y_spacing: float = 60.0  # pixels
    spacing_tolerance: float = 8.0
    
    # Colony detection (tetrad)
    tetrad_gray_method: str = "mean"
    tetrad_gaussian_sigma: float = 0.1
    tetrad_min_area: int = 4
    tetrad_max_area: int = 2000
    tetrad_circularity_threshold: float = 0.75
    tetrad_solidity_threshold: float = 0.9
    
    # Colony detection (marker)
    hyg_gaussian_sigma: float = 1
    hyg_min_area: int = 40
    hyg_max_area: int = 7000
    hyg_circularity_threshold: float = 0.6
    hyg_solidity_threshold: float = 0.85
```

## Directory Structure

```
DIT_HAP_verification_organization/
├── src/                          # Core processing modules
│   ├── tetrad_crop.py           # Image processing and plate cropping
│   ├── tetrad_genotype.py       # Genotype analysis and grid detection
│   ├── utils.py                 # Data management and configuration
│   ├── rename_functions.py      # File renaming utilities
│   ├── table_organizer.py       # Table generation and export
│   └── pdf_generator.py         # PDF report creation
│
├── scripts/                      # Executable scripts
│   ├── batch_crop_image.py      # Main image processing workflow
│   ├── batch_genotyping.py      # Batch genotype analysis
│   ├── rename_image_names.py    # File renaming workflow
│   ├── organize_tables.py       # Table and PDF generation
│   └── example_colony_size_visualization.py  # Visualization examples
│
├── resource/                     # Required data files
│   ├── gene_IDs_names_products/
│   │   └── 20251001_gene_IDs_names_products.tsv
│   ├── all_for_verification_genes_by_round.xlsx
│   ├── Hayles_2013_OB_merged_categories_sysIDupdated.xlsx
│   └── WT2nonWT_genes_verification_summary.tsv
│
├── notebook/                     # Jupyter tutorials
│   ├── tetrad_crop_tutorial.ipynb
│   ├── colony_analysis_tutorial.ipynb
│   ├── image_processing_tutorial.ipynb
│   ├── grid_restoration_explained.ipynb
│   └── verification_summary.ipynb
│
├── results/                      # Output directory
│   ├── arc/                     # Archived results
│   ├── merged_pdfs/             # Round-wise PDF reports
│   └── *.xlsx, *.pdf, *.csv     # Tables and reports
│
├── logs/                         # Log files
└── .vscode/                      # IDE configuration
```

### External Data Paths

```
/hugedata/YushengYang/DIT_HAP_verification/
├── data/
│   ├── raw_data/DIT_HAP_deletion/     # Original microscopy images
│   │   └── {round_name}/
│   │       ├── 3d/                    # 3-day growth
│   │       ├── 4d/                    # 4-day growth
│   │       ├── 5d/                    # 5-day growth
│   │       ├── 6d/                    # 6-day growth
│   │       └── replica/               # Marker plates (HYG, NAT, LEU, ADE)
│   │
│   ├── processed_data/DIT_HAP_deletion/  # Renamed images
│   │   └── {round_name}/
│   │       └── (same structure)
│   │
│   └── cropped_images/DIT_HAP_deletion/  # Processed output
│       └── {round_name}/
│           └── (same structure)
```

## Critical Requirements

1. **Environment**: Must use "opencv" mamba environment
2. **Working Directory**: Run all scripts from repository root
3. **Resource Files**: Ensure `resource/` directory contains:
   - Gene metadata Excel files
   - PomBase systematic ID mapping
   - Essentiality and phenotype annotations
4. **Data Paths**: Hardcoded paths in `/hugedata/` - update `BatchConfig` for different environments
5. **Path Management**: All scripts use `sys.path.append(str(Path(__file__).parent.parent / "src"))` to import modules

## Development Guidelines

### Code Style

- **Type Hints**: Use modern Python (PEP 585+): `dict[str, int]` not `Dict[str, int]`
- **Configuration**: Use `@dataclass` for all configuration objects
- **Error Handling**: Use `@logger.catch` decorator for main functions
- **Logging**: Configure with `loguru` for both stdout and file output
- **Code Organization**: Use `# %%` separators for Jupyter-style code cells

### Design Principles

- **Research-Focused**: Prioritize experimental flexibility over production robustness
- **Modular Architecture**: Each module has a single, well-defined responsibility
- **Configurable Parameters**: All thresholds and settings exposed in dataclasses
- **Batch Processing**: Support for multi-round, high-throughput workflows
- **Reproducibility**: Logging and intermediate outputs for debugging

### Best Practices

1. **Import Management**: Always append `src/` to path before importing custom modules
2. **Path Objects**: Use `pathlib.Path` for all file operations
3. **Pandas Integration**: Use DataFrames for metadata and batch processing
4. **Visualization**: Create debug visualizations with `visualize_colonies=True`
5. **Parameter Tuning**: Document parameter changes for different plate types

## Common Issues & Solutions

### 1. Environment Issues
**Problem**: Wrong Python environment or missing packages
```bash
# Check environment
which python
# Should output: /data/a/yangyusheng/.local/share/mamba/envs/opencv/bin/python

# Reinstall if needed
mamba activate opencv
pip install -r requirements.txt
```

### 2. Import Errors
**Problem**: Module not found errors
```python
# Ensure this is at the top of every script
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
```

### 3. Missing Resource Files
**Problem**: Gene metadata files not found
```bash
# Check resource directory
ls -la resource/
ls -la resource/gene_IDs_names_products/
```

### 4. Hardcoded Path Issues
**Problem**: Data paths not accessible
```python
# Update paths in BatchConfig
@dataclass
class BatchConfig:
    renamed_file_folder: Path = Path("/your/custom/path/processed_data")
    output_base_folder: Path = Path("/your/custom/path/cropped_images")
```

### 5. Memory Issues
**Problem**: Out of memory with large batches
```python
# Process in smaller chunks
df = pd.read_excel(config.table_file)
# Process subset: df = df.query("gene_num < 100")
```

### 6. Colony Detection Failures
**Problem**: Colonies not detected or too many false positives
```python
# Adjust parameters in ImageProcessingConfig
tetrad_config = ImageProcessingConfig(
    adaptive_block_size=40,  # Increase for larger colonies
    min_colony_size=10,      # Lower threshold
    circularity_threshold=0.6,  # More permissive
)
```

### 7. Grid Reconstruction Issues
**Problem**: Grid structure not identified
```python
# Check spacing parameters in Configuration
spacing_tolerance=10.0,  # Increase tolerance
average_x_spacing=60.0,  # Adjust based on actual images
```

## Testing & Validation

### Quick Test
```bash
# Test image processing on single round
python scripts/batch_crop_image.py
# Check logs/batch_crop_image.log for errors

# Test genotyping on sample data
python scripts/batch_genotyping.py
# Check results/ for output files
```

### Jupyter Notebooks
Explore and debug using tutorial notebooks in `notebook/`:
- [tetrad_crop_tutorial.ipynb](notebook/tetrad_crop_tutorial.ipynb): Image processing walkthrough
- [colony_analysis_tutorial.ipynb](notebook/colony_analysis_tutorial.ipynb): Genotype analysis examples
- [grid_restoration_explained.ipynb](notebook/grid_restoration_explained.ipynb): Grid detection deep dive

## Performance Optimization

- **Parallel Processing**: Not currently implemented - consider for future optimization
- **Caching**: Intermediate results saved to avoid reprocessing
- **Batch Size**: Process rounds independently to manage memory
- **Image Resolution**: Standardized to 300×750 for consistency

## Citation & References

- **PomBase**: Gene systematic IDs and annotations (pombase.org)
- **Hayles et al. (2013)**: Deletion library essentiality data
- **OpenCV**: Computer vision algorithms (opencv.org)
- **scikit-image**: Image processing toolkit (scikit-image.org)
