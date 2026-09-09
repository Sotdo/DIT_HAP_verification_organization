# DIT-HAP Verification Organization

Automated image processing and genotype analysis pipeline for DIT-HAP (Deletion-Induced Transcriptional Homeostasis and Asynchronous Progression) verification experiments in fission yeast (*Schizosaccharomyces pombe*).

## Overview

This pipeline processes high-throughput yeast colony microscopy images using computer vision techniques to:
- **Detect and analyze** yeast colonies with OpenCV-based algorithms
- **Crop and align** plate images with consistent geometry
- **Genotype tetrad dissections** with automated grid detection
- **Integrate gene metadata** from PomBase and deletion libraries
- **Generate systematic reports** with tables and visualizations

## Features

### Image Processing
- **Plate Detection**: Hough circle transform with CLAHE enhancement
- **Colony Detection**: Adaptive thresholding with morphological filtering
- **Quality Control**: Multi-criteria filtering (circularity, solidity, size)
- **Time-Series Support**: Multi-day growth tracking (3d, 4d, 5d, 6d)
- **Dual Plate Types**: Optimized parameters for tetrad and replica plates

### Genotype Analysis
- **Grid Reconstruction**: 4×12 colony grid identification and validation
- **Centroid Alignment**: Hungarian algorithm for optimal matching
- **Marker Integration**: HYG/NAT/LEU/ADE selection marker analysis
- **Growth Quantification**: Time-course colony size measurements
- **Phenotype Annotation**: Automated essentiality classification

### Data Management
- **Metadata Integration**: PomBase systematic IDs and gene names
- **Batch Processing**: Multi-round experiment organization
- **Multi-Format Export**: CSV, Excel, JSON, and PDF outputs
- **Reproducible Logging**: Detailed logs for debugging and validation

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd DIT_HAP_verification_organization

# Activate required conda/mamba environment
mamba activate opencv

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import cv2, numpy, pandas, tqdm, openpyxl, loguru, skimage, matplotlib, seaborn; print('✅ Environment Ready')"
```

### 2. Prepare Data

Ensure your data follows this structure:
```
/hugedata/YushengYang/DIT_HAP_verification/data/raw_data/DIT_HAP_deletion/
└── {round_name}/
    ├── 3d/          # 3-day growth images
    ├── 4d/          # 4-day growth images
    ├── 5d/          # 5-day growth images
    ├── 6d/          # 6-day growth images
    └── replica/     # Marker plates (HYG, NAT, LEU, ADE)
```

### 3. Run Complete Pipeline

```bash
# Step 1: Process and crop all images
python scripts/batch_crop_image.py
# Output: Cropped images in /hugedata/.../cropped_images/

# Step 2: Rename files with gene metadata
python scripts/rename_image_names.py
# Output: Files renamed with gene names and systematic IDs

# Step 3: Perform genotype analysis
python scripts/batch_genotyping.py
# Output: results/batch_genotyping_results.pdf and .xlsx

# Step 4: Generate verification tables
python scripts/organize_tables.py
# Output: results/all_combined_all_rounds_*.xlsx
```

## Project Structure

```
DIT_HAP_verification_organization/
│
├── src/                          # Core Processing Modules
│   ├── tetrad_crop.py           # Image processing and plate cropping
│   ├── tetrad_genotype.py       # Genotype analysis and grid detection
│   ├── utils.py                 # Data management and configuration
│   ├── rename_functions.py      # File organization utilities
│   ├── table_organizer.py       # Table generation and export
│   └── pdf_generator.py         # PDF report creation
│
├── scripts/                      # Executable Workflows
│   ├── batch_crop_image.py      # Main image processing pipeline
│   ├── batch_genotyping.py      # Genotype analysis workflow
│   ├── rename_image_names.py    # File renaming workflow
│   ├── organize_tables.py       # Report generation workflow
│   └── example_colony_size_visualization.py  # Visualization examples
│
├── notebook/                     # Jupyter Tutorials
│   ├── tetrad_crop_tutorial.ipynb
│   ├── colony_analysis_tutorial.ipynb
│   ├── image_processing_tutorial.ipynb
│   ├── grid_restoration_explained.ipynb
│   └── verification_summary.ipynb
│
├── resource/                     # Required Data Files
│   ├── gene_IDs_names_products/
│   ├── all_for_verification_genes_by_round.xlsx
│   ├── Hayles_2013_OB_merged_categories_sysIDupdated.xlsx
│   └── WT2nonWT_genes_verification_summary.tsv
│
├── results/                      # Output Directory
│   ├── arc/                     # Archived analysis results
│   └── merged_pdfs/             # Round-wise PDF reports
│
├── logs/                         # Execution Logs
├── CLAUDE.md                     # Development Guidelines
├── README.md                     # This File
└── requirements.txt              # Python Dependencies
```

## Detailed Workflows

### Image Processing Pipeline

**Objective**: Detect plates and colonies, standardize image geometry

```bash
python scripts/batch_crop_image.py
```

**Process**:
1. Load raw microscopy images
2. Detect circular plates using Hough transform
3. Detect colonies with adaptive thresholding
4. Filter by morphological criteria
5. Crop and standardize to 300×750 pixels
6. Save cropped images with metadata

**Configuration** ([src/tetrad_crop.py](src/tetrad_crop.py)):
```python
ImageProcessingConfig(
    target_radius=515,           # Plate radius in pixels
    min_colony_size=5,           # Minimum colony area
    circularity_threshold=0.7,   # Colony shape filter
    adaptive_block_size=30,      # Thresholding window
    visualize_colonies=True      # Save debug visualizations
)
```

### Genotype Analysis Pipeline

**Objective**: Identify grid structure, track growth, validate genotypes

```bash
python scripts/batch_genotyping.py
```

**Process**:
1. Load cropped tetrad and marker images
2. Detect colonies and extract centroids
3. Reconstruct 4×12 grid topology
4. Match centroids across time points
5. Quantify colony sizes over time
6. Validate with marker plates
7. Export results to Excel and PDF

**Output Files**:
- `results/batch_genotyping_results.xlsx`: Detailed genotype table
- `results/batch_genotyping_results.pdf`: Visual verification plots
- `logs/batch_genotyping.log`: Execution log

### File Organization Workflow

**Objective**: Rename files with systematic gene identifiers

```bash
python scripts/rename_image_names.py
```

**Naming Convention**:
```
{gene_num}_{gene_name}_{timepoint}_{replicate}_{date}.cropped.png

Example:
242_tyw3_3d_#1_202510.cropped.png
  ├── 242: Gene number in verification set
  ├── tyw3: Gene name from PomBase
  ├── 3d: 3-day growth timepoint
  ├── #1: Replicate number
  └── 202510: Date (YYYYMM)
```

### Report Generation Workflow

**Objective**: Create comprehensive verification tables and PDFs

```bash
python scripts/organize_tables.py
```

**Output Files**:
- `results/all_combined_all_rounds_crop_summary.xlsx`: Complete processing summary
- `results/all_combined_all_rounds_renamed_summary.xlsx`: Renamed file index
- `results/merged_pdfs/round_{round_name}_verification_summary.pdf`: Per-round reports

## Key Configuration

### Plate-Specific Parameters

**Tetrad Plates** (small colonies, high density):
```python
ImageProcessingConfig(
    min_colony_size=5,
    max_colony_size=2000,
    circularity_threshold=0.7,
    solidity_threshold=0.9,
    adaptive_block_size=30,
    contrast_alpha=1.0
)
```

**Replica Plates** (large colonies, selection markers):
```python
ImageProcessingConfig(
    min_colony_size=50,
    max_colony_size=4000,
    circularity_threshold=0.45,
    solidity_threshold=0.7,
    adaptive_block_size=200,
    contrast_alpha=1.6
)
```

### Customizing Data Paths

Edit `BatchConfig` in [scripts/batch_crop_image.py](scripts/batch_crop_image.py):
```python
@dataclass
class BatchConfig:
    renamed_file_folder: Path = Path("/your/path/processed_data/DIT_HAP_deletion")
    output_base_folder: Path = Path("/your/path/cropped_images/DIT_HAP_deletion")
    table_file: Path = Path("results/all_combined_all_rounds_renamed_summary.xlsx")
```

## Dependencies

### Required Packages
- **Python 3.8+**
- **OpenCV 4.x** (`opencv-python`): Computer vision algorithms
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **scikit-image**: Image processing toolkit
- **scikit-learn**: Machine learning utilities (Linear regression, Hungarian algorithm)
- **Matplotlib & Seaborn**: Visualization
- **tqdm**: Progress bars
- **loguru**: Enhanced logging
- **openpyxl**: Excel file operations

### Installation
```bash
pip install tqdm loguru numpy pandas openpyxl opencv-python scikit-image matplotlib seaborn scikit-learn
```

Or use the provided requirements file:
```bash
pip install -r requirements.txt
```

## Advanced Usage

### Processing Specific Rounds

Edit [scripts/batch_crop_image.py](scripts/batch_crop_image.py):
```python
# Process only specific rounds
not_processed_rounds = ["22th_round", "23th_round"]
df = df[~df['round'].isin(not_processed_rounds)]
```

### Custom Genotyping Subsets

Edit [scripts/batch_genotyping.py](scripts/batch_genotyping.py):
```python
# Filter by gene categories
df = df[df['DeletionLibrary_essentiality'] == 'Essential']

# Filter by specific genes
df = df[df['gene_num'].isin([242, 302, 165])]
```

### Parameter Tuning for Colony Detection

If colonies are not detected properly:

```python
# For smaller colonies
config = ImageProcessingConfig(
    min_colony_size=3,              # Lower threshold
    gaussian_sigma=0.05,            # Sharper edges
    adaptive_block_size=20          # Smaller window
)

# For noisy images
config = ImageProcessingConfig(
    gaussian_sigma=1.5,             # More smoothing
    circularity_threshold=0.5,      # More permissive
    morphology_disk_size=5          # Larger kernel
)
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Import errors** | Ensure `sys.path.append(str(Path(__file__).parent.parent / "src"))` at script top |
| **Missing metadata** | Check `resource/` directory contains required Excel/TSV files |
| **Wrong environment** | Verify with `which python` → should be in `opencv` environment |
| **No colonies detected** | Lower `min_colony_size` or adjust `adaptive_block_size` |
| **Too many false positives** | Increase `circularity_threshold` and `solidity_threshold` |
| **Grid not reconstructed** | Increase `spacing_tolerance` in `Configuration` |
| **Memory issues** | Process smaller batches by filtering DataFrame |

### Debug Visualizations

Enable visualization to inspect intermediate results:
```python
config = ImageProcessingConfig(visualize_colonies=True)
```

Check output images for:
- Detected plate circles (blue circles)
- Colony boundaries (colored contours)
- Centroid positions (red dots)
- Grid structure (connecting lines)

### Log Analysis

Check log files in `logs/` directory:
```bash
tail -f logs/batch_crop_image.log
tail -f logs/batch_genotyping.log
```

## Performance

### Typical Processing Times
- **Image Processing**: ~2-5 seconds per image (depends on colony count)
- **Genotype Analysis**: ~10-20 seconds per tetrad set (4 timepoints + marker)
- **Complete Round** (300 genes × 3 replicates × 5 images): ~2-4 hours

### Optimization Tips
- Process rounds in parallel on different machines
- Use SSD storage for image I/O
- Increase `adaptive_block_size` for faster processing (lower accuracy)
- Disable `visualize_colonies` for production runs

## Tutorials & Examples

Explore Jupyter notebooks in `notebook/` for interactive tutorials:

1. **[tetrad_crop_tutorial.ipynb](notebook/tetrad_crop_tutorial.ipynb)**: Step-by-step image processing walkthrough
2. **[colony_analysis_tutorial.ipynb](notebook/colony_analysis_tutorial.ipynb)**: Genotype analysis examples
3. **[image_processing_tutorial.ipynb](notebook/image_processing_tutorial.ipynb)**: Algorithm deep dive
4. **[grid_restoration_explained.ipynb](notebook/grid_restoration_explained.ipynb)**: Grid detection methodology
5. **[verification_summary.ipynb](notebook/verification_summary.ipynb)**: Results analysis and visualization

## Citation & References

### Databases
- **PomBase**: *S. pombe* genome database ([pombase.org](https://www.pombase.org))
- **Hayles et al. (2013)**: Fission yeast deletion library phenotypic data

### Software
- **OpenCV**: Open source computer vision library ([opencv.org](https://opencv.org))
- **scikit-image**: Image processing in Python ([scikit-image.org](https://scikit-image.org))
- **Pandas**: Data analysis toolkit ([pandas.pydata.org](https://pandas.pydata.org))

### Algorithms
- **Hough Circle Transform**: Plate detection
- **Adaptive Thresholding**: Colony segmentation
- **Hungarian Algorithm** (Linear Sum Assignment): Centroid matching
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization): Image enhancement

## Contributing

For development guidelines, code style, and best practices, see [CLAUDE.md](CLAUDE.md).

### Code Organization
- Place reusable functions in `src/` modules
- Create executable workflows in `scripts/`
- Use `@dataclass` for configuration
- Follow existing naming conventions
- Add logging with `@logger.catch` decorator
- Document with docstrings and type hints

## License

[Add license information here]

## Contact

[Add contact information here]

---

**For detailed development documentation, see [CLAUDE.md](CLAUDE.md)**
