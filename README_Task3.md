# Task 3: Table Structure Organization and PDF Generation

## Overview

This task processes the cropped images from Task 2 to create comprehensive verification tables and generate formatted PDF documents for visual inspection and record-keeping.

## Files Created

### Core Modules

1. **`src/table_organizer.py`** - Table structure creation and management
   - **Purpose**: Organizes cropped images into comprehensive verification tables with gene metadata
   - **Key Functions**:
     - `assess_image_quality()` - Quality metrics for image validation
     - `extract_image_metadata()` - Parse systematic filenames to extract experimental data
     - `create_verification_table()` - Build comprehensive verification tables
     - `generate_quality_report()` - Create quality assessment reports
     - `export_tables()` - Export data in multiple formats (CSV, Excel, JSON)
     - `process_round_tables()` - Process tables for a specific experimental round
     - `process_all_rounds()` - Process all available experimental rounds

   - **Configuration**: `TableOrganizerConfig` dataclass
     ```python
     @dataclass
     class TableOrganizerConfig:
         processed_data_base_path: Path  # Path to cropped images
         output_base_path: Path         # Output for table structures
         min_colony_count: int = 1     # Quality thresholds
         max_colony_count: int = 100
         export_formats: list[str] = ["csv", "xlsx"]
     ```

2. **`src/pdf_generator.py`** - PDF compilation and formatting
   - **Purpose**: Creates formatted PDF documents compiling verification images with metadata
   - **Key Functions**:
     - `validate_image()` - Check image quality and dimensions for PDF inclusion
     - `get_image_for_gene_colony()` - Retrieve specific image paths from verification table
     - `create_gene_info_row()` - Format gene metadata for PDF display
     - `create_pdf_page()` - Create single PDF page with gene information and images
     - `generate_verification_pdf()` - Generate complete verification PDF for a round
     - `generate_round_pdfs()` - Process specified round or all rounds
     - `create_sample_table_for_testing()` - Create sample data for testing

   - **Configuration**: `PDFGeneratorConfig` dataclass
     ```python
     @dataclass
     class PDFGeneratorConfig:
         processed_data_base_path: Path  # Path to cropped images
         output_base_path: Path         # Output for PDFs
         page_size: str = "A4"          # Page layout
         image_width: float = 1.2         # Image dimensions (inches)
         image_height: float = 0.9
         max_rows_per_page: int = 4        # Layout control
     ```

3. **`scripts/organize_tables.py`** - Main execution script
   - **Purpose**: Orchestrates table organization and PDF generation
   - **Key Functions**:
     - `process_tables()` - Execute table organization
     - `process_pdfs()` - Execute PDF generation
     - `create_sample_data()` - Create test data for development
     - `verify_prerequisites()` - Check required files and directories
     - `setup_logging()` - Initialize logging configuration
     - `main()` - Main execution workflow
     - `print_usage()` - Display usage instructions

   - **Configuration**: `OrganizeTablesConfig` dataclass
     ```python
     @dataclass
     class OrganizeTablesConfig:
         processed_data_base_path: Path  # Path to cropped images
         output_base_path: Path         # Output for organized data
         process_tables: bool = True     # Enable/disable processing
         generate_pdfs: bool = True      # Enable/disable PDF generation
         create_samples: bool = False     # Create sample data
         rounds_to_process: list[str] = None  # Specific rounds (None = all)
     ```

## Data Flow

### Input Requirements
- **Cropped Images**: From Task 2 with systematic naming
- **Metadata**: From resource files (gene verification, PomBase data, essentiality data)
- **Directory Structure**: Organized by round and time point

### Processing Steps

1. **Table Organization**:
   - Extract metadata from systematic filenames
   - Validate image quality (dimensions, file size, contrast, sharpness)
   - Create comprehensive verification tables with gene metadata
   - Generate quality assessment reports
   - Export data in multiple formats

2. **PDF Generation**:
   - Load verification tables from step 1
   - Validate images for PDF inclusion
   - Create formatted pages with gene information and image grids
   - Compile into high-resolution PDFs
   - Organize by experimental round

### Output Structure

```
DIT_HAP_verification_organization/
├── data/
│   └── table_structures/DIT_HAP_deletion/
│       ├── 1st_round_verification_table.csv
│       ├── 1st_round_verification_table.xlsx
│       ├── 2nd_round_verification_table.csv
│       └── ...
└── data/
    └── merged_pdfs/DIT_HAP_deletion/
        ├── round_1st_round_verification_summary.pdf
        ├── round_2nd_round_verification_summary.pdf
        └── ...
```

### Table Structure

Each verification table contains the following columns as specified in PROMPT.md:

- **gene_num** - Gene deletion number
- **round** - Experimental round identifier
- **systematic_id** - PomBase systematic ID
- **gene_name** - Human-readable gene name
- **gene_essentiality** - Essentiality status from Hayles et al.
- **phenotype_categories** - Phenotype classification
- **phenotype_descriptions** - Detailed phenotype descriptions
- **colony_id** - Individual colony identifier
- **date** - Experimental date
- **3d/4d/5d/6d_image_path** - Time point image paths
- **YES/HYG/NAT/LEU/ADE_image_path** - Replica plate image paths
- **Quality metrics** - File size, contrast, sharpness for each image

### PDF Layout

Each PDF page contains:
- **First column**: Gene information (gene_num, gene_name, essentiality, phenotype, colony_id, date)
- **Columns 2-5**: Time point images (3d, 4d, 5d, 6d)
- **Columns 6-10**: Replica plate images (YES, HYG, NAT, LEU, ADE)
- **Layout**: Maximum 4 rows per page, high-resolution images
- **Output**: `round_{round_number}_verification_summary.pdf`

## Usage

### Basic Usage
```bash
# Process all rounds (default)
python scripts/organize_tables.py

# Process specific rounds
# Modify OrganizeTablesConfig in scripts/organize_tables.py
config.rounds_to_process = ["1st_round", "2nd_round"]

# Create sample data for testing
config.create_samples = True
```

### Configuration
Edit the `OrganizeTablesConfig` class in `scripts/organize_tables.py`:
- **Base paths**: Update to your actual data locations
- **Processing options**: Enable/disable table processing and PDF generation
- **Output formats**: Choose CSV, Excel, or JSON export
- **PDF layout**: Adjust page size, image dimensions, and layout parameters

### Quality Control
The pipeline includes comprehensive quality assessment:
- **Image dimensions**: Validate minimum and maximum sizes
- **File sizes**: Check for corrupted or oversized files
- **Contrast metrics**: Ensure sufficient image quality
- **Sharpness assessment**: Detect blurred or out-of-focus images
- **Missing data handling**: Graceful handling of missing time points or replicas

## Error Handling

The implementation includes robust error handling:
- **Missing files**: Skip and log warnings for missing images
- **Invalid data**: Skip and log errors for corrupted files
- **Resource not found**: Clear error messages for missing resource files
- **Permission issues**: Handle directory creation and file access issues
- **Memory issues**: Graceful handling of large datasets

## Testing

### Sample Data Creation
```python
# Create sample verification table
from src.table_organizer import create_sample_table_for_testing
config = TableOrganizerConfig()
sample_file = create_sample_table_for_testing(config, "1st_round")

# Create sample PDF data
from src.pdf_generator import create_sample_table_for_testing
sample_pdf = create_sample_table_for_testing(config, "1st_round")
```

### Quality Validation
- Log files track all processing steps and errors
- Quality reports highlight problematic images
- Summary statistics show data completeness
- Visual PDFs enable manual verification

## Integration with Existing Pipeline

### Follows Task 2
- Uses systematic naming from Task 2: `{gene_num}_{gene_name}_{day_or_marker}_{colony_id}_{date}`
- Reads cropped images from Task 2 output directories
- Maintains consistency with existing `ImageProcessingConfig` parameters

### Resource Integration
- Uses same resource files as existing pipeline
- Integrates with `verificationMetadata` class from `src/utils.py`
- Maintains consistency with `roundConfig` structure

### Code Style Compliance
- Follows research code principles from CLAUDE.md
- Uses dataclasses for configuration management
- Includes comprehensive logging with loguru
- Implements type hints for all functions
- Uses pathlib for file system operations
- Follows existing import and module structure

## Troubleshooting

### Common Issues

1. **Missing cropped images**
   ```
   Error: Input folder does not exist: /path/to/cropped/images
   ```
   **Solution**: Ensure Task 2 has been completed successfully

2. **Resource files not found**
   ```
   Error: Required resource file not found: resource/all_for_verification_genes_by_round.xlsx
   ```
   **Solution**: Verify all resource files are present in resource/ directory

3. **Permission errors**
   ```
   Error: Permission denied when creating output directory
   ```
   **Solution**: Check write permissions for output directories

4. **Insufficient disk space**
   ```
   Error: No space left on device
   ```
   **Solution**: Ensure sufficient disk space for PDFs and exported tables

### Debug Mode
Enable verbose logging in `setup_logging()`:
```python
logger.add(
    log_file,
    level="DEBUG",  # Change to DEBUG for detailed logging
    rotation="10 MB",
    retention="7 days"
)
```

This implementation provides a complete solution for Task 3, following the existing codebase patterns and maintaining consistency with the research-focused development approach outlined in CLAUDE.md.