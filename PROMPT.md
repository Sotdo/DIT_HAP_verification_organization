# PROMPT.md

DIT-HAP Verification Pipeline Data Flow Analysis and Function Request Framework

---

## 🎯 **GENERAL GOAL**

The DIT-HAP verification pipeline processes fission yeast (*Schizosaccharomyces pombe*) tetrad dissection colony images to systematically organize genetic verification data. The pipeline converts raw microscopy images into standardized, gene-annotated cropped images suitable for quantitative analysis and high-throughput screening.

### **Primary Objectives**
1. **Rename and Organize**: Systematically rename and organize images based on deleted genes, time points, strain identifiers, and experimental conditions. Implement consistent naming conventions for traceability and analysis
2. **Crop and Standardize**: Crop images to focus on yeast colonies, ensuring uniform dimensions and centering across different time points and experimental setups
3. **Organize images to table structure**: Structure output images into directories reflecting experimental rounds, time points, and gene deletions for easy access and analysis
4. **Metadata Integration**: Integrate gene verification metadata from Excel files to annotate images with relevant biological information
5. **PDF Generation**: Compile processed images into PDF documents for visual inspection and record-keeping
6. **Manual Verification Support**: Facilitate manual verification of gene deletions by providing clear, organized image data alongside relevant metadata
7. **Image compression**: Optimize image file sizes without compromising quality for efficient storage and sharing

---

## **Code Style and Architecture Guidelines**
Refer to the [CLAUDE.md](CLAUDE.md) file for detailed code style guidelines, architectural principles, and best practices to ensure consistency and maintainability across the codebase.

## 📂 **INPUT FILES DESCRIPTION**

### **Raw Image Data**
```
Raw Data Structure:
├── 1st_round/
│   ├── 3d/          # Day 3 tetrad images
│   ├── 4d/          # Day 4 tetrad images
│   ├── 5d/          # Day 5 tetrad images
│   ├── 6d/          # Day 6 tetrad images
│   └── replica/     # Replica plate images
└── 2nd_round/       # Additional experimental rounds
    └── ...
```

**Image Specifications:**
- **Format**: TIFF, JPG files from microscopy
- **Content**: Multiple plates per image (tetrad: 1 plate, replica: 5 plates)
- **Naming**: Original microscopy system naming (date-based, experimental)
- **Resolution**: High-resolution images suitable for colony detection
- **Plates**: Circular petri dishes with yeast colonies

### **Metadata Files**

1. **`resource/all_for_verification_genes_by_round.xlsx`**
   - **Content**: Gene verification assignments by experimental round
   - **Columns**: Round, Num (gene number), SysID (systematic ID)
   - **Purpose**: Maps experimental plate numbers to specific gene deletions

2. **`resource/gene_IDs_names_products/20251001_gene_IDs_names_products.tsv`**
   - **Content**: PomBase gene database information
   - **Columns**: Systematic ID, gene name, gene product description
   - **Purpose**: Provides human-readable gene names for systematic IDs

3. **`resource/Hayles_2013_OB_merged_categories_sysIDupdated.xlsx`**
   - **Content**: Gene essentiality and phenotype data
   - **Columns**: Systematic ID, dispensability, phenotype categories, descriptions
   - **Purpose**: Additional biological context for verification experiments

---

## 🔧 **Finished PIPELINE TASKS**

### **Task 1: Rename raw images and organize directories**

**Input**: Raw microscopy images with original naming conventions
**Output**: Systematically named images organized by gene, time point, and colony ID

**Steps**:
1. **Load Verification Metadata**:
   - Parse `resource/all_for_verification_genes_by_round.xlsx` to get gene assignments
   - Load `resource/gene_IDs_names_products/20251001_gene_IDs_names_products.tsv` for gene names
   - Merge with `resource/Hayles_2013_OB_merged_categories_sysIDupdated.xlsx` for essentiality data

2. **Parse Original Filenames**:
   - Extract date, gene number, colony ID from original microscopy naming
   - Determine experimental round from directory structure
   - Identify time point (3d, 4d, 5d, 6d) or replica marker

3. **Generate New Naming Convention**:
   - Format: `{gene_num}_{gene_name}_{day_or_marker}_{colony_id}_{date}`
   - Example: `123_adh1_3d_colonyA_20251128.tif`
   - Replica plates: `{gene_num}_{gene_name}_{marker}_{colony_id}_{date}`
   - Example: `123_adh1_HYG_colonyA_20251128.tif`

4. **Directory Organization**:
   - Create round-specific output directories
   - Organize by time point and marker
   - Maintain colony grouping for each gene

5. **File Processing**:
   - Copy images with new systematic names
   - Preserve original image quality and metadata
   - Handle naming conflicts and edge cases

**Implementation**: `scripts/rename_image_names.py` → `src/rename_functions.py.rename_images_per_round()`


### **Task 2: Crop images to focus on yeast colonies**

**Input**: Systematically named raw images from Task 1
**Output**: Standardized cropped images centered on yeast colonies

**Steps**:
1. **Plate Detection (Hough Circle Transform)**:
   - Convert images to grayscale
   - Apply bilateral filter for edge-preserving smoothing
   - Use CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhancement
   - Detect circular plates using Hough circle transform
   - Parameters: min_radius=400px, max_radius=600px, adaptive parameter adjustment

2. **Plate Validation and Cropping**:
   - Validate plate detection (expect 1 plate for tetrad, 5 for replica)
   - Extract circular regions using detected coordinates
   - Handle edge cases: no detection, multiple detections, partial plates
   - Generate visualization images for debugging if enabled

3. **Colony Detection within Plates**:
   - Focus on central region (height: 45-80%, width: 10-90% of plate)
   - Apply contrast enhancement (alpha=1.0 for tetrads, 1.5 for replicas)
   - Use adaptive thresholding with configurable block sizes
   - Perform morphological operations to reduce noise

4. **Centroid Calculation**:
   - Filter colonies by minimum area and circularity thresholds
   - Calculate weighted centroid of all detected colonies
   - Generate visualization showing detected colonies and centroid

5. **Centroid Synchronization**:
   - Collect centroids from all plates in same experimental batch
   - Calculate average centroid position
   - Identify outliers (>75px deviation) and adjust to average
   - Ensure consistent positioning across time points and conditions

6. **Final Cropping to Standard Dimensions**:
   - Crop around adjusted centroid (75% width × 30% height of plate)
   - Maintain consistent pixel dimensions across all images
   - Save as PNG format with systematic naming
   - Generate optional visualization images

**Configuration**: `src/image_processing.py.ImageProcessingConfig` class
- Target radius: 490px (auto-calculated if not specified)
- Colony size thresholds: 50px (tetrads), 25px (replicas)
- Circularity thresholds: 0.7 (tetrads), 0.6 (replicas)
- Adaptive block sizes: 30 (tetrads), 120 (replicas)

**Implementation**: `scripts/batch_crop_image.py` → `src/image_processing.py.process_tetrad_images()`



## 🔧 **Ongoing and Future PIPELINE TASKS**

### **Task 3: Table structure organization of cropped images and PDF generation**

**Input**: Cropped images from Task 2 with systematic naming
**Output**: Organized table structures of images with gene metadata and phenotype context

**Steps**:
1. **Image Metadata Extraction**:
  - Parse systematic filenames to extract gene_num, gene_name, day_or_marker, colony_id, date
  - Link to gene verification metadata: round, SysID, gene_essentiality, phenotype categories, descriptions

2. **Table output Structure**:
  - The output table should has the following columns:
    - gene_num
    - round
    - systematic_id
    - gene_name
    - gene_essentiality
    - phenotype_categories
    - phenotype_descriptions
    - colony_id
    - date
    - 3d_image_path
    - 4d_image_path
    - 5d_image_path
    - 6d_image_path
    - YES_image_path
    - HYG_image_path
    - NAT_image_path
    - LEU_image_path
    - ADE_image_path
  - Some of the images may be missing part of the images, so the corresponding image_path column can be empty.
  - The table is saved as a excel file in results/ with the naming convention: `all_rounds_verification_summary.xlsx`

3. **PDF Generation**:
  - For each round, generate a PDF document compiling all cropped images
  - Each page of the PDF should have a grid layout showing images for each gene deletion
    - The first column shows the gene_num, gene_name, gene_essentiality, phenotype_categories, and colony_id, date.
    - The second column shows the 3d images.
    - The third column shows the 4d images.
    - The fourth column shows the 5d images.
    - The fifth column shows the 6d images.
    - The sixth to eleventh columns show the replica images (YES, HYG, NAT, LEU, ADE).
    - There are no more than 4 rows in each page.
  - Ensure high-resolution images in the PDF for clarity
  - Output these PDFs to results/merged_pdfs/ directory with naming convention: `round_{round_number}_verification_summary.pdf`

**Configuration**:
- Use the current `roundConfg` dataclass to manage round-specific settings
- Define PDF layout parameters (page size, margins, image sizes)
- Handle missing images gracefully in the PDF layout

**Potential Implementation Files**:
- `src/table_organizer.py` - Table structure creation and management
- `src/pdf_generator.py` - PDF compilation and formatting
- `scripts/organize_tables.py` - Main execution script for table organization and PDF generation