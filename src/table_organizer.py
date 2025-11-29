"""
Table structure organization for DIT-HAP verification pipeline.
Creates comprehensive data tables with image metadata and biological context.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import pandas as pd
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from utils import verificationMetadata, roundConfig


# %% ------------------------------------ Configuration ------------------------------------ #
@dataclass
class TableOrganizerConfig:
    """Configuration for table organization and metadata extraction."""
    # Input/output paths
    processed_data_base_path: Path = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion")
    output_base_path: Path = Path("/hugedata/YushengYang/DIT_HAP_verification/data/table_structures/DIT_HAP_deletion")

    # Quality assessment parameters
    min_colony_count: int = 1           # Minimum colonies expected in image
    max_colony_count: int = 100         # Maximum reasonable colony count
    min_image_size: tuple[int, int] = (100, 100)  # Minimum image dimensions
    max_image_size: tuple[int, int] = (2000, 2000)  # Maximum image dimensions

    # Export formats
    export_formats: list[str] = None  # None means export to all available formats

    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["csv", "xlsx"]

        # Ensure output directories exist
        self.output_base_path.mkdir(parents=True, exist_ok=True)


# %% ------------------------------------ Quality Assessment ------------------------------------ #
@logger.catch
def assess_image_quality(image_path: Path) -> Dict[str, Union[float, int, bool]]:
    """
    Assess image quality metrics for cropped images.

    Args:
        image_path: Path to image file

    Returns:
        Dictionary containing quality metrics
    """
    try:
        import cv2
        import numpy as np

        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return {"valid": False, "error": "Could not load image"}

        height, width = img.shape[:2]

        # Calculate basic metrics
        file_size_mb = image_path.stat().st_size / (1024 * 1024)

        # Convert to grayscale for quality analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Contrast metric (standard deviation)
        contrast = float(np.std(gray))

        # Sharpness metric (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(np.var(laplacian))

        # Noise estimation (high-frequency content)
        blur_kernel = np.ones((3, 3), np.float32) / 9
        blurred = cv2.filter2D(gray, -1, blur_kernel)
        noise_level = float(np.mean(np.abs(gray.astype(float) - blurred)))

        return {
            "valid": True,
            "width": width,
            "height": height,
            "file_size_mb": file_size_mb,
            "contrast": contrast,
            "sharpness": sharpness,
            "noise_level": noise_level
        }

    except Exception as e:
        logger.error(f"Error assessing image quality for {image_path}: {e}")
        return {"valid": False, "error": str(e)}


@logger.catch
def extract_image_metadata(image_path: Path, config: TableOrganizerConfig) -> Optional[dict]:
    """
    Extract metadata from systematic image filename.

    Args:
        image_path: Path to image file with systematic naming
        config: Table organizer configuration

    Returns:
        Dictionary with extracted metadata or None if parsing fails
    """
    try:
        # Parse systematic filename: gene_num_gene_name_day_or_marker_colony_id_date.png
        stem = image_path.stem
        parts = stem.split('_')

        if len(parts) < 5:
            logger.warning(f"Unexpected filename format: {image_path}")
            return None

        gene_num = parts[0]
        gene_name = parts[1]
        day_or_marker = parts[2]
        colony_id = parts[3]
        date_str = parts[4]

        # Determine if this is a time point or replica marker
        replica_markers = ["YES", "HYG", "NAT", "LEU", "ADE"]
        is_replica = day_or_marker in replica_markers

        # Quality assessment
        quality = assess_image_quality(image_path)
        if not quality.get("valid", False):
            logger.warning(f"Skipping invalid image: {image_path}")
            return None

        # Validate quality metrics
        dimensions = (quality["width"], quality["height"])
        if dimensions < config.min_image_size or dimensions > config.max_image_size:
            logger.warning(f"Image dimensions out of range: {image_path}")
            return None

        return {
            "gene_num": gene_num,
            "gene_name": gene_name,
            "day_or_marker": day_or_marker,
            "colony_id": colony_id,
            "date": date_str,
            "is_replica": is_replica,
            "image_path": str(image_path),
            "file_size_mb": round(quality["file_size_mb"], 3),
            "image_width": quality["width"],
            "image_height": quality["height"],
            "contrast": round(quality["contrast"], 2),
            "sharpness": round(quality["sharpness"], 2),
            "noise_level": round(quality["noise_level"], 3)
        }

    except Exception as e:
        logger.error(f"Error extracting metadata from {image_path}: {e}")
        return None


@logger.catch
def create_verification_table(config: TableOrganizerConfig, round_name: str) -> pd.DataFrame:
    """
    Create comprehensive verification table for a specific round.

    Args:
        config: Table organizer configuration
        round_name: Name of the experimental round

    Returns:
        DataFrame with verification table structure
    """
    # Load verification metadata
    verification_meta = verificationMetadata()

    # Configure round paths
    round_config = roundConfig(
        round_folder_name=round_name,
        raw_data_folder_path=Path("/hugedata/YushengYang/DIT_HAP_verification/data/processed_data/DIT_HAP_deletion"),
        output_folder_path=config.processed_data_base_path
    )

    # Collect all image metadata
    all_image_data = []

    # Process each subfolder (3d, 4d, 5d, 6d, replica)
    for subfolder_name, paths in round_config.all_sub_folders.items():
        input_folder = paths["input"]

        if not input_folder.exists():
            logger.warning(f"Input folder does not exist: {input_folder}")
            continue

        # Find all PNG files
        png_files = list(input_folder.glob("*.png"))
        if not png_files:
            logger.warning(f"No PNG files found in: {input_folder}")
            continue

        logger.info(f"Processing {len(png_files)} images in {subfolder_name}")

        for png_file in png_files:
            metadata = extract_image_metadata(png_file, config)
            if metadata:
                # Add round and subfolder information
                metadata["round"] = round_name
                metadata["subfolder"] = subfolder_name
                all_image_data.append(metadata)

    if not all_image_data:
        logger.error(f"No valid image data found for round {round_name}")
        return pd.DataFrame()

    # Convert to DataFrame
    df_images = pd.DataFrame(all_image_data)

    # Create pivot table structure as specified in PROMPT.md
    # Group by gene and colony
    grouped = df_images.groupby(["gene_num", "gene_name", "colony_id", "date"])

    verification_records = []

    for (gene_num, gene_name, colony_id, date), group in grouped:
        # Get verification metadata for this gene
        gene_info = verification_meta.verification_genes[
            verification_meta.verification_genes["Num"] == int(gene_num)
        ]

        if gene_info.empty:
            logger.warning(f"No verification info found for gene {gene_num}")
            continue

        gene_row = gene_info.iloc[0]

        # Create base record
        record = {
            "gene_num": gene_num,
            "gene_name": gene_name,
            "round": round_name,
            "systematic_id": gene_row.get("SysID", ""),
            "gene_essentiality": gene_row.get("DeletionLibrary_essentiality", ""),
            "phenotype_categories": gene_row.get("Category", ""),
            "phenotype_descriptions": gene_row.get("Phenotype_description", ""),
            "colony_id": colony_id,
            "date": date_str
        }

        # Add image paths for different time points and markers
        for _, row in group.iterrows():
            day_or_marker = row["day_or_marker"]
            image_path = row["image_path"]

            # Map to appropriate column
            if day_or_marker in ["3d", "4d", "5d", "6d"]:
                record[f"{day_or_marker}_image_path"] = image_path
                record[f"{day_or_marker}_file_size_mb"] = row["file_size_mb"]
                record[f"{day_or_marker}_contrast"] = row["contrast"]
                record[f"{day_or_marker}_sharpness"] = row["sharpness"]
            elif day_or_marker in ["YES", "HYG", "NAT", "LEU", "ADE"]:
                record[f"{day_or_marker}_image_path"] = image_path
                record[f"{day_or_marker}_file_size_mb"] = row["file_size_mb"]
                record[f"{day_or_marker}_contrast"] = row["contrast"]
                record[f"{day_or_marker}_sharpness"] = row["sharpness"]

        verification_records.append(record)

    return pd.DataFrame(verification_records)


@logger.catch
def generate_quality_report(df: pd.DataFrame, config: TableOrganizerConfig, round_name: str) -> dict:
    """
    Generate quality assessment reports.

    Args:
        df: Verification table DataFrame
        config: Table organizer configuration
        round_name: Name of the experimental round

    Returns:
        Dictionary containing quality report DataFrames
    """
    reports = {}

    # Overall statistics
    total_records = len(df)
    records_with_all_timepoints = df[
        df[["3d_image_path", "4d_image_path", "5d_image_path", "6d_image_path"]].notna().all(axis=1)
    ]

    summary_stats = pd.DataFrame([
        {
            "metric": "Total gene records",
            "count": total_records,
            "percentage": 100.0
        },
        {
            "metric": "Records with all time points",
            "count": len(records_with_all_timepoints),
            "percentage": (len(records_with_all_timepoints) / total_records * 100) if total_records > 0 else 0
        },
        {
            "metric": "Records with replica data",
            "count": len(df[df[["YES_image_path", "HYG_image_path", "NAT_image_path", "LEU_image_path", "ADE_image_path"]].notna().any(axis=1)],
            "percentage": (len(df[df[["YES_image_path", "HYG_image_path", "NAT_image_path", "LEU_image_path", "ADE_image_path"]].notna().any(axis=1)]) / total_records * 100) if total_records > 0 else 0
        }
    ])

    reports["summary"] = summary_stats

    # Image quality metrics by time point
    quality_columns = []
    for timepoint in ["3d", "4d", "5d", "6d", "YES", "HYG", "NAT", "LEU", "ADE"]:
        quality_columns.extend([
            f"{timepoint}_file_size_mb",
            f"{timepoint}_contrast",
            f"{timepoint}_sharpness"
        ])

    # Calculate quality statistics
    quality_data = []
    for timepoint in ["3d", "4d", "5d", "6d", "YES", "HYG", "NAT", "LEU", "ADE"]:
        size_col = f"{timepoint}_file_size_mb"
        contrast_col = f"{timepoint}_contrast"
        sharpness_col = f"{timepoint}_sharpness"

        if size_col in df.columns:
            valid_sizes = df[size_col].dropna()
            if len(valid_sizes) > 0:
                quality_data.append({
                    "timepoint": timepoint,
                    "metric": "File Size (MB)",
                    "mean": valid_sizes.mean(),
                    "std": valid_sizes.std(),
                    "min": valid_sizes.min(),
                    "max": valid_sizes.max(),
                    "count": len(valid_sizes)
                })

        if contrast_col in df.columns:
            valid_contrast = df[contrast_col].dropna()
            if len(valid_contrast) > 0:
                quality_data.append({
                    "timepoint": timepoint,
                    "metric": "Contrast",
                    "mean": valid_contrast.mean(),
                    "std": valid_contrast.std(),
                    "min": valid_contrast.min(),
                    "max": valid_contrast.max(),
                    "count": len(valid_contrast)
                })

        if sharpness_col in df.columns:
            valid_sharpness = df[sharpness_col].dropna()
            if len(valid_sharpness) > 0:
                quality_data.append({
                    "timepoint": timepoint,
                    "metric": "Sharpness",
                    "mean": valid_sharpness.mean(),
                    "std": valid_sharpness.std(),
                    "min": valid_sharpness.min(),
                    "max": valid_sharpness.max(),
                    "count": len(valid_sharpness)
                })

    if quality_data:
        reports["quality_metrics"] = pd.DataFrame(quality_data)

    return reports


@logger.catch
def export_tables(df: pd.DataFrame, quality_reports: Dict[str, pd.DataFrame], config: TableOrganizerConfig, round_name: str):
    """
    Export verification table and quality reports in specified formats.

    Args:
        df: Verification table DataFrame
        quality_reports: Dictionary of quality report DataFrames
        config: Table organizer configuration
        round_name: Name of the experimental round
    """
    round_output_path = config.output_base_path / round_name
    round_output_path.mkdir(parents=True, exist_ok=True)

    # Export main verification table
    for format_type in config.export_formats:
        if format_type == "csv":
            df.to_csv(round_output_path / f"{round_name}_verification_table.csv", index=False)
        elif format_type == "xlsx":
            with pd.ExcelWriter(round_output_path / f"{round_name}_verification_table.xlsx", engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Verification Data', index=False)

                # Add quality reports as separate sheets
                for report_name, report_df in quality_reports.items():
                    if not report_df.empty:
                        report_df.to_excel(writer, sheet_name=f'Quality_{report_name}', index=False)

        elif format_type == "json":
            df.to_json(round_output_path / f"{round_name}_verification_table.json", orient='records', indent=2)

        logger.info(f"Exported {format_type.upper()} for round {round_name}")

    logger.success(f"All tables exported for round {round_name} to {round_output_path}")


# %% ------------------------------------ Main Functions ------------------------------------ #
@logger.catch
def process_round_tables(config: TableOrganizerConfig, round_name: str):
    """
    Process tables for a specific experimental round.

    Args:
        config: Table organizer configuration
        round_name: Name of the experimental round to process
    """
    logger.info(f"Starting table organization for round: {round_name}")

    # Create verification table
    df = create_verification_table(config, round_name)

    if df.empty:
        logger.error(f"No data to process for round {round_name}")
        return

    logger.info(f"Created verification table with {len(df)} records")

    # Generate quality reports
    quality_reports = generate_quality_report(df, config, round_name)

    # Export tables and reports
    export_tables(df, quality_reports, config, round_name)

    logger.success(f"Completed table organization for round {round_name}")


@logger.catch
def process_all_rounds(config: TableOrganizerConfig):
    """
    Process tables for all available experimental rounds.

    Args:
        config: Table organizer configuration
    """
    # Find all available rounds
    base_data_path = Path("/hugedata/YushengYang/DIT_HAP_verification/data/processed_data/DIT_HAP_deletion")

    if not base_data_path.exists():
        logger.error(f"Base data path does not exist: {base_data_path}")
        return

    round_folders = [d for d in base_data_path.iterdir() if d.is_dir()]
    round_names = [f.name for f in round_folders if f.name.startswith("round_")]

    if not round_names:
        logger.error("No round directories found")
        return

    logger.info(f"Found {len(round_names)} rounds to process: {round_names}")

    for round_name in sorted(round_names):
        try:
            process_round_tables(config, round_name)
        except Exception as e:
            logger.error(f"Error processing round {round_name}: {e}")
            continue

    logger.success("Completed table organization for all rounds")


if __name__ == "__main__":
    # Example usage
    config = TableOrganizerConfig()
    process_all_rounds(config)