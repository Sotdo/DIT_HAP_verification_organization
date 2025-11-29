"""
Table structure organization for DIT-HAP verification pipeline.
Creates comprehensive data tables with image metadata and biological context.
"""

import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
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
    output_base_path: Path = Path("../results")

    # Quality assessment parameters
    min_colony_count: int = 1           # Minimum colonies expected in image
    max_colony_count: int = 100         # Maximum reasonable colony count
    min_image_size: tuple[int, int] = (100, 100)  # Minimum image dimensions
    max_image_size: tuple[int, int] = (2000, 2000)  # Maximum image dimensions

    # Export formats
    export_formats: list[str] | None = None  # None means export to all available formats

    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["csv", "xlsx"]

        # Ensure output directories exist
        self.output_base_path.mkdir(parents=True, exist_ok=True)


# %% ------------------------------------ Quality Assessment ------------------------------------ #
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
        stem = image_path.stem.removesuffix(".cropped")
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

        return {
            "gene_num": gene_num,
            "gene_name": gene_name,
            "day_or_marker": day_or_marker,
            "colony_id": colony_id,
            "date": date_str,
            "is_replica": is_replica,
            "image_path": str(image_path),
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
        raw_data_folder_path=config.processed_data_base_path,
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
                all_image_data.append(metadata)

    if not all_image_data:
        logger.error(f"No valid image data found for round {round_name}")
        return pd.DataFrame()

    # Convert to DataFrame
    df_images = pd.DataFrame(all_image_data)

    # Convert gene_num to int for proper sorting, but keep original for lookup
    df_images['gene_num_int'] = df_images['gene_num'].astype(int)

    # Create pivot table structure as specified in PROMPT.md
    # Sort by gene_num as integer first
    df_images_sorted = df_images.sort_values('gene_num_int')

    # Group by gene and colony
    grouped = df_images_sorted.groupby(["gene_num", "gene_name", "colony_id", "date"])

    verification_records = []

    for (gene_num, gene_name, colony_id, date_str), group in grouped:
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
            elif day_or_marker in ["YES", "HYG", "NAT", "LEU", "ADE"]:
                record[f"{day_or_marker}_image_path"] = image_path

        verification_records.append(record)

    df = pd.DataFrame(verification_records)

    # Sort final DataFrame by gene_num as integer to ensure proper ordering
    df['gene_num_int'] = df['gene_num'].astype(int)
    df = df.sort_values('gene_num_int').drop(columns=['gene_num_int'])

    return df


@logger.catch
def export_single_excel(all_rounds_data: dict[str, pd.DataFrame], config: TableOrganizerConfig):
    """
    Export all rounds to a single Excel file with multiple sheets.

    Args:
        all_rounds_data: Dictionary with round names as keys and DataFrames as values
        config: Table organizer configuration
    """
    output_file = config.output_base_path / "all_rounds_verification_summary.xlsx"

    # Create summary statistics
    summary_data = []
    for round_name, df in all_rounds_data.items():
        if not df.empty:
            total_records = len(df)
            complete_time_series = len(df.dropna(subset=['3d_image_path', '4d_image_path', '5d_image_path', '6d_image_path'], how='all'))
            complete_replicas = len(df.dropna(subset=['YES_image_path', 'HYG_image_path', 'NAT_image_path', 'LEU_image_path', 'ADE_image_path'], how='all'))

            summary_data.append({
                'Round': round_name,
                'Total Records': total_records,
                'Complete Time Series': complete_time_series,
                'Complete Replicas': complete_replicas,
                'Time Series Completion %': (complete_time_series / total_records * 100) if total_records > 0 else 0,
                'Replica Completion %': (complete_replicas / total_records * 100) if total_records > 0 else 0
            })

    summary_df = pd.DataFrame(summary_data)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Add summary sheet first
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Add each round as a separate sheet
        for round_name, df in all_rounds_data.items():
            if not df.empty:
                # Sheet names have a 31 character limit in Excel
                sheet_name = round_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                logger.info(f"Added sheet '{sheet_name}' with {len(df)} records")

    logger.success(f"All rounds exported to single Excel file: {output_file}")
    return output_file


@logger.catch
def export_tables(df: pd.DataFrame, config: TableOrganizerConfig, round_name: str, all_rounds_data: dict[str, pd.DataFrame] | None = None):
    """
    Export verification table and quality reports in specified formats.

    Args:
        df: Verification table DataFrame
        config: Table organizer configuration
        round_name: Name of the experimental round
        all_rounds_data: Dictionary of all rounds data for single Excel export (optional)
    """
    # If we have all rounds data, export to single Excel file
    if all_rounds_data is not None:
        export_single_excel(all_rounds_data, config)
        return

    # Legacy individual round export
    round_output_path = config.output_base_path / round_name
    round_output_path.mkdir(parents=True, exist_ok=True)

    # Export main verification table
    for format_type in config.export_formats:
        if format_type == "csv":
            df.to_csv(round_output_path / f"{round_name}_verification_table.csv", index=False)
        elif format_type == "xlsx":
            with pd.ExcelWriter(round_output_path / f"{round_name}_verification_table.xlsx", engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Verification Data', index=False)

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
    logger.info("")
    logger.info("-" * 40)
    logger.info(f"Starting table organization for round: {round_name}")

    # Create verification table
    df = create_verification_table(config, round_name)

    if df.empty:
        logger.error(f"No data to process for round {round_name}")
        return

    logger.info(f"Created verification table with {len(df)} records")


    # Export tables and reports
    export_tables(df, config, round_name)

    logger.success(f"Completed table organization for round {round_name}")


@logger.catch
def process_all_rounds(config: TableOrganizerConfig):
    """
    Process tables for all available experimental rounds and export to single Excel file.

    Args:
        config: Table organizer configuration
    """
    # Find all available rounds
    base_data_path = Path("/hugedata/YushengYang/DIT_HAP_verification/data/processed_data/DIT_HAP_deletion")

    if not base_data_path.exists():
        logger.error(f"Base data path does not exist: {base_data_path}")
        return

    round_folders = [d for d in base_data_path.iterdir() if d.is_dir()]
    round_names = [f.name for f in round_folders]

    if not round_names:
        logger.error("No round directories found")
        return

    def extract_round_number(round_name: str) -> int:
        """Extract numeric value from round name for proper sorting."""
        try:
            # Handle various round name formats: "1st_round", "2nd_round", "3rd_round", "4th_round", etc.
            match = re.search(r'(\d+)', round_name)
            return int(match.group(1)) if match else 0
        except Exception:
            return 0

    # Sort rounds by numeric value instead of alphabetical
    sorted_round_names = sorted(round_names, key=extract_round_number)

    logger.info(f"Found {len(round_names)} rounds to process:")
    for round_name in sorted_round_names:
        logger.info(f" - {round_name}")

    # Collect all rounds data for single Excel export
    all_rounds_data = {}
    total_records = 0

    for round_name in sorted_round_names:
        try:
            # Create verification table for this round
            df = create_verification_table(config, round_name)

            if not df.empty:
                all_rounds_data[round_name] = df
                total_records += len(df)
                logger.info(f"Created verification table for {round_name} with {len(df)} records")
            else:
                logger.warning(f"No data found for round {round_name}")

        except Exception as e:
            logger.error(f"Error processing round {round_name}: {e}")
            continue

    # Export all rounds to single Excel file if we have data
    if all_rounds_data:
        logger.info(f"Exporting {total_records} total records from {len(all_rounds_data)} rounds to single Excel file")
        export_single_excel(all_rounds_data, config)
    else:
        logger.error("No valid data found for any rounds")

    logger.success("Completed table organization for all rounds")


if __name__ == "__main__":
    # Example usage
    config = TableOrganizerConfig()
    process_all_rounds(config)