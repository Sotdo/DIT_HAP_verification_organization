"""
Table organization for DIT-HAP verification pipeline.
Creates comprehensive data tables with image metadata and biological context.
"""

import sys
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from utils import verificationMetadata, roundConfig


# %% ------------------------------------ Configuration ------------------------------------ #
@dataclass
class TableConfig:
    """Simple configuration for table organization."""
    image_base_path: Path
    table_output_path: Path
    image_formats: list[str] = field(default_factory=lambda: ['tif', 'png', 'jpg'])
    image_column_order: list[str] = field(default_factory=lambda: ['3d', '4d', '5d', '6d', 'YES', 'HYG', 'NAT', 'LEU', 'ADE'])


# %% ------------------------------------ Helper Functions ------------------------------------ #
def parse_image_filename(image_path: Path) -> Optional[dict]:
    """Extract metadata from systematic filename.

    Args:
        image_path: Path to image file with systematic naming

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
        replica_markers = ["YES", "HYG", "NAT", "LEU", "ADE", "YHZAY2A"]
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
        logger.error(f"Error parsing filename {image_path}: {e}")
        return None


def collect_images_for_round(round_name: str, config: TableConfig) -> list[dict]:
    """Find and collect all images for a specific round.

    Args:
        round_name: Name of the experimental round
        config: Table configuration

    Returns:
        List of image metadata dictionaries
    """
    # Configure round paths
    round_config = roundConfig(
        round_folder_name=round_name,
        raw_data_folder_path=config.image_base_path
    )

    all_image_data = []

    # Process each subfolder (3d, 4d, 5d, 6d, replica)
    for subfolder_name, paths in round_config.all_sub_folders.items():
        input_folder = paths["input"]

        if not input_folder.exists():
            logger.warning(f"Input folder does not exist: {input_folder}")
            continue

        # Find all image files
        all_image_files = []
        for image_format in config.image_formats:
            image_files = list(input_folder.glob(f"*.{image_format}"))
            all_image_files.extend(image_files)
        
        if not all_image_files:
            logger.warning(f"No image files found in: {input_folder}")
            continue

        for image_file in all_image_files:
            metadata = parse_image_filename(image_file)
            if metadata:
                # Add round and subfolder information
                metadata["round"] = round_name
                all_image_data.append(metadata)

    return all_image_data


def extract_round_number(round_name: str) -> int:
    """Extract numeric value from round name for proper sorting.

    Args:
        round_name: Round name (e.g., "1st_round", "2nd_round")

    Returns:
        Integer value for sorting
    """
    try:
        match = re.search(r'(\d+)', round_name)
        return int(match.group(1)) if match else 0
    except Exception:
        return 0


def extract_colony_number(colony_id: str) -> int:
    """Extract numeric value from colony ID for sorting.

    Args:
        colony_id: Colony identifier (e.g., "A", "B", "1", "2")

    Returns:
        Integer value for sorting
    """
    try:
        if colony_id.isdigit():
            return int(colony_id)
        elif colony_id.isalpha():
            # Convert A=1, B=2, C=3, etc.
            return ord(colony_id.upper()) - ord('A') + 1
        else:
            # Extract numbers from mixed IDs like "colonyA1"
            numbers = re.findall(r'\d+', colony_id)
            return int(numbers[0]) if numbers else 0
    except Exception:
        return 0


def sort_dataframe_by_gene_and_colony(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame by gene number and colony ID.

    Args:
        df: Input DataFrame

    Returns:
        Sorted DataFrame
    """
    # Convert gene_num to int for proper sorting
    df['gene_num_int'] = df['gene_num'].astype(int)
    df['colony_num'] = df['colony_id'].apply(extract_colony_number)

    # Sort by gene_num first, then by colony_num
    df_sorted = df.sort_values(['gene_num_int', 'colony_num'])

    # Drop sorting columns
    return df_sorted.drop(columns=['gene_num_int', 'colony_num'])


def build_verification_dataframe(image_data: list[dict], verification_meta: verificationMetadata, round_name: str, image_column_order: list[str]) -> pd.DataFrame:
    """Create verification DataFrame from image metadata.

    Args:
        image_data: List of image metadata dictionaries
        verification_meta: Verification metadata object
        round_name: Name of the experimental round

    Returns:
        Verification DataFrame
    """
    if not image_data:
        logger.error(f"No valid image data found for round {round_name}")
        return pd.DataFrame()

    # Convert to DataFrame
    df_images = pd.DataFrame(image_data)

    # Group by gene and colony
    grouped = df_images.sort_values('gene_num').groupby(["gene_num", "gene_name", "colony_id", "date"])

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
            "date": date_str,
            # New empty columns for user to fill
            "verification_phenotype": "",
            "verification_essentiality": "",
            "Kept": ""
        }

        # Initialize all image path columns to empty strings first
        for column in image_column_order:
            record[f"{column}_image_path"] = ""

        # Add image paths for different time points and markers
        for _, row in group.iterrows():
            day_or_marker = row["day_or_marker"]
            image_path = row["image_path"]

            # Map to appropriate column
            if day_or_marker in image_column_order:
                record[f"{day_or_marker}_image_path"] = image_path

        verification_records.append(record)

    df = pd.DataFrame(verification_records)
    return sort_dataframe_by_gene_and_colony(df)


def export_verification_table(df: pd.DataFrame, round_name: str, config: TableConfig):
    """Export verification table in specified formats.

    Args:
        df: Verification DataFrame
        round_name: Name of the experimental round
        config: Table configuration
    """
    config.table_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export main verification table
    if config.table_output_path.suffix == ".csv":
        df.to_csv(config.table_output_path, index=False)
    elif config.table_output_path.suffix == ".xlsx":
        df.to_excel(config.table_output_path.parent, index=False)

    logger.info(f"Exported {config.table_output_path.suffix.upper()} for round {round_name}")


def export_all_rounds_summary(all_rounds_data: dict[str, pd.DataFrame], config: TableConfig):
    """Export all rounds to single Excel file with multiple sheets.

    Args:
        all_rounds_data: Dictionary with round names as keys and DataFrames as values
        config: Table configuration
    """

    # Create summary statistics
    summary_data = []
    for round_name, df in all_rounds_data.items():
        if not df.empty:
            total_records = len(df)
            complete_images = df.dropna(subset=[f"{col}_image_path" for col in config.image_column_order], how='any')

            summary_data.append({
                'Round': round_name,
                'Total Records': total_records,
                'Complete Image Sets': len(complete_images),
                'Completion Rate (%)': (len(complete_images) / total_records * 100) if total_records > 0 else 0
            })

    summary_df = pd.DataFrame(summary_data)

    # Create concatenated DataFrame for all rounds
    all_rounds_concatenated = []
    for round_name, df in all_rounds_data.items():
        if not df.empty:
            # Add round name to each record if not already present
            df_copy = df.copy()
            if 'round' not in df_copy.columns:
                df_copy['round'] = round_name
            all_rounds_concatenated.append(df_copy)

    concatenated_df = pd.concat(all_rounds_concatenated, ignore_index=True)

    # Sort concatenated DataFrame by gene_num_int and colony_num for consistency
    if not concatenated_df.empty:
        concatenated_df = sort_dataframe_by_gene_and_colony(concatenated_df)

    # Export individual round sheets file
    with pd.ExcelWriter(config.table_output_path, engine='openpyxl') as writer:
        # Add summary sheet first
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Add each round as a separate sheet
        for round_name, df in all_rounds_data.items():
            if not df.empty:
                # Sheet names have a 31 character limit in Excel
                sheet_name = round_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                logger.info(f"Added sheet '{sheet_name}' with {len(df)} records")

    logger.success(f"Individual round sheets exported to: {config.table_output_path}")

    # Export independent file with all rounds combined
    if not concatenated_df.empty:
        combined_output_file = config.table_output_path.parent / f"all_combined_{config.table_output_path.name}"
        concatenated_df.to_excel(combined_output_file, sheet_name='All Rounds Combined', index=False)
        logger.success(f"Combined rounds exported to independent Excel file: {combined_output_file}")
        logger.info(f"Combined file contains {len(concatenated_df)} total records from {len(all_rounds_data)} rounds")

    return config.table_output_path


# %% ------------------------------------ Main Functions ------------------------------------ #
def create_verification_table(round_name: str, config: TableConfig) -> pd.DataFrame:
    """Create verification table for a specific round.

    Args:
        round_name: Name of the experimental round
        config: Table configuration

    Returns:
        Verification DataFrame
    """
    logger.info(f"Starting table organization for round: {round_name}")

    # Load verification metadata
    verification_meta = verificationMetadata()

    # Collect all image metadata
    image_data = collect_images_for_round(round_name, config)

    if not image_data:
        logger.error(f"No data to process for round {round_name}")
        return pd.DataFrame()

    logger.info(f"Creating verification table with {len(image_data)} image records")

    # Build verification DataFrame
    df = build_verification_dataframe(image_data, verification_meta, round_name, config.image_column_order)

    logger.info(f"Created verification table with {len(df)} records")
    return df


def process_round_tables(round_name: str, config: TableConfig):
    """Process and export tables for a single round.

    Args:
        round_name: Name of the experimental round
        config: Table configuration
    """
    logger.info("")
    logger.info("-" * 40)
    logger.info(f"Starting table organization for round: {round_name}")

    # Create verification table
    df = create_verification_table(round_name, config)

    if df.empty:
        logger.error(f"No data to process for round {round_name}")
        return

    # Export tables
    export_verification_table(df, round_name, config)

    logger.success(f"Completed table organization for round {round_name}")


def process_all_rounds(config: TableConfig):
    """Process tables for all available experimental rounds.

    Args:
        config: Table configuration
    """
    # Find all available rounds
    base_data_path = config.image_base_path

    if not base_data_path.exists():
        logger.error(f"Base data path does not exist: {base_data_path}")
        return

    round_folders = [d for d in base_data_path.iterdir() if d.is_dir()]
    round_names = [f.name for f in round_folders]

    if not round_names:
        logger.error("No round directories found")
        return

    # Sort rounds by numeric value instead of alphabetical
    sorted_round_names = sorted(round_names, key=extract_round_number)

    logger.info(f"Found {len(round_names)} rounds to process:")
    for round_name in sorted_round_names:
        logger.info(f" - {round_name}")
    logger.info("")

    # Collect all rounds data for single Excel export
    all_rounds_data = {}
    total_records = 0

    for round_name in sorted_round_names:
        logger.info("-" * 30 + f"Processing round: {round_name}" + "-" * 30)
        try:
            # Create verification table for this round
            df = create_verification_table(round_name, config)

            if not df.empty:
                all_rounds_data[round_name] = df
                total_records += len(df)
                logger.info(f"Created verification table for {round_name} with {len(df)} records")
            else:
                logger.warning(f"No data found for round {round_name}")

        except Exception as e:
            logger.error(f"Error processing round {round_name}: {e}")
            continue

    # Export all rounds to single Excel file
    if all_rounds_data:
        logger.info(f"Exporting {total_records} total records from {len(all_rounds_data)} rounds to single Excel file")
        export_all_rounds_summary(all_rounds_data, config)
    else:
        logger.error("No valid data found for any rounds")

    logger.success("Completed table organization for all rounds")


if __name__ == "__main__":
    # Example usage
    config = TableConfig(
        image_base_path=Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion"),
        table_output_path=Path("../results/all_rounds_verification_summary.xlsx"),
        image_column_order=["3d", "4d", "5d", "6d", "YES", "HYG", "NAT", "LEU", "ADE"]
    )
    process_all_rounds(config)