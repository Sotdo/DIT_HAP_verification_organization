"""
Batch image cropping script for DIT-HAP verification pipeline.

Simple research script to process tetrad and replica plate images with modern Python.
Usage: python scripts/batch_crop_image.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.resolve() / "src"))

from utils import roundConfig
from tetrad_crop import (
    ImageProcessingConfig,
    process_tetrad_images,
    process_time_course_tetrad_images
)

# %% ------------------------------------ Configuration ------------------------------------ #

@dataclass
class BatchConfig:
    """Simple configuration for research image processing."""
    # Directories
    renamed_file_folder: Path = Path("/hugedata/YushengYang/DIT_HAP_verification/data/processed_data/DIT_HAP_deletion")
    output_base_folder: Path = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion")

    use_table: bool = True # Whether to use the table for processed files
    table_file: Path = Path("../results/all_combined_all_rounds_renamed_summary.xlsx")
    
    # Image processing parameters
    tetrad_config: ImageProcessingConfig = field(
        default_factory=lambda: ImageProcessingConfig()
    )
    replica_config: ImageProcessingConfig = field(
        default_factory=lambda: ImageProcessingConfig(
            height_range = (45, 85),
            width_range = (10, 90),
            min_colony_size=50,
            max_colony_size=5000,
            circularity_threshold=0.45,
            solidity_threshold=0.8,
            adaptive_block_size=200,
            contrast_alpha=1.6
        )
    )

# %% ------------------------------------ Logging ------------------------------------ #
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{module:<20}</cyan>:{line:<4} | <level>{message}</level>",
    level="DEBUG"
)

logger.add(
    Path("../logs/batch_crop_image.log"),
    colorize=False,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {module:<20}:{line:<4} | {message}",
    level="DEBUG",
    mode="w"
)

# %% ------------------------------------ Functions ------------------------------------ #

@logger.catch
def main() -> None:
    """Simple main function for research batch processing."""
    logger.info("=" * 60)
    logger.info("DIT-HAP BATCH IMAGE PROCESSING")
    logger.info("=" * 60)
    logger.info(" ")

    config = BatchConfig()

    if config.use_table:
        logger.info("Loading processed file table...")

        df = pd.read_excel(config.table_file)
        # df = df.query("gene_num == 280")
        logger.info(f"Loaded {len(df)} entries from the table.")
        not_processed_rounds = [
            "22th_round"
        ]
        df = df[~df['round'].isin(not_processed_rounds)]
        logger.info(" ")
        logger.info("*"*30 + "Processing time-course tetrad images... " + "*"*30)
        process_time_course_tetrad_images(
            table_data = df,
            tetrad_config = config.tetrad_config,
            replica_config = config.replica_config,
            output_base_folder = config.output_base_folder
        )

    else:
        logger.info("Processing all rounds from renamed file folder...")
        all_rounds = [folder for folder in config.renamed_file_folder.iterdir() if folder.is_dir()]
        not_processed_rounds = [
            "22th_round"
        ]

        for round_folder in all_rounds:
            logger.info(f"{'#'*20} Processing round folder: {round_folder.name} {'#'*20}")
            if round_folder.name in not_processed_rounds:
                # logger.info(f"Skipping unrecognized round folder: {round_folder.name}")
                logger.info(" ")
                continue
            round_config = roundConfig(
                raw_data_folder_path = config.renamed_file_folder,
                round_folder_name = round_folder.name,  # Process the first round only for this example
                output_folder_path = config.output_base_folder
            )

            # Process tetrads
            logger.info("*"*30 + "Processing tetrad images... " + "*"*30)
            process_tetrad_images(
                round_config=round_config,
                image_processing_config=config.tetrad_config
            )
            
            logger.info(" ")
            logger.info("*"*30 + "Processing replica images... " + "*"*30)
            process_tetrad_images(
                round_config=round_config,
                image_processing_config=config.replica_config,
                replica=True
            )

            logger.info(f"{'#'*20} Completed processing for round folder: {round_folder.name} {'#'*20}")
            logger.info(" ")
            logger.info(" ")


if __name__ == '__main__':
    main()
