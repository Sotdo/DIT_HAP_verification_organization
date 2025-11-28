"""
Batch image cropping script for DIT-HAP verification pipeline.

Simple research script to process tetrad and replica plate images with modern Python.
Usage: python scripts/batch_crop_image.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional

sys.path.append(str(Path(__file__).parent.parent.resolve() / "src"))

from image_processing import process_tetrad_images, process_replica_images
from utils import roundConfig

# %% ------------------------------------ Configuration ------------------------------------ #

@dataclass
class BatchConfig:
    """Simple configuration for research image processing."""
    # Directories
    renamed_file_folder: Path = Path("/hugedata/YushengYang/DIT_HAP_verification/data/processed_data/DIT_HAP_deletion")
    output_base_folder: Path = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion")
    
    # Tetrad parameters
    target_radius: Optional[int] = 490 # plate radius in pixels
    height_range: Tuple[int, int] = (45, 80)
    width_range: Tuple[int, int] = (15, 90)
    final_height_percent: int = 30
    final_width_percent: int = 75
    min_colony_size: int = 50
    circularity_threshold: float = 0.7
    adaptive_thresh_c: int = 2
    adaptive_block_size: int = 30
    contrast_alpha: float = 1.0
    visualize_colonies: bool = True
    max_centroid_deviation: Optional[int] = 50

    # Replica overrides
    replica_min_colony_size: int = 100
    replica_circularity_threshold: float = 0.5
    replica_adaptive_block_size: int = 120
    replica_contrast_alpha: float = 1.5

# %% ------------------------------------ Functions ------------------------------------ #

def main() -> None:
    """Simple main function for research batch processing."""
    print("=" * 60)
    print("DIT-HAP BATCH IMAGE PROCESSING")
    print("=" * 60)

    config = BatchConfig()
    all_rounds = [folder for folder in config.renamed_file_folder.iterdir() if folder.is_dir()]
    round_config = roundConfig(
        raw_data_folder_path = config.renamed_file_folder,
        round_folder_name = "11th_round",  # Process the first round only for this example
        output_folder_path = config.output_base_folder
    )

    # Process tetrads
    print("Processing tetrad images...")
    tetrad_size, tetrad_radius = process_tetrad_images(
        round_config=round_config,
        target_radius=config.target_radius,
        plate_to_tetrad_height_range=config.height_range,
        plate_to_tetrad_width_range=config.width_range,
        final_tetrad_height_percent=config.final_height_percent,
        final_tetrad_width_percent=config.final_width_percent,
        min_colony_size=config.min_colony_size,
        circularity_threshold=config.circularity_threshold,
        adaptive_thresh_C=config.adaptive_thresh_c,
        adaptive_thresh_block_size=config.adaptive_block_size,
        contrast_alpha=config.contrast_alpha,
        visualize_colonies=config.visualize_colonies,
        max_centroid_deviation_px=config.max_centroid_deviation
    )

    if tetrad_size and tetrad_radius:
        print(f"Tetrad processing successful: {tetrad_size[0]}x{tetrad_size[1]}px")

        # Process replicas
        print("Processing replica images...")
        process_replica_images(
            round_config=round_config,
            final_output_size_px=tetrad_size,
            tetrad_crop_radius=tetrad_radius,
            plate_to_tetrad_height_range=config.height_range,
            plate_to_tetrad_width_range=config.width_range,
            min_colony_size=config.replica_min_colony_size,
            circularity_threshold=config.replica_circularity_threshold,
            adaptive_thresh_C=config.adaptive_thresh_c,
            adaptive_thresh_block_size=config.replica_adaptive_block_size,
            contrast_alpha=config.replica_contrast_alpha,
            visualize_colonies=config.visualize_colonies
        )
        print("Replica processing completed!")
    else:
        print("Tetrad processing failed!")

if __name__ == '__main__':
    main()
