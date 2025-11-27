from src.image_processing import process_tetrad_images, process_replica_images

if __name__ == '__main__':
    # Parameters for Tetrad Images
    tetrad_params = {
        "target_radius": 490, # Optional: Can be auto-calculated if set to None
        "plate_to_tetrad_height_range": (45, 80),
        "plate_to_tetrad_width_range": (15, 90),
        "final_tetrad_height_percent": 30,
        "final_tetrad_width_percent": 75,
        "min_colony_size": 50,
        "circularity_threshold": 0.7,
        "adaptive_thresh_C": 2,
        "adaptive_thresh_block_size": 30,
        "contrast_alpha": 1.0,
        "visualize_colonies": True
    }

    # Parameters for Replica Images
    replica_params = {
        # target_radius is passed dynamically
        "plate_to_tetrad_height_range": (45, 80),
        "plate_to_tetrad_width_range": (15, 90),
        "min_colony_size": 100,
        "circularity_threshold": 0.5,
        "adaptive_thresh_C": 2,
        "adaptive_thresh_block_size": 120, # Larger block size for potentially uneven replica background
        "contrast_alpha": 1.5,
        "visualize_colonies": True
    }

    # Process Tetrad Images and get the final dimensions and radius for alignment
    run_results, used_radius = process_tetrad_images(
        input_dir='data/tetrad',
        output_dir='results/tetrad_cropped',
        **tetrad_params
    )
    final_output_size_px = run_results

    # Process Replica Images only if the tetrad processing was successful
    if final_output_size_px and used_radius:
        process_replica_images(
            input_dir='data/replica',
            output_dir='results/replica_cropped',
            final_output_size_px=final_output_size_px,
            tetrad_crop_radius=used_radius,
            **replica_params
        )
    else:
        print("\nSkipping replica processing because final output size or radius could not be determined from tetrads.")
