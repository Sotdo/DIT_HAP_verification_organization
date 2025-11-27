import cv2
import numpy as np
from pathlib import Path

def find_tetrad_centroid(
    plate_to_tetrad_image,
    min_colony_size=50,
    circularity_threshold=0.5,
    adaptive_thresh_C=2,
    adaptive_thresh_block_size=11,
    contrast_alpha=1.0,
    visualize_colonies=False,
    base_filename_for_viz=None,
    visualization_output_path=None
):
    """
    Recognizes colonies in a grayscale image, calculates their centroid, and optionally saves a visualization.
    """
    if len(plate_to_tetrad_image.shape) == 3:
        gray_image = cv2.cvtColor(plate_to_tetrad_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = plate_to_tetrad_image

    normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    contrast_adjusted_image = cv2.convertScaleAbs(normalized_image, alpha=contrast_alpha, beta=0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(contrast_adjusted_image)
    
    # Ensure block size is an odd number > 1
    if adaptive_thresh_block_size < 3:
        adaptive_thresh_block_size = 3
    if adaptive_thresh_block_size % 2 == 0:
        adaptive_thresh_block_size += 1

    thresh = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_thresh_block_size, -adaptive_thresh_C)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    colony_centroids = []
    visualization_image = None
    if visualize_colonies and base_filename_for_viz and visualization_output_path:
        visualization_image = cv2.cvtColor(contrast_adjusted_image, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: continue
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if area > min_colony_size and circularity > circularity_threshold:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                colony_centroids.append((cx, cy))
                if visualization_image is not None:
                    cv2.drawContours(visualization_image, [contour], -1, (0, 255, 0), 2)
                    cv2.circle(visualization_image, (cx, cy), 5, (0, 0, 255), -1)

    tetrad_center_x = gray_image.shape[1] // 2
    tetrad_center_y = gray_image.shape[0] // 2

    if colony_centroids:
        tetrad_center_x = int(np.mean([p[0] for p in colony_centroids]))
        tetrad_center_y = int(np.mean([p[1] for p in colony_centroids]))
        if visualization_image is not None:
            cv2.circle(visualization_image, (tetrad_center_x, tetrad_center_y), 10, (255, 0, 0), -1)
    else:
        print(f"No colonies found for {base_filename_for_viz}. Using geometric center.")

    if visualization_image is not None:
        viz_output_path_file = visualization_output_path / f"{base_filename_for_viz}_viz.png"
        cv2.imwrite(str(viz_output_path_file), visualization_image)
        print(f"Visualization saved: {viz_output_path_file.name}")

    return tetrad_center_x, tetrad_center_y

def process_single_plate(
    plate_image_full,
    output_path,
    base_filename,
    plate_to_tetrad_height_range,
    plate_to_tetrad_width_range,
    min_colony_size,
    circularity_threshold,
    adaptive_thresh_C,
    adaptive_thresh_block_size,
    contrast_alpha,
    visualize_colonies,
    visualization_output_path,
    final_tetrad_height_percent=None,
    final_tetrad_width_percent=None,
    final_size_px=None
):
    """
    Processes a single cropped plate image to find colonies and extract a final, consistently sized image.
    """
    h_plate, w_plate = plate_image_full.shape[:2]

    intermed_start_h = int(h_plate * (plate_to_tetrad_height_range[0] / 100))
    intermed_end_h = int(h_plate * (plate_to_tetrad_height_range[1] / 100))
    intermed_start_w = int(w_plate * (plate_to_tetrad_width_range[0] / 100))
    intermed_end_w = int(w_plate * (plate_to_tetrad_width_range[1] / 100))
    
    plate_to_tetrad_image = plate_image_full[intermed_start_h:intermed_end_h, intermed_start_w:intermed_end_w]

    relative_centroid_x, relative_centroid_y = find_tetrad_centroid(
        plate_to_tetrad_image,
        min_colony_size,
        circularity_threshold,
        adaptive_thresh_C,
        adaptive_thresh_block_size,
        contrast_alpha,
        visualize_colonies,
        base_filename,
        visualization_output_path
    )

    absolute_centroid_x = intermed_start_w + relative_centroid_x
    absolute_centroid_y = intermed_start_h + relative_centroid_y

    if final_size_px:
        final_width, final_height = final_size_px
    elif final_tetrad_width_percent is not None and final_tetrad_height_percent is not None:
        final_width = int(w_plate * (final_tetrad_width_percent / 100))
        final_height = int(h_plate * (final_tetrad_height_percent / 100))
    else:
        raise ValueError("Either final_size_px or percentage dimensions must be provided.")

    start_x = max(0, absolute_centroid_x - final_width // 2)
    end_x = start_x + final_width
    start_y = max(0, absolute_centroid_y - final_height // 2)
    end_y = start_y + final_height

    final_image = plate_image_full[start_y:end_y, start_x:end_x]
    final_image_resized = cv2.resize(final_image, (final_width, final_height))
    
    cv2.imwrite(str(output_path), final_image_resized)
    print(f"Saved: {output_path.name} (Size: {final_width}x{final_height})")
    return final_width, final_height

def get_plate_crop_radius(image_files):
    """Calculates the average radius from a list of image files."""
    all_radii = []
    for file_path in image_files:
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None: continue
        blurred = cv2.GaussianBlur(image, (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=100, maxRadius=500)
        if circles is not None:
            all_radii.append(np.round(circles[0, :]).astype("int")[0][2])
    if not all_radii:
        raise ValueError("No circles detected in any images. Cannot determine crop radius.")
    return int(np.mean(all_radii))

def process_tetrad_images(input_dir, output_dir, **kwargs):
    """
    Processes all single-plate tetrad images, adjusts centroids for outliers,
    and returns the final output dimensions and the radius used.
    """
    print(f"\n--- Processing Tetrad Images from {input_dir} ---")
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    visualization_output_path = None
    if kwargs.get('visualize_colonies', False):
        visualization_output_path = output_path / "results_visualization"
        visualization_output_path.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in input_path.iterdir() if f.suffix in ['.tif', '.tiff']]

    crop_radius = kwargs.get('target_radius')
    if not crop_radius:
        try:
            crop_radius = get_plate_crop_radius(image_files)
            print(f"Calculated average radius for tetrad plate crop: {crop_radius}")
        except ValueError as e:
            print(e)
            return None, None
    else:
        print(f"Using specified target radius for tetrad plate crop: {crop_radius}")

    # Pass 1: Collect centroids and plate data
    processing_data = []
    for file_path in image_files:
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None: continue

        blurred = cv2.GaussianBlur(image, (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=100, maxRadius=500)

        if circles is not None:
            (x, y, r) = np.round(circles[0, :]).astype("int")[0]
            start_x, end_x = max(0, x - crop_radius), min(image.shape[1], x + crop_radius)
            start_y, end_y = max(0, y - crop_radius), min(image.shape[0], y + crop_radius)
            plate_image = image[start_y:end_y, start_x:end_x]
            plate_image = cv2.resize(plate_image, (2 * crop_radius, 2 * crop_radius))

            h_plate, w_plate = plate_image.shape[:2]
            intermed_start_h = int(h_plate * (kwargs['plate_to_tetrad_height_range'][0] / 100))
            intermed_end_h = int(h_plate * (kwargs['plate_to_tetrad_height_range'][1] / 100))
            intermed_start_w = int(w_plate * (kwargs['plate_to_tetrad_width_range'][0] / 100))
            intermed_end_w = int(w_plate * (kwargs['plate_to_tetrad_width_range'][1] / 100))

            plate_to_tetrad_image = plate_image[intermed_start_h:intermed_end_h, intermed_start_w:intermed_end_w]

            relative_centroid_x, relative_centroid_y = find_tetrad_centroid(
                plate_to_tetrad_image=plate_to_tetrad_image,
                min_colony_size=kwargs['min_colony_size'],
                circularity_threshold=kwargs['circularity_threshold'],
                adaptive_thresh_C=kwargs['adaptive_thresh_C'],
                adaptive_thresh_block_size=kwargs['adaptive_thresh_block_size'],
                contrast_alpha=kwargs.get('contrast_alpha', 1.0),
                visualize_colonies=kwargs.get('visualize_colonies', False),
                base_filename_for_viz=file_path.stem,
                visualization_output_path=visualization_output_path
            )

            absolute_centroid_x = intermed_start_w + relative_centroid_x
            absolute_centroid_y = intermed_start_h + relative_centroid_y

            processing_data.append({
                'file_path': file_path,
                'plate_image': plate_image,
                'centroid': (absolute_centroid_x, absolute_centroid_y)
            })
        else:
            print(f"No circles detected in {file_path.name}. Skipping.")

    if not processing_data:
        print("No images were processed successfully.")
        return None, None

    # Centroid Adjustment
    max_dev = kwargs.get('max_centroid_deviation_px')
    if max_dev is not None and len(processing_data) > 1:
        # Calculate average centroid from all images
        all_centroids = np.array([d['centroid'] for d in processing_data])
        avg_centroid = np.mean(all_centroids, axis=0)

        print(f"Average centroid position: ({avg_centroid[0]:.1f}, {avg_centroid[1]:.1f})")

        # Identify outliers - images whose centroids deviate significantly from average
        inlier_centroids = []
        outlier_indices = []

        for i, data in enumerate(processing_data):
            dist = np.linalg.norm(np.array(data['centroid']) - avg_centroid)
            if dist > max_dev:
                outlier_indices.append(i)
                print(f"Detected outlier: {data['file_path'].name} (deviation: {dist:.2f}px > {max_dev}px threshold)")
            else:
                inlier_centroids.append(data['centroid'])

        # If we have outliers and at least one inlier, recalculate average using only inliers
        if outlier_indices and len(inlier_centroids) >= 1:
            refined_avg_centroid = np.mean(np.array(inlier_centroids), axis=0)
            print(f"Refined average centroid (using inliers only): ({refined_avg_centroid[0]:.1f}, {refined_avg_centroid[1]:.1f})")

            # Adjust outliers to match refined average centroid
            for i in outlier_indices:
                data = processing_data[i]
                print(f"Adjusting centroid for {data['file_path'].name} to refined average position")
                data['final_centroid'] = tuple(refined_avg_centroid.astype(int))

            # Keep inliers with their original centroids
            for i, data in enumerate(processing_data):
                if i not in outlier_indices:
                    data['final_centroid'] = data['centroid']
        else:
            # No outliers detected, or too few inliers - use original centroids
            print("No centroid adjustment needed (no outliers or insufficient inliers)")
            for data in processing_data:
                data['final_centroid'] = data['centroid']
    else:
        print(f"Centroid adjustment disabled (max_centroid_deviation_px: {max_dev}, images: {len(processing_data)})")
        for data in processing_data:
            data['final_centroid'] = data['centroid']

    # Pass 2: Crop, resize, and save using adjusted centroids
    final_width_px, final_height_px = None, None
    for data in processing_data:
        plate_image = data['plate_image']
        h_plate, w_plate = plate_image.shape[:2]
        absolute_centroid_x, absolute_centroid_y = data['final_centroid']

        final_width = int(w_plate * (kwargs['final_tetrad_width_percent'] / 100))
        final_height = int(h_plate * (kwargs['final_tetrad_height_percent'] / 100))

        start_x = max(0, absolute_centroid_x - final_width // 2)
        end_x = start_x + final_width
        start_y = max(0, absolute_centroid_y - final_height // 2)
        end_y = start_y + final_height

        final_image = plate_image[start_y:end_y, start_x:end_x]
        final_image_resized = cv2.resize(final_image, (final_width, final_height))

        output_file_path = output_path / f"{data['file_path'].stem}.png"
        cv2.imwrite(str(output_file_path), final_image_resized)

        if final_width_px is None:
            final_width_px, final_height_px = final_width, final_height
            print(f"Saved: {output_file_path.name} (Size: {final_width_px}x{final_height_px})")

    if final_width_px and final_height_px:
        print(f"Determined final output size for alignment: {final_width_px}x{final_height_px}")
        return (final_width_px, final_height_px), crop_radius

    return None, None

def process_replica_images(input_dir, output_dir, final_output_size_px, tetrad_crop_radius, **kwargs):
    """Processes all multi-plate replica images in a directory."""
    print(f"\n--- Processing Replica Images from {input_dir} ---")
    if not final_output_size_px:
        print("Error: Final output size from tetrad processing is required. Aborting replica processing.")
        return
    if not tetrad_crop_radius:
        print("Error: Tetrad crop radius is required for replica processing. Aborting.")
        return

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    intermediate_output_path = output_path / "intermediate_plates"
    intermediate_output_path.mkdir(parents=True, exist_ok=True)
    
    plate_names = {0: "YES", 1: "NAT", 2: "HYG", 3: "ADE", 4: "LEU"}

    visualization_output_path = None
    if kwargs.get('visualize_colonies', False):
        visualization_output_path = output_path / "results_visualization"
        visualization_output_path.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in input_path.iterdir() if f.suffix in ['.tif', '.tiff']]
    
    crop_radius = tetrad_crop_radius
    print(f"Using synchronized target radius from tetrads for replica plate crop: {crop_radius}")

    single_plate_params = {
        'plate_to_tetrad_height_range': kwargs['plate_to_tetrad_height_range'],
        'plate_to_tetrad_width_range': kwargs['plate_to_tetrad_width_range'],
        'min_colony_size': kwargs['min_colony_size'],
        'circularity_threshold': kwargs['circularity_threshold'],
        'adaptive_thresh_C': kwargs['adaptive_thresh_C'],
        'adaptive_thresh_block_size': kwargs['adaptive_thresh_block_size'],
        'contrast_alpha': kwargs.get('contrast_alpha', 1.0),
        'visualize_colonies': kwargs.get('visualize_colonies', False),
    }

    for file_path in image_files:
        image_color = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        if image_color is None: continue
        
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(image_gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=crop_radius,
            param1=100, param2=40, minRadius=int(crop_radius * 0.8), maxRadius=int(crop_radius * 1.2)
        )
        
        if circles is not None and len(circles[0]) >= 5:
            sorted_by_y = sorted(circles[0], key=lambda c: c[1])
            yes_plate = sorted_by_y[0]
            replicas = sorted(sorted_by_y[1:5], key=lambda c: (c[1], c[0]))
            all_plates = [yes_plate] + replicas
            
            print(f"Detected {len(all_plates)} plates in {file_path.name}.")
            
            for i, (x, y, r) in enumerate(all_plates):
                x, y, r = int(x), int(y), int(r)
                start_x, end_x = max(0, x - crop_radius), min(image_gray.shape[1], x + crop_radius)
                start_y, end_y = max(0, y - crop_radius), min(image_gray.shape[0], y + crop_radius)
                
                plate_image_color = image_color[start_y:end_y, start_x:end_x]
                plate_image_color = cv2.resize(plate_image_color, (2 * crop_radius, 2 * crop_radius))
                
                plate_name = plate_names.get(i, f"plate_{i}")
                
                intermediate_filename = f"{file_path.stem}_{plate_name}_plate.png"
                cv2.imwrite(str(intermediate_output_path / intermediate_filename), plate_image_color)
                print(f"Saved intermediate plate: {intermediate_filename}")

                base_filename = f"{file_path.stem}_{plate_name}"
                output_file_path = output_path / f"{base_filename}.png"

                process_single_plate(
                    plate_image_full=plate_image_color,
                    output_path=output_file_path,
                    base_filename=base_filename,
                    final_size_px=final_output_size_px,
                    visualization_output_path=visualization_output_path,
                    **single_plate_params
                )
        else:
            print(f"Did not detect at least 5 plates in {file_path.name}. Found {len(circles[0]) if circles is not None else 0}. Skipping.")

if __name__ == '__main__':
    # Parameters for Tetrad Images
    tetrad_params = {
        "target_radius": 490, # Optional: Can be auto-calculated if set to None
        "plate_to_tetrad_height_range": (40, 85),
        "plate_to_tetrad_width_range": (10, 95),
        "final_tetrad_height_percent": 30,
        "final_tetrad_width_percent": 75,  
        "min_colony_size": 50,
        "circularity_threshold": 0.7,
        "adaptive_thresh_C": 2,
        "visualize_colonies": True
    }

    # Parameters for Replica Images - note different radius and colony detection settings
    replica_params = {
        "target_radius": 490, # Required for replicas
        "plate_to_tetrad_height_range": (10, 90),
        "plate_to_tetrad_width_range": (10, 90),
        "final_tetrad_height_percent": 80,
        "final_tetrad_width_percent": 80,
        "min_colony_size": 25,  # Smaller colonies expected on replica plates
        "circularity_threshold": 0.6, # Less strict circularity
        "adaptive_thresh_C": 2,
        "visualize_colonies": True
    }

    # Process Tetrad Images
    tetrad_output_size, tetrad_radius = process_tetrad_images(
        input_dir='data/tetrad',
        output_dir='results/tetrad_cropped',
        **tetrad_params
    )

    # Process Replica Images
    process_replica_images(
        input_dir='data/replica',
        output_dir='results/replica_cropped',
        final_output_size_px=tetrad_output_size,
        tetrad_crop_radius=tetrad_radius,
        **replica_params
    )
