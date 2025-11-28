"""
OpenCV-based image processing for DIT-HAP verification experiments.
Research-focused pipeline for yeast colony detection and plate processing.

Usage:
    # Process tetrad images
    tetrad_output_size, tetrad_radius = process_tetrad_images(
        input_dir='data/tetrad',
        output_dir='results/tetrad_cropped',
        target_radius=490,
        min_colony_size=50
    )

    # Process replica images using tetrad output
    process_replica_images(
        input_dir='data/replica',
        output_dir='results/replica_cropped',
        final_output_size_px=tetrad_output_size,
        tetrad_crop_radius=tetrad_radius
    )
"""

import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm
from utils import roundConfig

# %% ------------------------------------ Functions ------------------------------------ #

def find_tetrad_centroid(
    plate_image: np.ndarray,
    min_colony_size: int = 50,
    circularity_threshold: float = 0.7,
    adaptive_thresh_C: int = 2,
    adaptive_block_size: int = 30,
    contrast_alpha: float = 1.0,
    visualize_colonies: bool = False,
    base_filename: Optional[str] = None,
    viz_path: Optional[Path] = None
) -> Tuple[int, int]:
    """Find colony centroids in a plate image using OpenCV."""
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if len(plate_image.shape) == 3 else plate_image

    # Enhance contrast
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    contrast = cv2.convertScaleAbs(normalized, alpha=contrast_alpha, beta=0)

    # CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(contrast)

    # Adaptive thresholding
    if adaptive_block_size < 3: adaptive_block_size = 3
    if adaptive_block_size % 2 == 0: adaptive_block_size += 1

    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, adaptive_block_size, -adaptive_thresh_C)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find colonies
    colony_centroids = []
    viz_image = None

    if visualize_colonies and base_filename and viz_path:
        viz_image = cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: continue

        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if area > min_colony_size and circularity > circularity_threshold:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                colony_centroids.append((cx, cy))

                if viz_image is not None:
                    cv2.drawContours(viz_image, [contour], -1, (0, 255, 0), 2)
                    cv2.circle(viz_image, (cx, cy), 5, (0, 0, 255), -1)

    # Calculate centroid
    if colony_centroids:
        centroid_x = int(np.mean([p[0] for p in colony_centroids]))
        centroid_y = int(np.mean([p[1] for p in colony_centroids]))
    else:
        centroid_x, centroid_y = gray.shape[1] // 2, gray.shape[0] // 2
        if base_filename:
            print(f"No colonies found for {base_filename}. Using geometric center.")

    # Save visualization
    if viz_image is not None:
        cv2.imwrite(str(viz_path / f"{base_filename}_viz.png"), viz_image)
        # print(f"Visualization saved: {base_filename}_viz.png")

    if viz_image is not None:
        cv2.circle(viz_image, (centroid_x, centroid_y), 10, (255, 0, 0), -1)

    return centroid_x, centroid_y

def process_single_plate(
    plate_image: np.ndarray,
    output_path: Path,
    base_filename: str,
    height_range: Tuple[int, int] = (45, 80),
    width_range: Tuple[int, int] = (15, 90),
    final_height_percent: int = 30,
    final_width_percent: int = 75,
    final_size_px: Optional[Tuple[int, int]] = None,
    viz_path: Optional[Path] = None,
    max_centroid_deviation_px: Optional[int] = None,
    reference_centroid: Optional[Tuple[int, int]] = None,
    **kwargs
) -> Tuple[int, int]:
    """Process single plate image with colony detection, cropping, and optional centroid adjustment."""
    h, w = plate_image.shape[:2]

    # Extract processing region
    start_h, end_h = int(h * height_range[0] / 100), int(h * height_range[1] / 100)
    start_w, end_w = int(w * width_range[0] / 100), int(w * width_range[1] / 100)
    plate_region = plate_image[start_h:end_h, start_w:end_w]

    # Find colony centroid
    rel_cx, rel_cy = find_tetrad_centroid(
        plate_region,
        base_filename=base_filename,
        viz_path=viz_path,
        **kwargs
    )

    # Convert to absolute coordinates
    abs_cx, abs_cy = start_w + rel_cx, start_h + rel_cy

    # Centroid adjustment for both tetrad and replica plates
    if reference_centroid is not None and max_centroid_deviation_px is not None:
        # Calculate distance from reference centroid
        dist = np.linalg.norm(np.array([abs_cx, abs_cy]) - np.array(reference_centroid))
        if dist > max_centroid_deviation_px:
            print(f"Adjusting centroid for {base_filename} (deviation: {dist:.1f}px > {max_centroid_deviation_px}px)")
            abs_cx, abs_cy = reference_centroid
        else:
            if base_filename:
                print(f"Using detected centroid for {base_filename} (deviation: {dist:.1f}px)")

    # Determine final size
    if final_size_px:
        final_w, final_h = final_size_px
    else:
        final_w = int(w * final_width_percent / 100)
        final_h = int(h * final_height_percent / 100)

    # Crop and resize
    crop_x1, crop_x2 = max(0, abs_cx - final_w // 2), min(w, abs_cx + final_w // 2)
    crop_y1, crop_y2 = max(0, abs_cy - final_h // 2), min(h, abs_cy + final_h // 2)
    cropped = plate_image[crop_y1:crop_y2, crop_x1:crop_x2]
    final = cv2.resize(cropped, (final_w, final_h))

    cv2.imwrite(str(output_path), final)
    # print(f"Saved: {output_path.name} ({final_w}x{final_h})")
    return final_w, final_h

def get_plate_crop_radius(image_files: List[Path]) -> int:
    """Calculate average plate radius using Hough circle detection."""
    radii = []
    for file_path in image_files:
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        blurred = cv2.GaussianBlur(img, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=100, param1=50, param2=30,
            minRadius=100, maxRadius=500
        )

        if circles is not None:
            radii.append(int(circles[0, 0, 2]))

    if not radii:
        raise ValueError("No circles detected in any images")
    return int(np.mean(radii))

def process_tetrad_images(
    round_config: roundConfig,
    **kwargs
) -> Optional[Tuple[Tuple[int, int] | None, int | None]]:
    """Process tetrad images and return synchronized dimensions."""
    for sub_folder, input_output_paths in round_config.all_sub_folders.items():
        if sub_folder == "replica":
            continue
        input_folder_path = input_output_paths["input"]
        output_folder_path = input_output_paths["output"]
        print(f"Processing tetrad images in: {input_folder_path}")
        print(f"Output will be saved to: {output_folder_path}")
        # Set the output visualization path
        viz_path = None
        if kwargs.get('visualize_colonies', False):
            viz_path = output_folder_path / "results_visualization"
            viz_path.mkdir(parents=True, exist_ok=True)

        # Find all tetrad images
        all_images = list(input_folder_path.glob("*.tif*"))

        if not all_images:
            print("No .tif files found")
            return None, None

        print(f"Found {len(all_images)} tetrad images")

        # Get crop radius
        target_radius = kwargs.get('target_radius')
        if not target_radius:
            try:
                target_radius = get_plate_crop_radius(all_images)
                print(f"Auto-detected radius: {target_radius}px")
            except ValueError as e:
                print(f"Radius detection failed: {e}")
                return None, None
        else:
            print(f"Using specified radius: {target_radius}px")

        # Process images
        processed_data = []
        for img_path in tqdm(all_images, desc="Processing tetrads"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None: continue

            # Detect plate
            blurred = cv2.GaussianBlur(img, (9, 9), 2)
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT,
                dp=1.2, minDist=100, param1=50, param2=30,
                minRadius=100, maxRadius=500
            )

            if circles is not None:
                x, y = int(circles[0, 0, 0]), int(circles[0, 0, 1])

                # Extract plate region
                x1, x2 = max(0, x - target_radius), min(img.shape[1], x + target_radius)
                y1, y2 = max(0, y - target_radius), min(img.shape[0], y + target_radius)
                plate = img[y1:y2, x1:x2]
                plate = cv2.resize(plate, (2 * target_radius, 2 * target_radius))

                # Find centroid
                centroid_kwargs = {k: v for k, v in kwargs.items()
                               if k in ['min_colony_size', 'circularity_threshold',
                                         'adaptive_thresh_c', 'adaptive_block_size',
                                         'contrast_alpha', 'visualize_colonies']}
                rel_cx, rel_cy = find_tetrad_centroid(
                    plate,
                    base_filename=img_path.stem,
                    viz_path=viz_path,
                    **centroid_kwargs
                )

                processed_data.append({
                    'path': img_path,
                    'plate': plate,
                    'centroid': (rel_cx, rel_cy)
                })

        if not processed_data:
            print("No images processed successfully")
            return None, None

        # Centroid adjustment (simplified)
        max_deviation = kwargs.get('max_centroid_deviation_px')
        if max_deviation and len(processed_data) > 1:
            centroids = np.array([d['centroid'] for d in processed_data])
            avg_centroid = np.mean(centroids, axis=0)

            for data in processed_data:
                dist = np.linalg.norm(np.array(data['centroid']) - avg_centroid)
                if dist > max_deviation:
                    print(f"Adjusting {data['path'].name} (deviation: {dist:.1f}px)")
                    data['final_centroid'] = tuple(avg_centroid.astype(int))
                else:
                    data['final_centroid'] = data['centroid']
        else:
            for data in processed_data:
                data['final_centroid'] = data['centroid']

        # Process final images
        final_size = None
        for data in processed_data:
            final_w = int(2 * target_radius * kwargs.get('final_tetrad_width_percent', 75) / 100)
            final_h = int(2 * target_radius * kwargs.get('final_tetrad_height_percent', 30) / 100)

            cx, cy = data['final_centroid']
            x1, x2 = max(0, cx - final_w // 2), min(2 * target_radius, cx + final_w // 2)
            y1, y2 = max(0, cy - final_h // 2), min(2 * target_radius, cy + final_h // 2)

            cropped = data['plate'][y1:y2, x1:x2]
            final = cv2.resize(cropped, (final_w, final_h))

            out_file = output_folder_path / f"{data['path'].stem}.cropped.png"
            cv2.imwrite(str(out_file), final)

            if final_size is None:
                final_size = (final_w, final_h)

        if final_size:
            print(f"Final size: {final_size[0]}x{final_size[1]}")
            return final_size, target_radius
        else:
            return None, None

def process_replica_images(
    round_config: roundConfig,
    final_output_size_px: Optional[Tuple[int, int]],
    tetrad_crop_radius: Optional[int],
    **kwargs
) -> None:
    """Process replica images using tetrad dimensions."""
    if not final_output_size_px or not tetrad_crop_radius:
        print("Error: Need tetrad output size and radius")
        return

    input_path = round_config.all_sub_folders["replica"]["input"]
    output_path = round_config.all_sub_folders["replica"]["output"]

    plate_names = {0: "YES", 1: "NAT", 2: "HYG", 3: "ADE", 4: "LEU"}

    # Find replica images
    image_files = [f for f in input_path.glob("*.tif*")]

    if not image_files:
        print(f"No replica images found in {input_path}")
        return

    print(f"Processing {len(image_files)} replica images")
    crop_radius = tetrad_crop_radius

    for img_path in tqdm(image_files, desc="Processing replicas"):
        img_color = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_color is None: continue

        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(img_gray, (9, 9), 2)

        # Detect 5 replica plates
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.5, minDist=crop_radius,
            param1=100, param2=40,
            minRadius=int(crop_radius * 0.8),
            maxRadius=int(crop_radius * 1.2)
        )

        if circles is not None and len(circles[0]) >= 5:
            sorted_circles = sorted(circles[0], key=lambda c: c[1])
            plates = [sorted_circles[0]] + sorted(sorted_circles[1:5], key=lambda c: (c[1], c[0]))

            for i, (x, y, r) in enumerate(plates):
                x, y = int(x), int(y)
                plate_name = plate_names.get(i, f"plate_{i}")

                # Extract plate
                x1, x2 = max(0, x - crop_radius), min(img_color.shape[1], x + crop_radius)
                y1, y2 = max(0, y - crop_radius), min(img_color.shape[0], y + crop_radius)
                plate_img = img_color[y1:y2, x1:x2]
                plate_img = cv2.resize(plate_img, (2 * crop_radius, 2 * crop_radius))

                # Process final output
                out_path = output_path / (img_path.stem.replace("YHZAY2A", plate_name) + ".cropped.png")

                # Filter kwargs for process_single_plate
                plate_kwargs = {k: v for k, v in kwargs.items()
                               if k in ['min_colony_size', 'circularity_threshold',
                                         'adaptive_thresh_c', 'adaptive_block_size',
                                         'contrast_alpha', 'visualize_colonies',
                                         'height_range', 'width_range',
                                         'final_height_percent', 'final_width_percent',
                                         'max_centroid_deviation_px']}

                process_single_plate(
                    plate_image=plate_img,
                    output_path=out_path,
                    base_filename=f"{img_path.stem}_{plate_name}",
                    final_size_px=final_output_size_px,
                    **plate_kwargs
                )
        else:
            print(f"Only {len(circles[0]) if circles is not None else 0} plates found in {img_path.name}")

def main() -> None:
    """Simple main function for testing."""
    print("DIT-HAP Image Processing")
    tetrad_size, tetrad_radius = process_tetrad_images(
        round_config=roundConfig(
            raw_data_folder_path='data',
            round_folder_name='1st_round',
            output_folder_path='results/tetrad_cropped'
        ),
        target_radius=490,
        min_colony_size=50,
        visualize_colonies=True
    )

    if tetrad_size and tetrad_radius:
        process_replica_images(
            round_config=roundConfig(
                raw_data_folder_path='data',
                round_folder_name='replica',
                output_folder_path='results/replica_cropped'
            ),
            final_output_size_px=tetrad_size,
            tetrad_crop_radius=tetrad_radius,
            min_colony_size=100,
            visualize_colonies=True
        )

if __name__ == '__main__':
    main()
