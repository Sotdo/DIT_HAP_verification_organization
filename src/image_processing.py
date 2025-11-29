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


# %% ------------------------------------ Imports ------------------------------------ #
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

import cv2
import numpy as np
from tqdm import tqdm
from utils import roundConfig

# %% ------------------------------------ Constants ------------------------------------ #
REPLICA_PLATES_ORDER = {
    0: "YES",
    1: "HYG",
    2: "NAT",
    3: "LEU",
    4: "ADE"
}

REPLICA_NAME = "YHZAY2A"

# %% ------------------------------------ Dataclass ------------------------------------ #
@dataclass
class ImageProcessingConfig:
    """Configuration for image processing."""
    # Tetrad parameters
    target_radius: int = 490  # plate radius in pixels
    height_range: tuple[int, int] = (45, 80)
    width_range: tuple[int, int] = (10, 90)
    final_height_percent: int = 30
    final_width_percent: int = 75
    visualize_colonies: bool = True
    adaptive_thresh_c: int = 2
    max_centroid_deviation_px: int | None = 75
    min_colony_size: int = 50 # minimum colony size in pixels, 50 for tetrads, 100 for replicas
    circularity_threshold: float = 0.7 # circularity threshold for colony detection, 0.7 for tetrads, 0.5 for replicas
    adaptive_block_size: int = 30 # must be odd number, 30 for tetrads, 120 for replicas
    contrast_alpha: float = 1.0 # contrast adjustment factor, 1.0 for tetrads, 1.5 for replicas

# %% ------------------------------------ Functions ------------------------------------ #
@logger.catch
def find_circle_plates(
    image: str | Path | np.ndarray,
    min_radius: int = 400,
    max_radius: int = 600,
    min_dist: float = 500,
    param1: int = 100,
    param2: int = 50,
    dp: float = 1
) -> np.ndarray | None:
    """Find circular plate images with the given circle radius in a larger image using Hough circle detection."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    else:
        img = image if len(image.shape) != 3 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if img is None:
        raise ValueError(f"Could not read image: {image}")

    blurred = cv2.bilateralFilter(img, 9, 75, 75)  # Edge-preserving blur
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Detect circles using Hough transform
    circles = cv2.HoughCircles(
        enhanced, cv2.HOUGH_GRADIENT,
        dp=dp, minDist=min_dist,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius
    )
  
    if circles is not None:
        return circles[0]
    else:
        while param1 > 30:
            circles = cv2.HoughCircles(
                enhanced, cv2.HOUGH_GRADIENT,
                dp=dp, minDist=min_dist,
                param1=param1, param2=param2,
                minRadius=min_radius, maxRadius=max_radius
            )
            if circles is not None:
                return circles[0]
            param1 -= 5  # Decrease the higher threshold to be more sensitive

@logger.catch
def get_plate_crop_radius(image_files: list[Path]) -> int:
    """Calculate average plate radius using Hough circle detection."""
    radii = []
    for file_path in image_files:
        circles = find_circle_plates(file_path)

        if circles is not None:
            radii.append(int(circles[0, 2]))

    if not radii:
        raise ValueError("No circles detected in any images")
    return int(np.mean(radii))

@logger.catch
def crop_to_circle(
    image: Path,
    radius: int
) -> list[np.ndarray]:
    """Crop the image to a circular region defined by center and radius."""

    img = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {image}")
    circles = find_circle_plates(
        img,
        min_radius=radius - 50,
        max_radius=radius + 50,
    )

    plates = []
    if circles is None:
        raise ValueError(f"No circle detected in image: {image}")
    else:
        sorted_circles = sorted(circles, key=lambda c: (int(c[1]//100), int(c[0]//100)))
        for c in sorted_circles:
            x, y = int(c[0]), int(c[1])
            x1, x2 = max(0, x - radius), min(img.shape[1], x + radius)
            y1, y2 = max(0, y - radius), min(img.shape[0], y + radius)
            plate = img[y1:y2, x1:x2]
            plates.append(plate)

    return plates

@logger.catch
def find_tetrad_centroid(
    plate: np.ndarray,
    plate_config: ImageProcessingConfig,
    viz_image: np.ndarray | None = None
) -> tuple[int, int, np.ndarray | None]:
    """Find colony centroids in a plate image using OpenCV."""
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) if len(plate.shape) == 3 else plate
    h, w = gray.shape[:2]

    # Crop to specified height and width ranges
    start_h, end_h = int(h * plate_config.height_range[0] / 100), int(h * plate_config.height_range[1] / 100)
    start_w, end_w = int(w * plate_config.width_range[0] / 100), int(w * plate_config.width_range[1] / 100)
    gray = gray[start_h:end_h, start_w:end_w]

    # Enhance contrast
    if gray is None:
        raise ValueError("Input plate_image is invalid or could not be converted to grayscale.")
    normalized = cv2.normalize(gray, dst=np.zeros_like(gray), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    contrast = cv2.convertScaleAbs(normalized, alpha=plate_config.contrast_alpha, beta=0)

    # CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(contrast)

    # Adaptive thresholding
    if plate_config.adaptive_block_size < 3:
        plate_config.adaptive_block_size = 3
    if plate_config.adaptive_block_size % 2 == 0:
        plate_config.adaptive_block_size += 1

    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, plate_config.adaptive_block_size, -plate_config.adaptive_thresh_c)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find colonies
    colony_centroids = []
    viz_image = None

    if plate_config.visualize_colonies:
        viz_image = cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if area > plate_config.min_colony_size and circularity > plate_config.circularity_threshold:
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

    if viz_image is not None:
        cv2.circle(viz_image, (centroid_x, centroid_y), 10, (255, 0, 0), -1)
    else:
        viz_image = None

    # transform centroid back to original plate coordinates
    centroid_x += start_w
    centroid_y += start_h

    return centroid_x, centroid_y, viz_image

@logger.catch
def process_plates(
    plate_images: list[np.ndarray],
    plate_config: ImageProcessingConfig,
) -> list[np.ndarray]:
    """
    Process multiple plate images using the 3-step logic:
    1. Crop the image based on height_range and width_range to get focused image
    2. Detect colonies and perform centroid adjustment if necessary
    3. Crop the image based on both centroid and final_height_percent/final_width_percent
    """

    processed_plates = []
    for plate_image in plate_images:
        centroid_x, centroid_y, viz_image = find_tetrad_centroid(plate_image, plate_config=plate_config)
        processed_plates.append({
            'plate_image': plate_image,
            'centroid': (centroid_x, centroid_y),
            'viz_image': viz_image
        })

    # Centroid adjustment for both tetrad and replica plates
    if plate_config.max_centroid_deviation_px is not None:
        # calculate average centroid
        centroids = np.array([p['centroid'] for p in processed_plates])
        avg_centroid = np.mean(centroids, axis=0)
        for p in processed_plates:
            dist = np.linalg.norm(np.array(p['centroid']) - avg_centroid)
            if dist > plate_config.max_centroid_deviation_px:
                logger.info(f"Adjusting centroid from {p['centroid']} to {tuple(avg_centroid.astype(int))} (deviation: {dist:.1f}px)")
                p['final_centroid'] = tuple(avg_centroid.astype(int))
            else:
                p['final_centroid'] = p['centroid']

    # Process final cropped images
    for p in processed_plates:
        plate_image = p['plate_image']
        abs_cx, abs_cy = p['final_centroid']
        h, w = plate_image.shape[:2]

        # Crop around the detected centroid
        final_width = int(w * (plate_config.final_width_percent / 100))
        final_height = int(h * (plate_config.final_height_percent / 100))

        start_x = max(0, abs_cx - final_width // 2)
        end_x = start_x + final_width
        start_y = max(0, abs_cy - final_height // 2)
        end_y = start_y + final_height
        cropped = plate_image[start_y:end_y, start_x:end_x]
        p["final_cropped"] = cropped
    return processed_plates

@logger.catch
def process_tetrad_images(
    round_config: roundConfig,
    image_processing_config: ImageProcessingConfig,
    replica: bool = False
):
    """Process tetrad images and return synchronized dimensions."""
    for sub_folder, input_output_paths in round_config.all_sub_folders.items():
        if replica:
            if sub_folder != "replica":
                continue
        else:
            if sub_folder == "replica":
                continue
        input_folder_path = input_output_paths["input"]
        output_folder_path = input_output_paths["output"]
        logger.info(" ")
        logger.info("-" * 100)
        logger.info(f"Processing tetrad images in: {input_folder_path}")
        logger.info(f"Output will be saved to: {output_folder_path}")
        # Set the output visualization path
        viz_path = None
        if image_processing_config.visualize_colonies:
            viz_path = output_folder_path / "results_visualization"
            viz_path.mkdir(parents=True, exist_ok=True)

        # Find all tetrad images
        all_tif_images = list(input_folder_path.glob("*.tif*"))
        all_jpg_images = list(input_folder_path.glob("*.jpg*"))
        all_images = all_tif_images + all_jpg_images

        if not all_images:
            logger.error("No .tif or .jpg files found")

        logger.info(f"Found {len(all_images)} tetrad images: {len(all_tif_images)} .tif and {len(all_jpg_images)} .jpg")

        # Get crop radius
        target_radius = image_processing_config.target_radius
        if not target_radius:
            try:
                target_radius = get_plate_crop_radius(all_images)
                logger.info(f"Auto-detected radius: {target_radius}px")
            except ValueError as e:
                logger.error(f"Radius detection failed: {e}")
        else:
            logger.info(f"Using specified radius: {target_radius}px")
        # Process images
        plates_with_path = {}
        failed_images = []
        for img_path in tqdm(all_images, desc="Processing tetrads"):
            plates = crop_to_circle(img_path, target_radius)
            if len(plates) == 0:
                logger.error(f"No plates detected in image: {img_path}")
                failed_images.append(img_path)
                continue
            if replica:
                if len(plates) == 5:
                    for idx, plate in enumerate(plates):
                        replica_plate_name = REPLICA_PLATES_ORDER.get(idx)
                        img_name = img_path.stem.replace(REPLICA_NAME, replica_plate_name)
                        plates_with_path[img_name] = plate
                elif len(plates) > 5:
                    logger.error(f"More than 5 plates detected in image: {img_path}, skipping.")
                    failed_images.append(img_path)
            else:
                if len(plates) == 1:
                    plate = plates[0]
                    plates_with_path[img_path.stem] = plate
                elif 1 < len(plates) < 5:
                    plate = plates[0]
                    logger.warning(f"Only {len(plates)} plates detected in image: {img_path}, expected 1. Using the first detected plate.")
                    plates_with_path[img_path.stem] = plate
                else:
                    logger.error(f"Multiple plates detected in image: {img_path}, skipping.")
                    failed_images.append(img_path)

    
        if len(plates_with_path) > 0:
            processed_plates = process_plates(
                plate_images=list(plates_with_path.values()),
                plate_config=image_processing_config
            )

            # Save processed images
            for (img_name, _), cropped_plate in zip(plates_with_path.items(), processed_plates):
                out_path = output_folder_path / (img_name + ".cropped.png")
                cv2.imwrite(str(out_path), cropped_plate["final_cropped"])

                if cropped_plate["viz_image"] is not None:
                    viz_out_path = viz_path / (img_name + ".viz.png")
                    cv2.imwrite(str(viz_out_path), cropped_plate["viz_image"])

            logger.info(f"Processed {len(processed_plates)} tetrad images.")
            if failed_images:
                logger.error("*-" * 30 + f"Failed to process {len(failed_images)} images:" + " -*" * 30)
                for f_img in failed_images:
                    logger.error(f" - {f_img}")
                logger.error("*-" * 70)