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
import pandas as pd
from tqdm import tqdm
from utils import roundConfig
from scipy import ndimage

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

    # plate parameters
    target_radius: int = 490  # plate radius in pixels

    # plate narrowing parameters
    height_range: tuple[int, int] = (45, 85)
    width_range: tuple[int, int] = (5, 95)

    # binary processing parameters
    adaptive_thresh_c: int = 2
    normal_centroid_deviation_px: int = 15
    max_centroid_deviation_px: int | None = 100
    min_colony_size: int = 5 # minimum colony size in pixels, 50 for tetrads, 100 for replicas
    circularity_threshold: float = 0.7 # circularity threshold for colony detection, 0.7 for tetrads, 0.5 for replicas
    solidity_threshold: float = 0.9 # solidity threshold for colony detection, 0.8 for tetrads, 0.7 for replicas
    adaptive_block_size: int = 30 # must be odd number, 30 for tetrads, 120 for replicas
    contrast_alpha: float = 1.0 # contrast adjustment factor, 1.0 for tetrads, 1.5 for replicas

    # final cropping parameters
    final_height: int = 300
    final_width: int = 750
    visualize_colonies: bool = True

# %% ------------------------------------ Functions ------------------------------------ #
@logger.catch
def find_circle_plates(
    image: str | Path | np.ndarray,
    target_radius: int = 490,
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
    binary = enhanced[enhanced > 10]

    min_radius = target_radius - 25
    max_radius = target_radius + 25
    # Detect circles using Hough transform
    circles = cv2.HoughCircles(
        binary, cv2.HOUGH_GRADIENT,
        dp=dp, minDist=min_dist,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius
    )
  
    if circles is not None:
        return circles[0]
    else:
        init_param1 = param1
        while param1 > 30:
            circles = cv2.HoughCircles(
                enhanced, cv2.HOUGH_GRADIENT,
                dp=dp, minDist=min_dist,
                param1=param1, param2=param2,
                minRadius=min_radius, maxRadius=max_radius
            )
            if circles is not None:
                return circles[0]
            param1 -= 2  # Decrease the higher threshold to be more sensitive
        
        if circles is None:
            min_radius = target_radius - 50
            max_radius = target_radius + 50
            circles = cv2.HoughCircles(
                enhanced, cv2.HOUGH_GRADIENT,
                dp=dp, minDist=min_dist,
                param1=param1, param2=param2,
                minRadius=min_radius, maxRadius=max_radius
            )
            if circles is not None:
                return circles[0]
            else:
                param1 = init_param1
                while param1 > 30:
                    circles = cv2.HoughCircles(
                        enhanced, cv2.HOUGH_GRADIENT,
                        dp=dp, minDist=min_dist,
                        param1=param1, param2=param2,
                        minRadius=min_radius, maxRadius=max_radius
                    )
                    if circles is not None:
                        return circles[0]
                    param1 -= 2  # Decrease the higher threshold to be more sensitive
                
                logger.error(f"No circles detected in image: {image}")
    return None

@logger.catch
def get_plate_crop_radius(image_files: list[Path]) -> int:
    """Calculate average plate radius using Hough circle detection. (Optional)"""
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
        target_radius=radius,
    )

    plates = []
    if circles is None:
        raise ValueError(f"No circle detected in image: {image}")
    else:
        sorted_keys = [(int(round(c[1]/1000)), int(round(c[0]/600))) for c in circles]
        ys = set([c[0] for c in sorted_keys])
        if (REPLICA_NAME in image.stem) and len(ys) != 3:
            logger.error(f"Expected 3 rows of plates, but detected {len(ys)} rows in image: {image}")
        elif "56_hal3_YHZAY2A_#2_202411" == image.stem and len(ys) == 2:
            logger.warning("The raw image missing YES plate")
        sorted_circles = sorted(circles, key=lambda c: (int(round(c[1]/1000)), int(round(c[0]/600))))
        for c in sorted_circles:
            x, y = int(c[0]), int(c[1])
            x1, x2 = max(0, x - radius), min(img.shape[1], x + radius)
            y1, y2 = max(0, y - radius), min(img.shape[0], y + radius)
            plate = img[y1:y2, x1:x2]
            plates.append(plate)

    return plates

@logger.catch
def remove_background(
    gray_image: np.ndarray,
    threshold_percentile: float = 25
) -> np.ndarray:
    """Remove background from the image using the given threshold percentile."""
    threshold_value = np.percentile(gray_image[gray_image > 0], threshold_percentile)
    bg_removed = cv2.subtract(gray_image, np.full_like(gray_image, threshold_value))
    bg_removed = np.clip(bg_removed, 0, 255).astype(np.uint8)
    return bg_removed

@logger.catch
def binarize_image(
    gray_image: np.ndarray,
    replica: bool = False,
) -> tuple[np.ndarray, float]:
    """Binarize the grayscale image using Otsu's thresholding."""
    if replica:
        # Apply adaptive thresholding for replica images
        block_size = 201 if gray_image.shape[0] > 500 else 101
        thresh = cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            -2
        )
    else:
        # Apply Otsu's thresholding
        otsu_threshold, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    signal_ratio = np.sum(thresh > 0) / thresh.size
    # if replica and signal_ratio < 0.15:
    #     while signal_ratio < 0.15 and otsu_threshold > 5:
    #         otsu_threshold -= 2
    #         otsu_threshold, thresh = cv2.threshold(gray_image, otsu_threshold, 255, cv2.THRESH_BINARY)
    #         signal_ratio = np.sum(thresh > 0) / thresh.size

    return thresh, signal_ratio

@logger.catch
def morphological_processing(
    binary_image: np.ndarray
) -> np.ndarray:
    """Apply morphological operations to clean up the binary image."""
    kernel = np.ones((3, 3), np.uint8)
    # closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    # opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    dilated = cv2.morphologyEx(binary_image, cv2.MORPH_DILATE, kernel, iterations=1)
    # Fill holes in the binary image
    thresh_hole_filled = np.array(ndimage.binary_fill_holes(dilated)).astype(np.uint8) * 255
    eroded = cv2.morphologyEx(thresh_hole_filled, cv2.MORPH_ERODE, kernel, iterations=1)

    processed = eroded

    return processed

@logger.catch
def watershed_segmentation(
    gray_image: np.ndarray,
    binary_image: np.ndarray
) -> list[np.ndarray]:
    """Apply watershed segmentation to separate touching colonies."""
    # Distance transform
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)

    # Find local maxima as markers
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Find unknown region
    kernel_dilate = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary_image, kernel_dilate, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Add 1 to all labels so background is not 0 but 1
    markers[unknown == 255] = 0  # Mark unknown region as 0

    # Apply watershed
    plate_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(plate_bgr, markers)

    # Convert watershed labels back to contours with robust detection
    contours = []
    
    for region_label in np.unique(markers):
        if region_label <= 1:  # Skip background (1) and boundaries (-1)
            continue
        
        # Create binary mask for this region
        region_mask = (markers == region_label).astype(np.uint8) * 255
        
        # Apply slight morphological operations to smooth region boundaries
        kernel_smooth = np.ones((2, 2), np.uint8)
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel_smooth)
        
        # Find contours for this region
        region_contours, _ = cv2.findContours(
            region_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Only add contours that meet minimum size requirements
        for contour in region_contours:
            if cv2.contourArea(contour) > 1:  # Filter out noise
                contours.extend([contour])

    return contours

@logger.catch
def filter_contours(
    contours: list[np.ndarray],
    plate_config: ImageProcessingConfig
):
    """Filter contours based on area, circularity, and solidity."""
    filtered_contours = []
    filtered_centroids = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        # hull_perimeter = cv2.arcLength(hull, True)
        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        solidity = area / hull_area if hull_area > 0 else 0
        # convexity = perimeter / hull_perimeter if hull_perimeter > 0 else 0
        
        if area > plate_config.min_colony_size and circularity > plate_config.circularity_threshold and solidity > plate_config.solidity_threshold:
            filtered_contours.append(contour)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                filtered_centroids.append((cx, cy))
    
    return filtered_contours, filtered_centroids

@logger.catch
def calculate_centroids(
    gray: np.ndarray,
    colony_centroids: list[tuple[int, int]],
) -> tuple[int, int]:
    # Calculate centroid
    if colony_centroids:
        # Fallback to original centroids if all were filtered out
        x_coords = sorted([p[0] for p in colony_centroids])
        y_coords = sorted([p[1] for p in colony_centroids])
        if len(colony_centroids) > 14:
            centroid_x = int(np.round(np.mean(x_coords[2:-2])))  # exclude extreme points
            centroid_y = int(np.round(np.mean(y_coords[2:-2])))  # exclude extreme points
        elif len(colony_centroids) > 7:
            centroid_x = int(np.round(np.mean(x_coords[1:-1])))  # exclude extreme points
            centroid_y = int(np.round(np.mean(y_coords[1:-1])))  # exclude extreme points
        else:
            logger.warning("Very few colonies detected, using all points for centroid calculation.")
            centroid_x = int(np.round(np.mean(x_coords)))
            centroid_y = int(np.round(np.mean(y_coords)))
    else:
        centroid_x, centroid_y = gray.shape[1] // 2, gray.shape[0] // 2

    return centroid_x, centroid_y


@logger.catch
def visualize_contours(
    plate_config: ImageProcessingConfig,
    background_image: np.ndarray,
    filtered_contours: list[np.ndarray],
    tetrad_centroid: tuple[int, int],
) -> np.ndarray | None:
    """Visualize detected contours on the plate image."""

    viz_image = None
    if plate_config.visualize_colonies:
        viz_image = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR)
        for contour in filtered_contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.drawContours(viz_image, [contour], -1, (0, 255, 0), 2)
                cv2.circle(viz_image, (cx, cy), 5, (0, 0, 255), -1)
        cv2.rectangle(viz_image, (tetrad_centroid[0]-10, tetrad_centroid[1]-10), (tetrad_centroid[0]+10, tetrad_centroid[1]+10), (0, 0, 255), -1)
    
    return viz_image

@logger.catch
def find_tetrad_centroid(
    plate: np.ndarray,
    plate_config: ImageProcessingConfig,
    viz_image: np.ndarray | None = None,
    replica: bool = False
) -> tuple[int, int, np.ndarray | None, list[tuple[int, int]], int, int]:
    """Find colony centroids in a plate image using OpenCV."""
    # only use red channel for grayscale conversion
    gray = plate[:, :, 2] if len(plate.shape) == 3 else plate
    h, w = gray.shape[:2]
    if gray is None:
        raise ValueError("Input plate_image is invalid or could not be converted to grayscale.")

    # Crop to specified height and width ranges
    start_h, end_h = int(h * plate_config.height_range[0] / 100), int(h * plate_config.height_range[1] / 100)
    start_w, end_w = int(w * plate_config.width_range[0] / 100), int(w * plate_config.width_range[1] / 100)
    gray = gray[start_h:end_h, start_w:end_w]

    # remove noise with Gaussian blur
    if replica:
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        # remove the background signal by reducing the 10th percentile of the positive signal
        gray = remove_background(gray, threshold_percentile=50)
    else:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # remove the background signal by reducing the 10th percentile of the positive signal
        gray = remove_background(gray, threshold_percentile=30)

    # enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(15,15))
    enhanced = clahe.apply(gray)

    normalized = cv2.normalize(enhanced, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # apply otsu's thresholding
    thresh, signal_ratio = binarize_image(normalized, replica=replica)

    # Morphological operations
    processed = morphological_processing(thresh)

    # Apply watershed segmentation to separate touching colonies
    contours = watershed_segmentation(normalized, processed)
    
    # Filter contours based on area, circularity, and solidity
    filtered_contours, colony_centroids = filter_contours(contours, plate_config)

    # Calculate centroid
    tetrad_centroid = calculate_centroids(processed, colony_centroids)
    
    # Visualize detected contours
    viz_image = visualize_contours(plate_config, processed, filtered_contours, tetrad_centroid)

    # transform centroid back to original plate coordinates
    centroid_x, centroid_y = tetrad_centroid
    centroid_x += start_w
    centroid_y += start_h

    return centroid_x, centroid_y, viz_image, colony_centroids, start_w, start_h


@logger.catch
def process_plates(
    plate_images: list[np.ndarray],
    plate_config: ImageProcessingConfig,
    replica: bool = False
) -> list[dict]:
    """
    Process multiple plate images using the 3-step logic:
    1. Crop the image based on height_range and width_range to get focused image
    2. Detect colonies and perform centroid adjustment if necessary
    3. Crop the image based on both centroid and final_height_percent/final_width_percent
    """

    processed_plates = []
    for plate_image in plate_images:
        centroid_x, centroid_y, viz_image, colony_centroids, start_w, start_h = find_tetrad_centroid(plate_image, plate_config=plate_config, replica=replica)
        processed_plates.append({
            'plate_image': plate_image,
            'centroid': (centroid_x, centroid_y),
            'viz_image': viz_image,
            'colony_centroids': colony_centroids,
            'start_w': start_w,
            'start_h': start_h
        })

    # Centroid adjustment for both tetrad and replica plates
    if plate_config.max_centroid_deviation_px is not None:
        # calculate average centroid
        centroids = np.array([p['centroid'] for p in processed_plates])
        avg_centroid = np.mean(centroids, axis=0)
        for p in processed_plates:
            dist = np.linalg.norm(np.array(p['centroid']) - avg_centroid)
            if dist > plate_config.max_centroid_deviation_px:
                logger.warning(f"Large centroid deviation detected: {dist:.1f}px for centroid {p['centroid']} vs average {tuple(avg_centroid.astype(int))}")
                p['final_centroid'] = tuple(avg_centroid.astype(int))
            if dist > plate_config.normal_centroid_deviation_px:
                new_centroid = (np.array(p['centroid']) + avg_centroid) / 2
                logger.info(f"Adjusting centroid from {p['centroid']} to {tuple(new_centroid.astype(int))} (deviation: {dist:.1f}px)")
                p['final_centroid'] = tuple(new_centroid.astype(int))
            else:
                p['final_centroid'] = p['centroid']
            
            cv2.circle(p["viz_image"], p['final_centroid'], 10, (255, 0, 0), -1)

    # Process final cropped images
    for p in processed_plates:
        plate_image = p['plate_image']
        abs_cx, abs_cy = p['final_centroid']
        h, w = plate_image.shape[:2]

        # Crop around the detected centroid
        final_width = plate_config.final_width
        final_height = plate_config.final_height

        start_x = max(0, abs_cx - final_width // 2)
        end_x = start_x + final_width
        start_y = max(0, abs_cy - final_height // 2)
        end_y = start_y + final_height
        cropped = plate_image[start_y:end_y, start_x:end_x]
        p["final_cropped"] = cropped
    return processed_plates

@logger.catch
def process_time_course_plates(
    plate_images: dict[str, dict[str, str | np.ndarray]],
    tetrad_plate_config: ImageProcessingConfig,
    replica_plate_config: ImageProcessingConfig
) -> list[dict]:
    """
    Process multiple plate images using the 3-step logic:
    1. Crop the image based on height_range and width_range to get focused image
    2. Detect colonies and perform centroid adjustment if necessary
    3. Crop the image based on both centroid and final_height_percent/final_width_percent
    """

    processed_plates = []
    for plate_path, plate_image_dict in plate_images.items():
        plate_tag = plate_image_dict['plate_tag']
        plate_image = plate_image_dict['plate_image']
        if plate_tag.startswith(REPLICA_NAME):
            replica = True
            plate_config = replica_plate_config
        else:
            replica = False
            plate_config = tetrad_plate_config
        centroid_x, centroid_y, viz_image, colony_centroids, start_w, start_h = find_tetrad_centroid(plate_image, plate_config=plate_config, replica=replica)
        processed_plates.append({
            'plate_tag': plate_tag,
            'plate_image': plate_image,
            'centroid': (centroid_x, centroid_y),
            'viz_image': viz_image,
            'colony_centroids': colony_centroids,
            'start_w': start_w,
            'start_h': start_h
        })

    plate_config = tetrad_plate_config
    # Centroid adjustment for both tetrad and replica plates
    if plate_config.max_centroid_deviation_px is not None:
        # calculate average centroid
        tetrad_centroids = np.array([p['centroid'] for p in processed_plates if p["plate_tag"] != "replica"])
        replica_centroids = np.array([p['centroid'] for p in processed_plates if p["plate_tag"] == "replica"])
        avg_tetrad_centroid = np.mean(tetrad_centroids, axis=0)
        avg_replica_centroid = np.mean(replica_centroids, axis=0)
        for p in processed_plates:
            n_centroids = len(p["colony_centroids"])
            if p["plate_tag"] == "replica":
                avg_centroid = avg_replica_centroid
            else:
                avg_centroid = avg_tetrad_centroid
            dist = np.linalg.norm(np.array(p['centroid']) - avg_centroid)
            if dist > plate_config.max_centroid_deviation_px:
                logger.warning(f"Large centroid deviation detected: {dist:.1f}px for centroid {p['centroid']} vs average {tuple(avg_centroid.astype(int))}")
                p['final_centroid'] = tuple(avg_centroid.astype(int))
            elif dist > plate_config.normal_centroid_deviation_px:
                new_centroid = np.round((np.array(p['centroid']) + avg_centroid) / 2)
                # logger.info(f"Adjusting centroid from {p['centroid']} to {tuple(new_centroid.astype(int))} (deviation: {dist:.1f}px)")
                p['final_centroid'] = tuple(new_centroid.astype(int))
            else:
                p['final_centroid'] = p['centroid']
            
            draw_init_centroid = np.array(p['centroid']) - np.array([p['start_w'], p['start_h']])
            draw_final_centroid = np.array(p['final_centroid']) - np.array([p['start_w'], p['start_h']])
            draw_avg_centroid = np.array(avg_centroid) - np.array([p['start_w'], p['start_h']])
            cv2.circle(p["viz_image"], tuple(draw_avg_centroid.astype(int)), 10, (0, 255, 255), -1)
            cv2.circle(p["viz_image"], tuple(draw_final_centroid.astype(int)), 10, (255, 0, 0), -1)
            cv2.arrowedLine(p["viz_image"], tuple(draw_init_centroid.astype(int)), tuple(draw_final_centroid.astype(int)), (255, 0, 255), 2)
            cv2.putText(p["viz_image"], f"{dist:.1f}px", 
                        ((draw_init_centroid[0].astype(int) + draw_final_centroid[0].astype(int)) // 2,
                         (draw_init_centroid[1].astype(int) + draw_final_centroid[1].astype(int)) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Process final cropped images
    for p in processed_plates:
        plate_image = p['plate_image']
        abs_cx, abs_cy = p['final_centroid']
        h, w = plate_image.shape[:2]

        # Crop around the detected centroid
        final_width = plate_config.final_width
        final_height = plate_config.final_height

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
                elif len(plates) < 5:
                    if "56_hal3_YHZAY2A_#2_202411" in img_path.stem:
                        logger.warning("The raw image missing YES plate")
                        for idx, plate in enumerate(plates):
                            replica_plate_name = REPLICA_PLATES_ORDER.get(idx+1)
                            img_name = img_path.stem.replace(REPLICA_NAME, replica_plate_name)
                            plates_with_path[img_name] = plate
                    else:
                        logger.error(f"Less than 5 plates detected in image: {img_path}, skipping.")
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
                plate_config=image_processing_config,
                replica = replica
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

@logger.catch
def process_time_course_tetrad_images(
    table_data: pd.DataFrame,
    tetrad_config: ImageProcessingConfig,
    replica_config: ImageProcessingConfig,
    output_base_folder: Path
):
    """Process time-lapse tetrad images."""

    # Get crop radius
    target_radius = tetrad_config.target_radius
    if not target_radius:
        try:
            target_radius = get_plate_crop_radius(table_data["3d_image_path"].dropna().map(Path).tolist()[:20])
            logger.info(f"Auto-detected radius: {target_radius}px")
        except ValueError as e:
            logger.error(f"Radius detection failed: {e}")
    else:
        logger.info(f"Using specified radius: {target_radius}px")

    for index, row in tqdm(table_data.iterrows(), total=table_data.shape[0], desc="Processing time-lapse tetrad images"):
        round = row['round']
        image_paths = row.filter(like='_image_path').dropna().map(Path).to_dict()
        image_paths = {k.replace('_image_path', ''): v for k, v in image_paths.items()}

        # Process images
        plates_with_path = {}
        failed_images = []
        for img_tag, img_path in image_paths.items():
            if img_tag.startswith(REPLICA_NAME):
                img_tag = "replica"
                replica = True
            else:
                replica = False
            output_folder_path = output_base_folder / round / img_tag
            output_folder_path.mkdir(parents=True, exist_ok=True)
            # Set the output visualization path
            viz_path = None
            if tetrad_config.visualize_colonies and replica_config.visualize_colonies:
                viz_path = output_folder_path / "results_visualization"
                viz_path.mkdir(parents=True, exist_ok=True)

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
                        plates_with_path[img_name] = {
                            'plate_tag': img_tag,
                            'plate_image': plate
                        }
                elif len(plates) > 5:
                    logger.error(f"More than 5 plates detected in image: {img_path}, skipping.")
                    failed_images.append(img_path)
                elif len(plates) < 5:
                    if "56_hal3_YHZAY2A_#2_202411" in img_path.stem:
                        logger.warning("The raw image missing YES plate")
                        for idx, plate in enumerate(plates):
                            replica_plate_name = REPLICA_PLATES_ORDER.get(idx+1)
                            img_name = img_path.stem.replace(REPLICA_NAME, replica_plate_name)
                            plates_with_path[img_name] = {
                                'plate_tag': img_tag,
                                'plate_image': plate
                            }
                    else:
                        logger.error(f"Less than 5 plates detected in image: {img_path}, skipping.")
                        failed_images.append(img_path)
            else:
                if len(plates) == 1:
                    plate = plates[0]
                    plates_with_path[img_path.stem] = {
                        'plate_tag': img_tag,
                        'plate_image': plate
                    }
                elif 1 < len(plates) < 5:
                    plate = plates[0]
                    logger.warning(f"Only {len(plates)} plates detected in image: {img_path}, expected 1. Using the first detected plate.")
                    plates_with_path[img_path.stem] = {
                        'plate_tag': img_tag,
                        'plate_image': plate
                    }
                else:
                    logger.error(f"Multiple plates detected in image: {img_path}, skipping.")
                    failed_images.append(img_path)
            
        if len(plates_with_path) > 0:
            processed_plates = process_time_course_plates(
                plate_images=plates_with_path,
                tetrad_plate_config=tetrad_config,
                replica_plate_config=replica_config
            )

            # Save processed images
            for (img_name, image_info), cropped_plate in zip(plates_with_path.items(), processed_plates):
                image_tag = image_info['plate_tag']
                # if image_tag.startswith(REPLICA_NAME):
                #     image_tag = "replica"
                out_path = output_base_folder / round / image_tag / (img_name + ".cropped.png")
                cv2.imwrite(str(out_path), cropped_plate["final_cropped"])

                if cropped_plate["viz_image"] is not None:
                    viz_path = output_base_folder / round / image_tag / "results_visualization" / (img_name + ".viz.png")
                    cv2.imwrite(str(viz_path), cropped_plate["viz_image"])

            if failed_images:
                logger.error("*-" * 30 + f"Failed to process {len(failed_images)} images:" + " -*" * 30)
                for f_img in failed_images:
                    logger.error(f" - {f_img}")
                logger.error("*-" * 70)
