# %% =============================== Imports ===============================
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle ,Circle
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

# OpenCV - Image alignment and affine transformation
import cv2

# scikit-image - Image processing toolkit
from skimage import io, measure, morphology #,filters
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu

# scipy - Distance calculation and optimization
# from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy import ndimage

# Set Matplotlib global font to Arial
# If Arial is not available, it will fallback to DejaVu Sans
plt.rcParams.update({
    'axes.unicode_minus': True,  # Correctly display minus sign
    'figure.dpi': 300,            # Image resolution
    'axes.titlesize': 20,         # Title font size
    'axes.titleweight': 'bold', # Title font weight
    'axes.labelsize': 18,         # Axis label font size
    'axes.labelweight': 'bold',  # Axis label font weight
    'font.family': 'Arial',       # Set global font to Arial
    'font.size': 16,              # Base font size
    'legend.fontsize': 16,        # Legend font size
    'axes.spines.top': False,      # Remove top spine
    'axes.spines.right': False,     # Remove right spine
    'axes.linewidth': 3,   # Axis line width
    'lines.linewidth': 3,  # Line width
    'lines.solid_joinstyle': 'round',  # Line join style
    'lines.solid_capstyle': 'round',    # Line cap style
    'image.interpolation': 'nearest',  # Image interpolation
    'pdf.compression': 9  # PDF compression level (0-9)
})

# %% ============================ Constants ==============================
E_GENE_3D_TETRAD_IMG = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/17th_round/3d/242_tyw3_3d_#1_202510.cropped.png")
E_GENE_4D_TETRAD_IMG = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/17th_round/4d/242_tyw3_4d_#1_202510.cropped.png")
E_GENE_5D_TETRAD_IMG = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/17th_round/5d/242_tyw3_5d_#1_202510.cropped.png")
E_GENE_6D_TETRAD_IMG = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/17th_round/6d/242_tyw3_6d_#1_202510.cropped.png")
E_GENE_HYG_IMG = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/17th_round/replica/242_tyw3_HYG_#1_202510.cropped.png")
V_GENE_3D_TETRAD_IMG = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/17th_round/3d/302_meu23_3d_#2_202511.cropped.png")
V_GENE_4D_TETRAD_IMG = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/17th_round/4d/302_meu23_4d_#2_202511.cropped.png")
V_GENE_5D_TETRAD_IMG = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/17th_round/5d/302_meu23_5d_#2_202511.cropped.png")
V_GENE_6D_TETRAD_IMG = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/17th_round/6d/302_meu23_6d_#2_202511.cropped.png")
V_GENE_HYG_IMG = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/17th_round/replica/302_meu23_HYG_#2_202511.cropped.png")


# %% ============================ Configuration =============================
@dataclass
class Configuration:
    """Configuration for colony analysis."""

    tetrad_image_paths: dict[int, Path] = field(default_factory=dict)
    marker_image_path: Path | None = None 
    
    # Grid parameters
    expected_rows: int = 4
    expected_cols: int = 12
    average_x_spacing: float = 52.5 # averaged x spacing in pixels for 4x12 grid from pilot experiments
    average_y_spacing: float = 48 # averaged y spacing in pixels for 4x12 grid from pilot experiments
    spacing_tolerance: float = 3  # tolerance for spacing deviation in pixels
    
    # Colony detection parameters
    # tetrad plate
    tetrad_gray_method: str = "mean"
    tetrad_gaussian_sigma: float = 1.0 # smaller sigma for sharper colonies, larger sigma for smoother colonies
    tetrad_min_area: int = 5 # minimum area of colonies to be detected
    tetrad_max_area: int = 2000 # maximum area of colonies to be detected

    # marker plate
    hyg_gray_channel: int = 0
    hyg_gaussian_sigma: float = 1
    hyg_min_area: int = 40 # larger minimum area for marker colonies
    hyg_max_area: int = 7000 # larger maximum area for marker colonies
    hyg_segmentation_min_distance: int = 10

    # common settings
    morphology_disk_size: int = 3
    morphology_operation: str = "open"  # Options: "open", "close", "dilate", "erode"

    # Alignment parameters
    max_match_distance_ratio: float = 0.2  # Relative to image size
    # min_matched_pairs: int = 4
    # ransac_threshold: float = 5.0
    # ransac_max_iters: int = 1000
    # ransac_confidence: float = 0.99

    # signal detection radius
    signal_detection_radius: int = 15

# %% =========================== Functions =============================
@logger.catch
def convert_to_grayscale(image: np.ndarray, method: str = 'mean', channel: int | None = None) -> np.ndarray:
    """Convert an image to grayscale if it is in color."""
    if image.ndim == 2:
        # Image is already grayscale
        return image
    elif image.ndim == 3:
        if channel is not None:
            # Select specific channel
            return image[:, :, channel]
        elif method == 'mean':
            # Average across color channels
            return np.mean(image, axis=2).astype(image.dtype)
        elif method == 'luminosity':
            # Weighted average for luminosity
            weights = np.array([0.2989, 0.5870, 0.1140])
            return np.dot(image[..., :3], weights).astype(image.dtype)
        else:
            raise ValueError("Invalid method for grayscale conversion.")
    else:
        raise ValueError("Invalid image shape.")

@logger.catch
def apply_gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur to an image."""
    return gaussian(image, sigma=sigma, preserve_range=True).astype(image.dtype)

@logger.catch
def apply_morphology(binary_image: np.ndarray, disk_size: int = 3, operation: str = "open") -> np.ndarray:
    """Apply morphological operations to a binary image."""
    selem = morphology.disk(disk_size)
    if operation == "open":
        morphed = morphology.binary_opening(binary_image, selem)
    elif operation == "close":
        morphed = morphology.binary_closing(binary_image, selem)
    elif operation == "dilate":
        morphed = morphology.binary_dilation(binary_image, selem)
    elif operation == "erode":
        morphed = morphology.binary_erosion(binary_image, selem)
    else:
        raise ValueError("Invalid morphological operation.")
    return morphed

@logger.catch
def binarize_image(
    image: np.ndarray,
    gray_method: str = "mean",
    channel: int | None = None,
    sigma: float = 1.0,
    threshold: int | None = None,
    disk_size: int = 3,
    operation: str = "open"
) -> np.ndarray:
    """Binarize an image using Otsu's thresholding and morphological operations."""
    if channel is None:
        image = convert_to_grayscale(image, method=gray_method)
    else:
        image = convert_to_grayscale(image, channel=channel)
    
    # Apply Gaussian blur
    blurred = apply_gaussian_blur(image, sigma=sigma)
    
    # Direct thresholding if threshold is provided
    if threshold is not None:
        binary = blurred > threshold
    else:
        # Use Otsu's method to determine threshold
        thresh = threshold_otsu(blurred)
        binary = blurred > thresh

    signal_pixels_ratio = np.sum(binary) / binary.size
    if (signal_pixels_ratio < 0.05 or signal_pixels_ratio > 0.9) and channel is not None:
        threshold = np.percentile(blurred, 85)
        logger.warning(f"Unusual signal pixel ratio: {signal_pixels_ratio:.4f}. Using 85th percentile threshold: {threshold:.2f}")
        binary = blurred > threshold

    # Apply morphological operations
    morphed = apply_morphology(binary, disk_size=disk_size, operation=operation)

    return morphed

def watershed_segmentation(
    binary_image: np.ndarray,
    min_distance: int = 20
) -> np.ndarray:
    """Apply watershed segmentation to separate touching colonies."""
    # Compute distance transform
    distance = ndimage.distance_transform_edt(binary_image)
    
    # Find local maxima (colony centers)
    local_max = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=binary_image,
        exclude_border=False
    )
    
    # Create markers for watershed
    markers = np.zeros_like(binary_image, dtype=int)
    markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
    
    # Apply watershed
    labels = watershed(-distance, markers, mask=binary_image)
    
    return labels

@logger.catch
def detect_colonies(
    binary_image: np.ndarray,
    min_area: int = 5,
    max_area: int = 2000,
    is_marker: bool = False,
    segmentation: bool = False,
    min_distance: int = 20
) -> pd.DataFrame:
    """Detect colonies in a binary image and return their properties."""
    if segmentation:
        watershed_labels = watershed_segmentation(binary_image, min_distance=min_distance)
        labeled_image = watershed_labels
    else:
        labeled_image = measure.label(binary_image)
    region_properties_table = pd.DataFrame(measure.regionprops_table(labeled_image, properties=("label","area", "centroid", "eccentricity")))
    filtered_regions = region_properties_table.query(f"area >= {min_area} and area <= {max_area}").copy()

    # centroid calculation, note the order of centroid-1 and centroid-0 for x and y
    filtered_regions.rename(
        columns={
            "centroid-1": "centroid_x",
            "centroid-0": "centroid_y"
        },
        inplace=True
    )

    # eccentricity filter and position filter for tetrad colonies
    if is_marker:
        pass
    else:
        h, w = binary_image.shape
        x_min, x_max = w * 0.02, w * 0.98
        y_min, y_max = h * 0.02, h * 0.98
        filtered_regions = filtered_regions.query(
            f"centroid_x >= {x_min} and centroid_x <= {x_max} and centroid_y >= {y_min} and centroid_y <= {y_max}"
        ).copy()
        filtered_regions = filtered_regions.query("eccentricity <= 0.6").copy()

    return filtered_regions

@logger.catch
def colony_grid_fitting(
    colony_regions: pd.DataFrame,
    expected_rows: int = 4,
    expected_cols: int = 12
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit a grid to detected colony centroids."""
    # Grid Fitting Parameters
    # Extract centroids
    centroids = colony_regions[["centroid_x", "centroid_y"]].to_numpy()

    # Calculate x and y spacing from points that are close to each other
    x_diffs = []
    y_diffs = []

    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            x_diff = abs(centroids[i, 0] - centroids[j, 0])
            y_diff = abs(centroids[i, 1] - centroids[j, 1])
            
            # Points in the same row (y similar, x different)
            if y_diff <= 10 and x_diff > 30:
                x_diffs.append(x_diff)
            
            # Points in the same column (x similar, y different)
            if x_diff <= 10 and y_diff > 30:
                y_diffs.append(y_diff)

    x_diff = np.array(x_diffs)
    y_diff = np.array(y_diffs)

    filtered_x_diffs = x_diff[(x_diff > 30) & (x_diff < 70)]
    filtered_y_diffs = y_diff[(y_diff > 30) & (y_diff < 70)]

    x_spacing = np.mean(filtered_x_diffs)
    y_spacing = np.mean(filtered_y_diffs)
    # Use median of smallest values for robust min calculation
    # x_min = np.median(np.sort(centroids[:, 0])[:3])
    x_min = centroids[:, 0].min()
    y_min = np.median(np.sort(centroids[:, 1])[:5])

    fitted_grid = np.zeros((expected_rows, expected_cols, 2))
    for row in range(expected_rows):
        for col in range(expected_cols):
            fitted_grid[row, col, 0] = x_min + col * x_spacing
            fitted_grid[row, col, 1] = y_min + row * y_spacing

    # colony to grid point assignment
    for idx, region in colony_regions.iterrows():
        centroid = np.array([region["centroid_x"], region["centroid_y"]])
        col_idx = min(11, int(round((centroid[0] - x_min) / x_spacing))) if x_spacing > 0 else 0
        row_idx = min(3, int(round((centroid[1] - y_min) / y_spacing))) if y_spacing > 0 else 0
        colony_regions.loc[idx, "row"] = row_idx
        colony_regions.loc[idx, "col"] = col_idx
        colony_regions.loc[idx, "grid_point_x"] = fitted_grid[row_idx, col_idx, 0]
        colony_regions.loc[idx, "grid_point_y"] = fitted_grid[row_idx, col_idx, 1]
        colony_regions.loc[idx, "distance"] = np.sqrt(
            (centroid[0] - fitted_grid[row_idx, col_idx, 0])
            ** 2 + (centroid[1] - fitted_grid[row_idx, col_idx, 1]) ** 2
        )

    colony_regions["row"] = colony_regions["row"].astype(int)
    colony_regions["col"] = colony_regions["col"].astype(int)

    # keep the minimum distance colony for each row and each col
    colony_regions = colony_regions.groupby(["row", "col"], as_index=False).apply(
        lambda group: group.loc[group["distance"].idxmin()],
        include_groups=False
    ).reset_index(drop=True)

    return fitted_grid, colony_regions

@logger.catch
def rotate_grid(
    grid: np.ndarray,
    colony_regions: pd.DataFrame
):
    """Rotate the grid to best fit the centroids."""
    # find the best fitted colony
    best_fit = colony_regions.loc[colony_regions["distance"].idxmin()]
    best_fitted_row, best_fitted_col = int(best_fit["row"]), int(best_fit["col"])
    rotation_center = (best_fit["centroid_x"], best_fit["centroid_y"])

    # calculate the best rotation angle
    slopes = {}
    for row, row_data in colony_regions.groupby("row"):
        with np.errstate(invalid='ignore'):
            try:
                slope, intercept = np.polyfit(
                    row_data["centroid_x"].tolist(),
                    row_data["centroid_y"].tolist(),
                    1,
                    rcond=None
                )
                slopes[row] = slope
            except:
                # Skip rows with poorly conditioned data
                continue

    mean_slope = np.mean(list(slopes.values()))
    rotation_angle = np.degrees(np.arctan(mean_slope))

    # Rotate the grid
    theta = np.radians(rotation_angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    pts_centered = grid - np.array(rotation_center)
    rotated_pts = np.empty_like(pts_centered)
    rotated_pts[:, :, 0] = pts_centered[:, :, 0] * cos_theta - pts_centered[:, :, 1] * sin_theta
    rotated_pts[:, :, 1] = pts_centered[:, :, 0] * sin_theta + pts_centered[:, :, 1] * cos_theta
    rotated_grid = rotated_pts + np.array(rotation_center)

    return rotated_grid, rotation_angle, best_fitted_row, best_fitted_col

@logger.catch
def optimize_zoom_factor(
    grid_centered: np.ndarray,
    grid_center_point: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    centroids: np.ndarray,
    zoom_range: tuple[float, float] = (0.9, 1.05),
    zoom_step: float = 0.002,
    x_spacing_ref : float | None = None,
    y_spacing_ref : float | None = None,
    spacing_tolerance: float | None = None
) -> tuple[float, float]:
    """Find optimal zoom factor using grid search with vectorized distance calculation."""
    # Generate all zoom factors at once
    x_zoom_factors = np.arange(zoom_range[0], zoom_range[1] + zoom_step, zoom_step)
    y_zoom_factors = np.arange(zoom_range[0], zoom_range[1] + zoom_step, zoom_step)
    
    # Vectorized optimization
    best_error = np.inf
    best_zoom = None
    
    for x_zoom_factor in x_zoom_factors:
        for y_zoom_factor in y_zoom_factors:
            zoom_factor = np.array([x_zoom_factor, y_zoom_factor])
            zoomed_grid = grid_centered * zoom_factor + grid_center_point
            
            # Vectorized distance calculation
            grid_points = zoomed_grid[rows, cols]  # Extract all relevant grid points at once
            grid_x_spacing = np.diff(zoomed_grid, axis=1)[:, :, 0]
            grid_y_spacing = np.diff(zoomed_grid, axis=0)[:, :, 1]
            if x_spacing_ref is not None and spacing_tolerance is not None:
                if np.any(np.abs(grid_x_spacing - x_spacing_ref) > spacing_tolerance):
                    continue
            if y_spacing_ref is not None and spacing_tolerance is not None:
                if np.any(np.abs(grid_y_spacing - y_spacing_ref) > spacing_tolerance):
                    continue
            distances = np.sqrt(np.sum((centroids - grid_points) ** 2, axis=1))
            grid_alignment_error = np.sum(distances)
            
            if grid_alignment_error < best_error:
                best_error = grid_alignment_error
                best_zoom = (x_zoom_factor, y_zoom_factor)
    
    if best_zoom is None:
        best_zoom = (1.0, 1.0)  # Default to no zoom if no valid zoom found
    
    return (float(best_zoom[0]), float(best_zoom[1]))

@logger.catch
def zoom_rotated_grid(
    rotated_grid: np.ndarray,
    colony_regions: pd.DataFrame,
    best_rotated_row: int | None = None,
    best_rotated_col: int | None = None,
    zoom_range: tuple[float, float] = (0.9, 1.1),
    zoom_step: float = 0.002,
    x_spacing_ref : float | None = None,
    y_spacing_ref : float | None = None,
    spacing_tolerance: float | None = None
) -> tuple[np.ndarray, pd.DataFrame, tuple[float, float]]:
    """Zoom the rotated grid to best fit the centroids."""
    # Find the best fitted colony if not provided
    if best_rotated_row is None or best_rotated_col is None:
        best_fit = colony_regions.loc[colony_regions["distance"].idxmin()]
        best_rotated_row, best_rotated_col = int(best_fit["row"]), int(best_fit["col"])
    
    # Pre-extract data as numpy arrays for vectorized operations
    rows = colony_regions["row"].to_numpy().astype(int)
    cols = colony_regions["col"].to_numpy().astype(int)
    centroids = colony_regions[["centroid_x", "centroid_y"]].to_numpy()
    
    # Pre-compute grid centered
    grid_center_point = rotated_grid[best_rotated_row, best_rotated_col]
    grid_centered = rotated_grid - grid_center_point
    
    # Find optimal zoom factor
    best_zoom = optimize_zoom_factor(
        grid_centered,
        grid_center_point,
        rows,
        cols,
        centroids,
        zoom_range,
        zoom_step,
        x_spacing_ref,
        y_spacing_ref,
        spacing_tolerance
    )
    
    # Compute best zoomed grid
    best_zoomed_grid = grid_centered * np.array(best_zoom) + grid_center_point

    # Update colony_regions with vectorized operations
    colony_regions = colony_regions.set_index(["row", "col"]).sort_index()  # Add sort_index() here
    n_rows, n_cols = best_zoomed_grid.shape[:2]
    
    # Vectorized update of grid points
    for row in range(n_rows):
        for col in range(n_cols):
            colony_regions.loc[(row, col), "grid_point_x"] = best_zoomed_grid[row, col, 0]
            colony_regions.loc[(row, col), "grid_point_y"] = best_zoomed_grid[row, col, 1]
    
    colony_regions["area"] = colony_regions["area"].fillna(0)

    return best_zoomed_grid, colony_regions, best_zoom

@logger.catch
def grid_fitting_and_optimization(
    colony_regions: pd.DataFrame,
    expected_rows: int = 4,
    expected_cols: int = 12,
    x_spacing_ref : float | None = None,
    y_spacing_ref : float | None = None,
    spacing_tolerance: float | None = None
) -> tuple[pd.DataFrame, np.ndarray, float, int, int, tuple[float, float]]:
    """Fit and optimize a colony grid to detected centroids."""
    fitted_grid, colony_regions = colony_grid_fitting(
        colony_regions,
        expected_rows,
        expected_cols
    )
    rotated_grid, rotation_angle, best_row, best_col = rotate_grid(
        fitted_grid,
        colony_regions
    )
    rotated_zoom_grid, colony_regions, best_zoom = zoom_rotated_grid(
        rotated_grid,
        colony_regions,
        best_row,
        best_col
    )
    return (
        colony_regions,
        rotated_zoom_grid,
        rotation_angle,
        best_row,
        best_col,
        best_zoom
    )

@logger.catch
def colony_grid_table(
    image: np.ndarray,
    config: Configuration
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Create a grid table of colonies."""

    # Binarize image
    binary_image = binarize_image(
        image,
        gray_method=config.tetrad_gray_method,
        sigma=config.tetrad_gaussian_sigma,
        disk_size=config.morphology_disk_size,
        operation=config.morphology_operation
    )

    # Detect colonies
    detected_regions = detect_colonies(
        binary_image,
        min_area=config.tetrad_min_area,
        max_area=config.tetrad_max_area
    )

    # Grid fitting and optimization
    colony_regions, rotated_zoom_grid, rotation_angle, best_row, best_col, best_zoom = grid_fitting_and_optimization(
        detected_regions,
        expected_rows=config.expected_rows,
        expected_cols=config.expected_cols,
        x_spacing_ref=config.average_x_spacing,
        y_spacing_ref=config.average_y_spacing,
        spacing_tolerance=config.spacing_tolerance
    )

    return binary_image, colony_regions, rotated_zoom_grid

@logger.catch
def marker_plate_point_matching(
    colony_regions: pd.DataFrame,
    marker_plate_image: np.ndarray,
    config: Configuration
) -> tuple[np.ndarray | None, pd.DataFrame, float, float, float, float, list[np.ndarray], list[np.ndarray]]:
    """Match marker points to plate points using the Hungarian algorithm."""
    # Detect marker colonies
    marker_binary = binarize_image(marker_plate_image, channel=config.hyg_gray_channel, sigma=config.hyg_gaussian_sigma)
    marker_morphed = apply_morphology(marker_binary, disk_size=config.morphology_disk_size, operation=config.morphology_operation)
    # marker segmentation can be added here
    # watershed_labels = watershed_segmentation(marker_morphed, min_distance=20)
    marker_regions = detect_colonies(marker_morphed, min_area=config.hyg_min_area, max_area=config.hyg_max_area, is_marker=True, segmentation=True, min_distance=config.hyg_segmentation_min_distance)
    marker_centroids = marker_regions[["centroid_x", "centroid_y"]].to_numpy()
    tetrad_centroids = colony_regions[["centroid_x", "centroid_y"]].dropna().to_numpy()
    tetrad_centroids_indices = colony_regions[["centroid_x", "centroid_y"]].dropna().index
    
    if len(marker_centroids) == 0 or len(tetrad_centroids) == 0:
        logger.error("No centroids detected in marker or tetrad images for matching.")
        return None, colony_regions, 1.0, 0.0, 0.0, 0.0, [], []

    # Match marker centroids to tetrad centroids
    dist_matrix = cdist(marker_centroids, tetrad_centroids)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    h_ref, w_ref = marker_plate_image.shape[:2]
    max_distance = config.max_match_distance_ratio * np.sqrt(h_ref**2 + w_ref**2)
    matched_marker_centroids = []
    matched_tetrad_centroids = []
    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] <= max_distance:
            matched_tetrad_index = tetrad_centroids_indices[c]
            colony_regions.loc[matched_tetrad_index, "matched_marker_centroid_x"] = marker_centroids[r, 0]
            colony_regions.loc[matched_tetrad_index, "matched_marker_centroid_y"] = marker_centroids[r, 1]
            matched_marker_centroids.append(marker_centroids[r])
            matched_tetrad_centroids.append(tetrad_centroids[c])
    
    # RANSAC transform estimation for robust alignment
    M_estimate, inliers = cv2.estimateAffinePartial2D(
        np.array(matched_marker_centroids),
        np.array(matched_tetrad_centroids),
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0
    )
    marker_aligned = cv2.warpAffine(
        marker_plate_image,
        M_estimate,
        (w_ref, h_ref)
    )
    scale = np.sqrt(M_estimate[0, 0]**2 + M_estimate[0, 1]**2)
    angle = np.degrees(np.arctan2(M_estimate[1, 0], M_estimate[0, 0]))
    tx, ty = M_estimate[0, 2], M_estimate[1, 2]

    return marker_aligned, colony_regions, scale, angle, tx, ty, matched_tetrad_centroids, matched_marker_centroids

@logger.catch
def genotyping(
    binary_tetrad_image: np.ndarray,
    aligned_marker_image_gray: np.ndarray,
    colony_regions: pd.DataFrame,
    radius: int = 15
) -> pd.DataFrame:
    """Genotype colonies based on tetrad binary image."""

    for idx, region in colony_regions.iterrows():
        cx, cy = int(region["grid_point_x"]), int(region["grid_point_y"])
        x1, x2 = max(0, cx - radius), min(binary_tetrad_image.shape[1], cx + radius)
        y1, y2 = max(0, cy - radius), min(binary_tetrad_image.shape[0], cy + radius)
        tetrad_patch = binary_tetrad_image[y1:y2, x1:x2]
        marker_patch = aligned_marker_image_gray[y1:y2, x1:x2]
        if marker_patch.size == 0:
            logger.warning(f"Empty marker patch for colony at index {idx}. Skipping genotype assignment.")
            colony_regions.loc[idx, "genotype"] = "Unknown"
            continue
        mean_tetrad_intensity = np.mean(tetrad_patch)
        mean_marker_intensity = np.mean(marker_patch)
        median_tetrad_intensity = np.median(tetrad_patch)
        median_marker_intensity = np.median(marker_patch)
        quantile_80_marker_intensity = np.percentile(marker_patch, 80)
        otsu_threshold = threshold_otsu(aligned_marker_image_gray)
        positive_signal_median = np.median(aligned_marker_image_gray[aligned_marker_image_gray > otsu_threshold])

        marker_intensity_threshold = max(100, otsu_threshold)
        logger.debug(f"Using marker intensity threshold: {marker_intensity_threshold}")
        colony_regions.loc[idx, "tetrad_intensity"] = mean_tetrad_intensity
        colony_regions.loc[idx, "marker_intensity"] = mean_marker_intensity
        colony_regions.loc[idx, "median_tetrad_intensity"] = median_tetrad_intensity
        colony_regions.loc[idx, "median_marker_intensity"] = median_marker_intensity
        colony_regions.loc[idx, "otsu_threshold"] = otsu_threshold
        colony_regions.loc[idx, "positive_signal_median"] = positive_signal_median
        if mean_tetrad_intensity > 0.2 and quantile_80_marker_intensity < marker_intensity_threshold:
            colony_regions.loc[idx, "genotype"] = "WT"
        else:
            colony_regions.loc[idx, "genotype"] = "Deletion"

    return colony_regions

@logger.catch
def plot_genotype_results(
    tetrad_results: dict[int, dict],
    aligned_marker_image: np.ndarray,
    colony_regions: pd.DataFrame,
    radius: int = 15
):
    """Plot genotyping results with colony annotations."""
    n_days = len(tetrad_results)
    last_day = max(tetrad_results.keys())
    n_cols = n_days + 1 + 1 # +1 for marker plate, +1 for colony area plot

    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 4))
    for day_idx, (day, day_data) in enumerate(sorted(tetrad_results.items())):
        ax = axes[day_idx]
        ax.imshow(day_data["image"], rasterized=True)
        ax.set_title(f"Tetrad Day {day}")
        ax.axis('off')
        ax.scatter(
            day_data["table"][f"centroid_x_day{day}"],
            day_data["table"][f"centroid_y_day{day}"],
            s=60,
            color="orange",
            linewidths=1,
            alpha=0.5,
            marker="+"
        )
        for idx, region in colony_regions.iterrows():
            cx, cy = region[f"grid_point_x_day{day}"], region[f"grid_point_y_day{day}"]
            genotype = region["genotype"]
            color = 'green' if genotype == "WT" else 'red'
            square = Rectangle((cx - radius, cy - radius), 2*radius, 2*radius, edgecolor=color, facecolor='none', linewidth=2, alpha=0.4)
            ax.add_patch(square)

    axes[-2].imshow(aligned_marker_image, rasterized=True)
    axes[-2].set_title("Aligned Marker Plate")
    axes[-2].axis('off')
    for idx, region in colony_regions.iterrows():
        cx, cy = region[f"grid_point_x_day{last_day}"], region[f"grid_point_y_day{last_day}"]
        genotype = region["genotype"]
        color = 'green' if genotype == "WT" else 'red'
        square = Rectangle((cx - radius, cy - radius), 2*radius, 2*radius, edgecolor=color, facecolor='none', linewidth=2, alpha=0.4)
        axes[-2].add_patch(square)
    

    # Colony area plot
    ax_area = axes[-1]
    area_table = colony_regions.set_index("genotype", append=True).filter(like="area")
    area_table["area_day0"] = 0
    area_table = area_table.rename_axis("day", axis=1).stack().reset_index().rename(columns={0: "area"})
    area_table["day_num"] = area_table["day"].str.extract(r'day(\d+)').astype(int)
    last_day_WT_colonies_area_mean = area_table.query("genotype == 'WT' and day_num == @last_day")["area"].mean()
    area_table["area[normalized]"] = area_table["area"] / last_day_WT_colonies_area_mean
    area_table = area_table.query("genotype in ['WT', 'Deletion']")
    sns.lineplot(x="day_num", y="area[normalized]", hue="genotype", data=area_table, ax=ax_area, palette={"WT": "green", "Deletion": "red"}, errorbar="se")
    WT_count = colony_regions.query("genotype == 'WT'")["genotype"].count()
    deletion_count = colony_regions.query("genotype == 'Deletion'")["genotype"].count()
    ax_area.set_title(f"Colony Area Over Time:\n WT ({WT_count}) vs Deletion ({deletion_count})")
    ax_area.axhline(1, color='gray', linestyle='--')

    plt.tight_layout()

    return fig


# %% ============================ Main Code ================================
@logger.catch
def genotyping_pipeline(
    tetrad_image_paths: dict[int, Path],
    marker_image_path: Path,
    image_info: tuple | None = None,
    config: Configuration | None = None
) -> tuple[pd.DataFrame, Figure]:
    config = Configuration(
        tetrad_image_paths=tetrad_image_paths,
        marker_image_path=marker_image_path
    )

    last_day = max(config.tetrad_image_paths.keys())
    day_colonies = {}
    last_day_binary = None
    last_day_colony_regions = None
    for day, day_image_path in config.tetrad_image_paths.items():
        day_colonies[day] = {}
        image = io.imread(day_image_path)
        day_colonies[day]["image_notes"] = (*image_info, day) if image_info is not None else (day_image_path.stem, day)
        day_colonies[day]["image"] = image
        day_colonies[day]["binary_image"], day_colonies[day]["table"], day_colonies[day]["grids"] = colony_grid_table(
            image,
            config
        )
        if day == last_day and last_day_binary is None and last_day_colony_regions is None:
            last_day_binary = day_colonies[day]["binary_image"]
            last_day_colony_regions = day_colonies[day]["table"]
        day_colonies[day]["table"] = day_colonies[day]["table"].add_suffix(f"_day{day}")
    
    if last_day_binary is None or last_day_colony_regions is None:
        raise ValueError("No tetrad images were processed.")

    marker_plate_image = io.imread(config.marker_image_path)
    marker_aligned, colony_regions, scale, angle, tx, ty, matched_tetrad_centroids, matched_marker_centroids = marker_plate_point_matching(
        last_day_colony_regions,
        marker_plate_image,
        config
    )

    if marker_aligned is None or len(matched_tetrad_centroids) < 3 or marker_aligned.size == 0:
        logger.error("Marker plate alignment failed. No aligned marker image available.")
        marker_aligned = marker_plate_image

    marker_aligned_gray = convert_to_grayscale(marker_aligned, channel=config.hyg_gray_channel)
    genotyping_colony_regions = genotyping(last_day_binary, marker_aligned_gray, last_day_colony_regions, radius=config.signal_detection_radius)

    all_colony_regions = pd.concat(
        [day_colonies[day]["table"] for day in sorted(day_colonies.keys())] + [genotyping_colony_regions[["genotype", "tetrad_intensity", "marker_intensity", "median_tetrad_intensity", "median_marker_intensity", "otsu_threshold", "positive_signal_median"]]],
        axis=1
    ).sort_index(level=[1,0], axis=0)
    fig = plot_genotype_results(day_colonies, marker_aligned, all_colony_regions, radius=config.signal_detection_radius)
    return all_colony_regions, fig
# %% =========================== Debug Functions =============================
def plot_detected_colonies(
    binary_image: np.ndarray,
    colony_regions: pd.DataFrame
):
    """ Plot detected colonies on binary image."""
    fig, ax = plt.subplots()
    ax.imshow(binary_image, cmap='gray')
    ax.set_title("Detected Colonies")
    ax.scatter(
        colony_regions["centroid_x"],
        colony_regions["centroid_y"],
        s=60, c='red', label='Detected Colonies', marker="+", linewidths=2
    )
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_detected_colonies_and_grids(day_colonies: dict[int, dict]):
    """ Plot detected colonies and fitted grids for each day."""
    n_days = len(day_colonies)
    fig, axes = plt.subplots(1, n_days, figsize=(6 * n_days, 6))
    for day_idx, (day, day_data) in enumerate(sorted(day_colonies.items())):
        ax = axes[day_idx]
        ax.imshow(day_data["image"])
        ax.set_title(f"Tetrad Day {day} - Detected Colonies and Fitted Grid")
        ax.scatter(
            day_data["table"][f"centroid_x_day{day}"],
            day_data["table"][f"centroid_y_day{day}"],
            s=60, c='red', label='Detected Colonies', marker="+", linewidths=2
        )
        ax.axis('off')
        for idx, region in day_data["table"].iterrows():
            cx, cy = region[f"grid_point_x_day{day}"], region[f"grid_point_y_day{day}"]
            circle = Circle((cx, cy), radius=25, edgecolor='green', facecolor='none', linewidth=2, alpha=0.6)
            ax.add_patch(circle)
    plt.tight_layout()
    plt.show()
    plt.close()
# %% =========================== Example Usage =============================

if __name__ == "__main__":
    all_colony_regions, fig = genotyping_pipeline(
        tetrad_image_paths={
            3: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/2nd_round/3d/48_pub1_3d_#3_202411.cropped.png"),
            4: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/2nd_round/4d/48_pub1_4d_#3_202411.cropped.png"),
            5: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/2nd_round/5d/48_pub1_5d_#3_202411.cropped.png"),
            6: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/2nd_round/6d/48_pub1_6d_#3_202411.cropped.png")
        },
        marker_image_path=Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/2nd_round/replica/48_pub1_HYG_#3_202411.cropped.png")
    )

    plt.show()
    plt.close()

# %% =========================== Manual Run ================================

tetrad_image_paths={
    3: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/3rd_round/3d/66_isu1_3d_#1_202412.cropped.png"),
    4: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/3rd_round/4d/66_isu1_4d_#1_202412.cropped.png"),
    5: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/3rd_round/5d/66_isu1_5d_#1_202412.cropped.png"),
    6: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/3rd_round/6d/66_isu1_6d_#1_202412.cropped.png")
}
marker_image_path=Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/3rd_round/replica/66_isu1_HYG_#1_202412.cropped.png")

config = Configuration(
    tetrad_image_paths=tetrad_image_paths,
    marker_image_path=marker_image_path
)

last_day = max(config.tetrad_image_paths.keys())
day_colonies = {}
last_day_binary = None
last_day_colony_regions = None
for day, day_image_path in config.tetrad_image_paths.items():
    day_colonies[day] = {}
    image = io.imread(day_image_path)
    day_colonies[day]["image"] = image
    day_colonies[day]["binary_image"], day_colonies[day]["table"], day_colonies[day]["grids"] = colony_grid_table(
        image,
        config
    )
    if day == last_day and last_day_binary is None and last_day_colony_regions is None:
        last_day_binary = day_colonies[day]["binary_image"]
        last_day_colony_regions = day_colonies[day]["table"]
    day_colonies[day]["table"] = day_colonies[day]["table"].add_suffix(f"_day{day}")

if last_day_binary is None or last_day_colony_regions is None:
    raise ValueError("No tetrad images were processed.")

marker_plate_image = io.imread(config.marker_image_path)
marker_aligned, colony_regions, scale, angle, tx, ty, matched_tetrad_centroids, matched_marker_centroids = marker_plate_point_matching(
    last_day_colony_regions,
    marker_plate_image,
    config
)

if marker_aligned is None or len(matched_tetrad_centroids) < 3 or marker_aligned.size == 0:
    logger.error("Marker plate alignment failed. No aligned marker image available.")
    marker_aligned = marker_plate_image

marker_aligned_gray = convert_to_grayscale(marker_aligned, channel=config.hyg_gray_channel)
genotyping_colony_regions = genotyping(last_day_binary, marker_aligned_gray, last_day_colony_regions, radius=config.signal_detection_radius)

all_colony_regions = pd.concat(
    [day_colonies[day]["table"] for day in sorted(day_colonies.keys())] + [genotyping_colony_regions[["genotype", "tetrad_intensity", "marker_intensity", "median_tetrad_intensity", "median_marker_intensity", "otsu_threshold", "positive_signal_median"]]],
    axis=1
).sort_index(level=[1,0], axis=0)
fig = plot_genotype_results(day_colonies, marker_aligned, all_colony_regions, radius=config.signal_detection_radius)
# %%
