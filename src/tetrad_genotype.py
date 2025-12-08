# %% =============================== Imports ===============================
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger

import numpy as np
import pandas as pd
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

# OpenCV - Image alignment and affine transformation
import cv2

# scikit-image - Image processing toolkit
from skimage import io, filters, measure, morphology
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu

# scipy - Distance calculation and optimization
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment, minimize
from scipy import ndimage

# Set Matplotlib global font to Arial
# If Arial is not available, it will fallback to DejaVu Sans
plt.rcParams.update({
    'axes.unicode_minus': False,  # Correctly display minus sign
    'figure.dpi': 100,            # Image resolution
    'axes.titlesize': 12,         # Title font size
    'axes.labelsize': 10,         # Axis label font size
})

# %% ============================ Configuration =============================
@dataclass
class AnalysisConfig:
    """Configuration for colony analysis."""
    
    # Grid parameters
    expected_rows: int = 4
    expected_cols: int = 12
    
    # Colony detection parameters
    # tetrad plate
    tetrad_gray_method: str = "mean"
    tetrad_gaussian_sigma: float = 1.0
    tetrad_min_area: int = 50

    # marker plate
    hyg_gray_method: str = "channel_0"
    hyg_gaussian_sigma: float = 1.0
    hyg_min_area: int = 50

    # common settings
    morphology_disk_size: int = 3
    morphology_operation: str = "open"  # Options: "open", "close", "dilate", "erode"

    
    # Alignment parameters
    max_match_distance_ratio: float = 0.15  # Relative to image size
    min_matched_pairs: int = 4
    ransac_threshold: float = 5.0
    ransac_max_iters: int = 1000
    ransac_confidence: float = 0.99
    
    # Grid fitting parameters
    max_rotation_angle: float = 15.0  # degrees
    max_translation: float = 30.0  # pixels
    translation_search_step: int = 5  # pixels
    translation_search_range: int = 20  # pixels
    
    # Output parameters
    output_dpi: int = 300
    figure_size: tuple[int, int] = field(default_factory=lambda: (16, 12))

# %% ============================ Constants ==============================
E_GENE_TETRAD_IMG = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/17th_round/6d/242_tyw3_6d_#1_202510.cropped.png")
E_GENE_HYG_IMG = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/17th_round/replica/242_tyw3_HYG_#1_202510.cropped.png")
V_GENE_TETRAD_IMG = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/17th_round/6d/302_meu23_6d_#2_202511.cropped.png")
V_GENE_HYG_IMG = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/17th_round/replica/302_meu23_HYG_#2_202511.cropped.png")
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
def binarize_image(image: np.ndarray, gray_method: str = "mean", channel: int | None = None, sigma: float = 1.0, threshold: int | None = None) -> np.ndarray:
    """Binarize an image using Otsu's thresholding."""
    if channel is None:
        image = convert_to_grayscale(image, method=gray_method)
    else:
        image = convert_to_grayscale(image, channel=channel)
    
    # Apply Gaussian blur
    blurred = apply_gaussian_blur(image, sigma=sigma)
    
    # Direct thresholding if threshold is provided
    if threshold is not None:
        binary = blurred > threshold
        return binary

    # Use Otsu's method to determine threshold
    thresh = threshold_otsu(blurred)
    binary = blurred > thresh
    return binary

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
def detect_colonies(
    binary_image: np.ndarray,
    min_area: int = 100,
    max_area: int = 2000
) -> pd.DataFrame:
    """Detect colonies in a binary image and return their properties."""
    labeled_image = measure.label(binary_image)
    region_properties_table = pd.DataFrame(measure.regionprops_table(labeled_image, properties=("label","area", "centroid", "eccentricity")))
    filtered_regions = region_properties_table.query(f"area >= {min_area} and area <= {max_area}")
    # centroid calculation, note the order of centroid-1 and centroid-0 for x and y
    filtered_regions.rename(
        columns={
            "centroid-1": "centroid_x",
            "centroid-0": "centroid_y"
        },
        inplace=True
    )
    return filtered_regions

@logger.catch
def colony_grid_fitting(
    colony_regions: pd.DataFrame,
    expected_rows: int = 4,
    expected_cols: int = 12
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit a grid to detected colony centroids."""
    # Grid Fitting Parameters
    x_min, x_max = colony_regions["centroid_x"].min(), colony_regions["centroid_x"].max()
    y_min, y_max = colony_regions["centroid_y"].min(), colony_regions["centroid_y"].max()
    x_spacing = (x_max - x_min) / (expected_cols - 1) if expected_cols > 1 else 0
    y_spacing = (y_max - y_min) / (expected_rows - 1) if expected_rows > 1 else 0

    fitted_grid = np.zeros((expected_rows, expected_cols, 2))
    for row in range(expected_rows):
        for col in range(expected_cols):
            fitted_grid[row, col, 0] = x_min + col * x_spacing
            fitted_grid[row, col, 1] = y_min + row * y_spacing

    # colony to grid point assignment
    for idx, row in colony_regions.iterrows():
        centroid = np.array([row["centroid_x"], row["centroid_y"]])
        col_idx = int(round((centroid[0] - x_min) / x_spacing)) if x_spacing > 0 else 0
        row_idx = int(round((centroid[1] - y_min) / y_spacing)) if y_spacing > 0 else 0
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
        slope, intercept = np.polyfit(
            row_data["centroid_x"].tolist(),
            row_data["centroid_y"].tolist(),
            1
        )
        slopes[row] = slope

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
    zoom_range: tuple[float, float] = (0.9, 1.1),
    zoom_step: float = 0.002
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
            distances = np.sqrt(np.sum((centroids - grid_points) ** 2, axis=1))
            grid_alignment_error = np.sum(distances)
            
            if grid_alignment_error < best_error:
                best_error = grid_alignment_error
                best_zoom = (x_zoom_factor, y_zoom_factor)
    
    return best_zoom

@logger.catch
def zoom_rotated_grid(
    rotated_grid: np.ndarray,
    colony_regions: pd.DataFrame,
    best_rotated_row: int | None = None,
    best_rotated_col: int | None = None,
    zoom_range: tuple[float, float] = (0.9, 1.1),
    zoom_step: float = 0.002
) -> tuple[np.ndarray, pd.DataFrame, tuple[float, float]]:
    """Zoom the rotated grid to best fit the centroids."""
    # Find the best fitted colony if not provided
    if best_rotated_row is None or best_rotated_col is None:
        best_fit = colony_regions.loc[colony_regions["distance"].idxmin()]
        best_rotated_row, best_rotated_col = int(best_fit["row"]), int(best_fit["col"])
    
    # Pre-extract data as numpy arrays for vectorized operations
    rows = colony_regions["row"].values.astype(int)
    cols = colony_regions["col"].values.astype(int)
    centroids = colony_regions[["centroid_x", "centroid_y"]].values
    
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
        zoom_step
    )
    
    # Compute best zoomed grid
    best_zoomed_grid = grid_centered * np.array(best_zoom) + grid_center_point

    # Update colony_regions with vectorized operations
    colony_regions.set_index(["row", "col"], inplace=True)
    n_rows, n_cols = best_zoomed_grid.shape[:2]
    
    # Vectorized update of grid points
    for row in range(n_rows):
        for col in range(n_cols):
            if (row, col) in colony_regions.index:
                colony_regions.loc[(row, col), "grid_point_x"] = best_zoomed_grid[row, col, 0]
                colony_regions.loc[(row, col), "grid_point_y"] = best_zoomed_grid[row, col, 1]
    
    return best_zoomed_grid, colony_regions, best_zoom

@logger.catch
def grid_fitting_and_optimization(
    colony_regions: pd.DataFrame,
    expected_rows: int = 4,
    expected_cols: int = 12
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, float, int, int, np.ndarray, float]:
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
def marker_plate_point_matching(
    colony_regions: pd.DataFrame,
    marker_plate_image: np.ndarray,
    channel: int = 0, # Use channel 0 (Red) by default
    sigma: float = 1.0,
    morphology_disk_size: int = 5,
    morphology_operation: str = "open",
    max_distance_ratio: float = 0.15
):
    """Match marker points to plate points using the Hungarian algorithm."""
    # Detect marker colonies
    marker_binary = binarize_image(marker_plate_image, channel=channel, sigma=sigma)
    marker_morphed = apply_morphology(marker_binary, disk_size=morphology_disk_size, operation=morphology_operation)
    # marker segmentation can be added here
    #watershed_labels = watershed_segmentation(marker_morphed, min_distance=20)
    marker_regions = detect_colonies(marker_morphed)
    marker_centroids = marker_regions[["centroid_x", "centroid_y"]].to_numpy()
    tetrad_centroids = colony_regions[["centroid_x", "centroid_y"]].dropna().to_numpy()
    
    # Match marker centroids to tetrad centroids
    dist_matrix = cdist(marker_centroids, tetrad_centroids)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    h_ref, w_ref = marker_plate_image.shape[:2]
    max_distance = max_distance_ratio * np.sqrt(h_ref**2 + w_ref**2)
    matched_marker_centroids = []
    matched_tetrad_centroids = []
    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] <= max_distance:
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

    return marker_aligned, scale, angle, tx, ty, matched_tetrad_centroids, matched_marker_centroids

@logger.catch
def watershed_segmentation(
    binary_image: np.ndarray,
    min_distance: int = 20
) -> np.ndarray:
    """Perform watershed segmentation on a distance map."""
    # Compute distance map
    distance_map = ndimage.distance_transform_edt(binary_image)
    
    # Find local maxima
    local_maxi = peak_local_max(
        distance_map,
        min_distance=min_distance,
        labels=binary_image
    )
    
    # Marker labelling
    markers = measure.label(local_maxi)
    
    # Apply watershed
    labels = watershed(-distance_map, markers, mask=binary_image)
    
    return labels

@logger.catch
def genotyping(binary_tetrad_image: np.ndarray, ):
    """Genotype colonies based on tetrad binary image."""
    labels = watershed_segmentation(binary_tetrad_image, min_distance=15)
    return labels

# %% ============================ Main Code ================================
E_gene_tetrad_img = io.imread(E_GENE_TETRAD_IMG)
E_gene_hyg_img = io.imread(E_GENE_HYG_IMG)
V_gene_tetrad_img = io.imread(V_GENE_TETRAD_IMG)
V_gene_hyg_img = io.imread(V_GENE_HYG_IMG)

# %%
E_tetrad_gray = convert_to_grayscale(E_gene_tetrad_img)
V_tetrad_gray = convert_to_grayscale(V_gene_tetrad_img)

E_tetrad_binary = binarize_image(E_tetrad_gray, sigma=1)
V_tetrad_binary = binarize_image(V_tetrad_gray, sigma=1)

E_tetrad_morphed = apply_morphology(E_tetrad_binary, disk_size=3, operation="open")
V_tetrad_morphed = apply_morphology(V_tetrad_binary, disk_size=3, operation="open")

detected_E_tetrad_regions = detect_colonies(E_tetrad_morphed)
detected_V_tetrad_regions = detect_colonies(V_tetrad_morphed)


E_colony_regions, E_rotated_zoom_grid, E_rotation_angle, E_best_row, E_best_col, E_best_zoom = grid_fitting_and_optimization(detected_E_tetrad_regions)
V_colony_regions, V_rotated_zoom_grid, V_rotation_angle, V_best_row, V_best_col, V_best_zoom = grid_fitting_and_optimization(detected_V_tetrad_regions)

# %%
E_HYG_marker_aligned, E_scale, E_angle, E_tx, E_ty, matched_E_tetrad_centroids, matched_E_marker_centroids = marker_plate_point_matching(
    E_colony_regions,
    E_gene_hyg_img,
    channel=0,
    max_distance_ratio=0.15
)
V_HYG_marker_aligned, V_scale, V_angle, V_tx, V_ty, matched_V_tetrad_centroids, matched_V_marker_centroids = marker_plate_point_matching(
    V_colony_regions,
    V_gene_hyg_img, 
    channel=0,
    max_distance_ratio=0.15
)

# %%
E_HYG_marker_aligned_gray = convert_to_grayscale(E_HYG_marker_aligned, channel=0)
V_HYG_marker_aligned_gray = convert_to_grayscale(V_HYG_marker_aligned, channel=0)

# %%
h, w = E_tetrad_binary.shape
n_rows, n_cols = E_rotated_zoom_grid.shape[:2]
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(E_tetrad_binary, cmap='gray')
axes[1].imshow(E_HYG_marker_aligned_gray, cmap='gray')
for row in range(n_rows):
    for col in range(n_cols):
        cx, cy = E_rotated_zoom_grid[row, col]
        x1, x2 = max(0, int(cx - 10)), min(w, int(cx + 10))
        y1, y2 = max(0, int(cy - 10)), min(h, int(cy + 10))
        tetrad_patch = E_tetrad_binary[y1:y2, x1:x2]
        hyg_patch = E_HYG_marker_aligned_gray[y1:y2, x1:x2]

        mean_tetrad_intensity = np.mean(tetrad_patch)
        mean_hyg_intensity = np.mean(hyg_patch)

        if mean_tetrad_intensity > 0.5 and mean_hyg_intensity < 100:
            axes[0].add_patch(Circle((cx, cy), radius=25, edgecolor='green', facecolor='none', linewidth=1.5))
            axes[1].add_patch(Circle((cx, cy), radius=25, edgecolor='green', facecolor='none', linewidth=1.5))

plt.show()
plt.close()

# %%
fig, axes = plt.subplots(3, 2, figsize=(12, 15))
axes[0, 0].imshow(E_gene_tetrad_img)
axes[0, 0].set_title("E gene tetrad")
axes[0, 0].scatter(E_tetrad_centroids[:, 0], E_tetrad_centroids[:, 1], c='r', s=30, label='Detected Centroids')
axes[0, 0].scatter(E_rotated_zoom_grid[:, :, 0], E_rotated_zoom_grid[:, :, 1], c='g', s=20, label='Fitted Grid Points')
axes[0, 1].imshow(V_gene_tetrad_img)
axes[0, 1].set_title("V gene tetrad")
axes[0, 1].scatter(V_tetrad_centroids[:, 0], V_tetrad_centroids[:, 1], c='r', s=30, label='Detected Centroids')
axes[0, 1].scatter(V_rotated_zoom_grid[:, :, 0], V_rotated_zoom_grid[:, :, 1], c='g', s=20, label='Fitted Grid Points')
axes[1, 0].imshow(E_gene_hyg_img)
axes[1, 0].set_title("E gene HYG")
axes[1, 0].scatter(E_HYG_marker_aligned[:, 0], E_HYG_marker_aligned[:, 1], c='r', s=30, label='Detected Centroids')
axes[1, 1].imshow(V_gene_hyg_img)
axes[1, 1].set_title("V gene HYG")
axes[1, 1].scatter(V_HYG_marker_aligned[:, 0], V_HYG_marker_aligned[:, 1], c='r', s=30, label='Detected Centroids')
axes[2, 0].imshow(E_gene_tetrad_img, alpha=0.5)
axes[2, 0].imshow(E_gene_hyg_img, alpha=0.5)
for t, h in zip(matched_E_tetrad_centroids, matched_E_marker_centroids):
    axes[2, 0].plot([t[0], h[0]], [t[1], h[1]], 'y-', linewidth=1, alpha=0.7)
    axes[2, 0].plot(t[0], t[1], 'go', markersize=5)
    axes[2, 0].plot(h[0], h[1], 'r^', markersize=5)
axes[2, 0].set_title("E gene HYG aligned to Tetrad")

axes[2, 1].imshow(V_gene_tetrad_img, alpha=0.5)
axes[2, 1].imshow(V_gene_hyg_img, alpha=0.5)
for t, h in zip(matched_V_tetrad_centroids, matched_V_marker_centroids):
    axes[2, 1].plot([t[0], h[0]], [t[1], h[1]], 'y-', linewidth=1, alpha=0.7)
    axes[2, 1].plot(t[0], t[1], 'go', markersize=5)
    axes[2, 1].plot(h[0], h[1], 'r^', markersize=5)

plt.tight_layout()
plt.show()
plt.close()
# %%
