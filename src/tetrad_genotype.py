# %% =============================== Imports ===============================
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger
from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle ,Circle
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from sklearn.linear_model import LinearRegression

# OpenCV - Image alignment and affine transformation
import cv2

# scikit-image - Image processing toolkit
from skimage import io, measure, morphology #,filters
from skimage.segmentation import watershed
from skimage.feature import peak_local_max, canny
from skimage.filters import gaussian, threshold_otsu, sobel, scharr
from skimage.exposure import equalize_adapthist

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
    tetrad_gaussian_sigma: float = 0.1 # smaller sigma for sharper colonies, larger sigma for smoother colonies
    tetrad_min_area: int = 4 # minimum area of colonies to be detected
    tetrad_max_area: int = 2000 # maximum area of colonies to be detected
    tetrad_circularity_threshold: float = 0.75
    tetrad_solidity_threshold: float = 0.9

    # marker plate
    hyg_gray_channel: int = 0
    hyg_gaussian_sigma: float = 1
    hyg_min_area: int = 40 # larger minimum area for marker colonies
    hyg_max_area: int = 7000 # larger maximum area for marker colonies
    hyg_segmentation_min_distance: int = 10
    hyg_circularity_threshold: float = 0.6
    hyg_solidity_threshold: float = 0.85

    morphology_disk_size: int = 3

    # Alignment parameters
    max_match_distance_ratio: float = 0.2  # Relative to image size
    # min_matched_pairs: int = 4
    # ransac_threshold: float = 5.0
    # ransac_max_iters: int = 1000
    # ransac_confidence: float = 0.99

    # signal detection radius
    signal_detection_radius: int = 15

# ========================== Grid Restoration =============================
def solve_grid_dataframe(df, approx_spacing, match_tolerance=0.25, img_coords=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Input:
        df: DataFrame containing 'centroid_x', 'centroid_y'
        approx_spacing: approximate spacing between points
        img_coords: True for image coordinate system (Y increases downward, small Y is top);
                    False for mathematical coordinate system (Y increases upward, large Y is top)
    
    Output:
        tagged_df: Original DataFrame with row, col labels
        ideal_grid: 48 perfect grid point coordinates (order: R0C0, R0C1... R3C11)
    """
    points = df[['centroid_x', 'centroid_y']].values
    n_points = len(points)
    
    # ---------------- 1. RANSAC to find basis (invariant) ----------------
    best_score = -1
    best_model = None 
    best_indices = []
    
    # Number of iterations
    iterations = 2000 if n_points > 10 else 100
    
    for _ in range(iterations):
        if n_points < 2: break
        idx_a, idx_b = np.random.choice(n_points, 2, replace=False)
        vec = points[idx_b] - points[idx_a]
        dist = np.linalg.norm(vec)
        
        # Spacing filter
        if abs(dist - approx_spacing) > approx_spacing * 0.25:
            continue
            
        u = vec
        v = np.array([-u[1], u[0]]) # Perpendicular vector
        basis = np.column_stack((u, v))
        
        try:
            basis_inv = np.linalg.inv(basis)
        except:
            continue
            
        # Projection check
        rel_pos = points - points[idx_a]
        coords = rel_pos @ basis_inv.T
        coords_round = np.round(coords)
        diff = np.linalg.norm(coords - coords_round, axis=1)
        
        inliers_mask = diff < match_tolerance
        score = np.sum(inliers_mask)
        
        if score > best_score:
            best_score = score
            best_model = (points[idx_a], basis_inv)
            best_indices = np.where(inliers_mask)[0]
            if score > min(48, n_points * 0.9): break

    if best_model is None:
        print("Error: Failed to detect grid structure")
        return df, pd.DataFrame()

    # ---------------- 2. Initial mapping and window search ----------------
    origin, basis_inv = best_model
    inlier_pts = points[best_indices]
    
    # Get original integer coordinates (may be unordered or rotated)
    raw_uv = np.round((inlier_pts - origin) @ basis_inv.T).astype(int)
    
    # Find best 12x4 coverage
    best_win_score = -1
    best_orientation = None # 'normal' or 'transposed'
    best_offset = (0, 0)
    
    # Search range
    u_min, u_max = raw_uv[:,0].min(), raw_uv[:,0].max()
    v_min, v_max = raw_uv[:,1].min(), raw_uv[:,1].max()
    
    # Brute-force search for best window
    for u_start in range(u_min - 12, u_max + 1):
        for v_start in range(v_min - 12, v_max + 1):
            # Assume u=col, v=row (12x4)
            mask_norm = (raw_uv[:,0] >= u_start) & (raw_uv[:,0] < u_start + 12) & \
                        (raw_uv[:,1] >= v_start) & (raw_uv[:,1] < v_start + 4)
            score_norm = np.sum(mask_norm)
            
            if score_norm > best_win_score:
                best_win_score = score_norm
                best_orientation = 'normal'
                best_offset = (u_start, v_start)
                
            # Assume u=row, v=col (4x12) - transposed case
            mask_trans = (raw_uv[:,0] >= u_start) & (raw_uv[:,0] < u_start + 4) & \
                         (raw_uv[:,1] >= v_start) & (raw_uv[:,1] < v_start + 12)
            score_trans = np.sum(mask_trans)
            
            if score_trans > best_win_score:
                best_win_score = score_trans
                best_orientation = 'transposed'
                best_offset = (u_start, v_start)

    # ---------------- 3. Build temporary logical coordinates ----------------
    # At this step, we only have relative relationships, not which is up or down
    temp_mapping = {} # idx -> [logic_c, logic_r]
    u_start, v_start = best_offset
    
    # Select final inliers
    if best_orientation == 'normal':
        mask = (raw_uv[:,0] >= u_start) & (raw_uv[:,0] < u_start + 12) & \
               (raw_uv[:,1] >= v_start) & (raw_uv[:,1] < v_start + 4)
        valid_indices = best_indices[mask]
        valid_uv = raw_uv[mask]
        for idx, (u, v) in zip(valid_indices, valid_uv):
            temp_mapping[idx] = [u - u_start, v - v_start] # [0..11, 0..3]
    else:
        mask = (raw_uv[:,0] >= u_start) & (raw_uv[:,0] < u_start + 4) & \
               (raw_uv[:,1] >= v_start) & (raw_uv[:,1] < v_start + 12)
        valid_indices = best_indices[mask]
        valid_uv = raw_uv[mask]
        for idx, (u, v) in zip(valid_indices, valid_uv):
            temp_mapping[idx] = [v - v_start, u - u_start] # Transposed: v is col, u is row

    # ---------------- 4. Key correction: physical ordering (top to bottom, left to right) ----------------
    # We have 4 logical rows (0,1,2,3) and 12 logical columns (0..11)
    # But whether logical row 0 is physically top or bottom depends on rotation and basis vector direction
    
    valid_phys_pts = points[valid_indices]
    current_cols = np.array([temp_mapping[i][0] for i in valid_indices])
    current_rows = np.array([temp_mapping[i][1] for i in valid_indices])
    
    # A. Correct row order
    # Calculate the mean physical Y coordinate for each logical row
    row_y_means = {}
    for r in range(4):
        mask_r = (current_rows == r)
        if np.any(mask_r):
            row_y_means[r] = np.mean(valid_phys_pts[mask_r, 1])
        else:
            row_y_means[r] = np.nan
    
    # Determine Y axis trend
    # If img_coords=True (image coordinates), Row 0 should be Min Y
    # Compare Y values of Row 0 and Row 3
    if not np.isnan(row_y_means[0]) and not np.isnan(row_y_means[3]):
        y0 = row_y_means[0]
        y3 = row_y_means[3]
        
        should_flip_row = False
        if img_coords: 
            # Image mode: want Row0 < Row3 (top < bottom)
            if y0 > y3: should_flip_row = True
        else:
            # Plotting mode: want Row0 > Row3 (top > bottom)
            if y0 < y3: should_flip_row = True
        
        if should_flip_row:
            # Perform flip
            for idx in valid_indices:
                temp_mapping[idx][1] = 3 - temp_mapping[idx][1]

    # B. Correct column order
    # Col 0 should always be Min X (leftmost)
    col_x_means = {}
    for c in [0, 11]:
        mask_c = (current_cols == c)
        if np.any(mask_c):
            col_x_means[c] = np.mean(valid_phys_pts[mask_c, 0])
        else:
            col_x_means[c] = np.nan
            
    if not np.isnan(col_x_means[0]) and not np.isnan(col_x_means[11]):
        if col_x_means[0] > col_x_means[11]: # If Col 0's X is greater than Col 11
            # Perform flip
            for idx in valid_indices:
                temp_mapping[idx][0] = 11 - temp_mapping[idx][0]

    # ---------------- 5. Output assembly and secondary fitting ----------------
    # Label DataFrame
    df['row'] = np.nan
    df['col'] = np.nan
    
    # Prepare data for fitting
    A, b = [], []
    for idx, (c, r) in temp_mapping.items():
        # Write to df
        df.at[df.index[idx], 'row'] = r
        df.at[df.index[idx], 'col'] = c
        
        # Fitting equation
        A.append([1, 0, c, 0, r, 0])
        b.append(points[idx][0])
        A.append([0, 1, 0, c, 0, r])
        b.append(points[idx][1])
        
    # Calculate perfect grid parameters
    A = np.array(A)
    b = np.array(b)
    # Use least squares to compute [Ox, Oy, Ux, Uy, Vx, Vy]
    if len(A) > 0:
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        O_fit = params[0:2]
        u_fit = params[2:4] # col vector
        v_fit = params[4:6] # row vector
        
        # Generate ideal grid (in order)
        ideal_data = []
        for r in range(4):
            for c in range(12):
                pt = O_fit + c * u_fit + r * v_fit
                ideal_data.append({
                    'row': r, 
                    'col': c, 
                    'expected_x': pt[0], 
                    'expected_y': pt[1]
                })
        ideal_df = pd.DataFrame(ideal_data)
    else:
        ideal_df = pd.DataFrame()

    return df, ideal_df

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
def remove_background(
    gray_image: np.ndarray,
    threshold_percentile: float = 25
) -> np.ndarray:
    """Remove background from the image using the given threshold percentile."""
    threshold_value = np.percentile(gray_image[gray_image > 0], threshold_percentile)
    bg_removed = gray_image - threshold_value
    bg_removed = np.clip(bg_removed, 0, 255).astype(np.uint8)
    return bg_removed

@logger.catch
def apply_gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur to an image."""
    return gaussian(image, sigma=sigma, preserve_range=True).astype(image.dtype)

@logger.catch
def binarize_image(
    image: np.ndarray,
    gray_method: str = "mean",
    sigma: float = 1.0,
    threshold: int | None = None
) -> np.ndarray:
    """Binarize an image using Otsu's thresholding and morphological operations."""
    image = convert_to_grayscale(image, method=gray_method)    
    bg_removed = remove_background(image, threshold_percentile=25)
    # Apply Gaussian blur
    blur = apply_gaussian_blur(bg_removed, sigma=sigma)

    # Direct thresholding if threshold is provided
    if threshold is not None:
        binary = blur > threshold
    else:
        # Use Otsu's method to determine threshold
        thresh = threshold_otsu(blur)
        binary = blur > thresh

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

def watershed_segmentation(
    binary_image: np.ndarray,
    min_distance: int = 15
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
    segmentation: bool = False,
    min_distance: int = 15
) -> pd.DataFrame:
    """Detect colonies in a binary image and return their properties."""
    if segmentation:
        labeled_image = watershed_segmentation(binary_image, min_distance=min_distance)
    else:
        labeled_image = measure.label(binary_image)
    region_properties_table = pd.DataFrame(measure.regionprops_table(labeled_image, properties=("label","area", "area_convex", "centroid", "eccentricity", "perimeter")))
    region_properties_table["radius"] = np.sqrt(region_properties_table["area"] / np.pi)
    region_properties_table["circularity"] = 4 * np.pi * region_properties_table["area"] / (region_properties_table["perimeter"] ** 2 + 1e-2)
    region_properties_table["solidity"] = region_properties_table["area"] / (region_properties_table["area_convex"] + 1e-2)
    # centroid calculation, note the order of centroid-1 and centroid-0 for x and y
    region_properties_table.rename(
        columns={
            "centroid-1": "centroid_x",
            "centroid-0": "centroid_y"
        },
        inplace=True
    )

    return region_properties_table

@logger.catch
def colony_grid_fitting(
    colony_regions: pd.DataFrame,
    expected_rows: int = 4,
    expected_cols: int = 12
) -> tuple[pd.DataFrame, np.ndarray]:
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

    if filtered_x_diffs.size == 0:
        filtered_x_diffs = x_diff[(x_diff > 55) & (x_diff < 135)]
        filtered_x_diffs = filtered_x_diffs / 2
    if filtered_y_diffs.size == 0:
        filtered_y_diffs = y_diff[(y_diff > 55) & (y_diff < 135)]
        filtered_y_diffs = filtered_y_diffs / 2
    x_spacing = np.mean(filtered_x_diffs)
    y_spacing = np.mean(filtered_y_diffs)
    # Use median of smallest values for robust min calculation
    x_min = centroids[:, 0].min()
    if len(centroids) > 18:
        y_min = np.median(np.sort(centroids[:, 1])[:3])
    else:
        y_min = centroids[:, 1].min()

    fitted_grid = np.zeros((expected_rows, expected_cols, 2))
    for row in range(expected_rows):
        for col in range(expected_cols):
            fitted_grid[row, col, 0] = x_min + col * x_spacing
            fitted_grid[row, col, 1] = y_min + row * y_spacing

    # colony to grid point assignment
    for idx, region in colony_regions.iterrows():
        centroid = np.array([region["centroid_x"], region["centroid_y"]])
        col_idx = min(expected_cols-1, int(round((centroid[0] - x_min) / x_spacing))) if x_spacing > 0 else 0
        row_idx = min(expected_rows-1, int(round((centroid[1] - y_min) / y_spacing))) if y_spacing > 0 else 0
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
    dedup_colony_regions = colony_regions.groupby(["row", "col"], as_index=False).apply(
        lambda group: group.loc[group["distance"].idxmin()],
        include_groups=False
    ).reset_index(drop=True)

    dedup_colony_regions = dedup_colony_regions.query(
        "distance <= 10"
    ).copy()

    return dedup_colony_regions, fitted_grid

@logger.catch
def rotate_grid(
    grid: np.ndarray,
    colony_regions: pd.DataFrame
) -> tuple[np.ndarray, float, int, int, float]:
    """Rotate the grid to best fit the centroids."""
    # find the best fitted colony
    best_fit = colony_regions.loc[colony_regions["distance"].idxmin()]
    best_fitted_row, best_fitted_col = int(best_fit["row"]), int(best_fit["col"])
    rotation_center = (best_fit["centroid_x"], best_fit["centroid_y"])

    # calculate the best rotation angle
    slopes = {}
    for row, row_data in colony_regions.groupby("row"):
        if row_data.shape[0] < 2:
            continue
        with np.errstate(invalid='ignore'):
            try:
                slope, intercept = np.polyfit(
                    row_data["centroid_x"].tolist(),
                    row_data["centroid_y"].tolist(),
                    1
                )
                slopes[row] = slope
            except:
                # Skip rows with poorly conditioned data
                continue
    if not slopes:
        use_slope = 0.0
    else:
        mean_slope = np.mean(list(slopes.values()))
        median_slope = np.median(list(slopes.values()))
        if abs(mean_slope - median_slope) > 0.1:
            use_slope = median_slope
        else:
            use_slope = mean_slope
    rotation_angle = np.degrees(np.arctan(use_slope))

    # Rotate the grid
    theta = np.radians(rotation_angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    pts_centered = grid - np.array(rotation_center)
    rotated_pts = np.empty_like(pts_centered)
    rotated_pts[:, :, 0] = pts_centered[:, :, 0] * cos_theta - pts_centered[:, :, 1] * sin_theta
    rotated_pts[:, :, 1] = pts_centered[:, :, 0] * sin_theta + pts_centered[:, :, 1] * cos_theta
    rotated_grid = rotated_pts + np.array(rotation_center)

    return rotated_grid, rotation_angle, best_fitted_row, best_fitted_col, use_slope

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
    zoom_range: tuple[float, float] = (0.975, 1.025),
    zoom_step: float = 0.002,
    x_spacing_ref : float | None = None,
    y_spacing_ref : float | None = None,
    spacing_tolerance: float | None = None,
    use_slope: float = 0
) -> tuple[np.ndarray, pd.DataFrame, tuple[float, float]]:
    """Zoom the rotated grid to best fit the centroids."""
    # Find the best fitted colony if not provided
    if best_rotated_row is None or best_rotated_col is None:
        best_fit = colony_regions.loc[colony_regions["distance"].idxmin()]
        best_rotated_row, best_rotated_col = int(best_fit["row"]), int(best_fit["col"])
    
    # Pre-extract data as numpy arrays for vectorized operations
    has_area_colony_region = colony_regions.dropna(subset=["area"]).copy()
    rows = has_area_colony_region["row"].to_numpy().astype(int)
    cols = has_area_colony_region["col"].to_numpy().astype(int)
    centroids = has_area_colony_region[["centroid_x", "centroid_y"]].to_numpy()
    
    # Pre-compute grid centered
    grid_center_point = rotated_grid[best_rotated_row, best_rotated_col]
    grid_centered = rotated_grid - grid_center_point
    
    # use slope to adjust zoom range if slope is significant
    adjusted_zoom_range = (
        zoom_range[0] - abs(use_slope)*1.5,
        zoom_range[1] + abs(use_slope)*1.5
    )

    # Find optimal zoom factor
    best_zoom = optimize_zoom_factor(
        grid_centered,
        grid_center_point,
        rows,
        cols,
        centroids,
        adjusted_zoom_range,
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
    colony_regions, fitted_grid = colony_grid_fitting(
        colony_regions,
        expected_rows,
        expected_cols
    )
    rotated_grid, rotation_angle, best_row, best_col, use_slope = rotate_grid(
        fitted_grid,
        colony_regions
    )
    rotated_zoom_grid, colony_regions, best_zoom = zoom_rotated_grid(
        rotated_grid,
        colony_regions,
        best_row,
        best_col,
        use_slope = use_slope
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
def filter_colonies(
    binary_image: np.ndarray,
    colony_regions: pd.DataFrame,
    config: Configuration,
    is_marker: bool = False
) -> pd.DataFrame:
    
    h, w = binary_image.shape

    if is_marker:
        min_area = config.hyg_min_area
        max_area = config.hyg_max_area
        circularity_threshold = config.hyg_circularity_threshold
        solidity_threshold = config.hyg_solidity_threshold
    else:
        min_area = config.tetrad_min_area
        max_area = config.tetrad_max_area
        circularity_threshold = config.tetrad_circularity_threshold
        solidity_threshold = config.tetrad_solidity_threshold
    
    filtered_regions = colony_regions.query(
        f"area >= {min_area} and area <= {max_area} and circularity >= {circularity_threshold} and solidity >= {solidity_threshold}"
    ).copy()
    
    filtered_regions, restored_grid = solve_grid_dataframe(filtered_regions, approx_spacing=52)


    if filtered_regions.empty or restored_grid.empty:
        logger.warning("*** Grid restoration failed: No colonies detected or grid fitting failed. Filter colonies at the boundaries and try again.")
        left, right = w * 0.015, w * 0.985
        top, bottom = h * 0.015, h * 0.985
        filtered_regions = filtered_regions.query(
            f"centroid_x >= {left} and centroid_x <= {right} and centroid_y >= {top} and centroid_y <= {bottom}"
        ).copy()
        filtered_regions, restored_grid = solve_grid_dataframe(filtered_regions, approx_spacing=52)

    if filtered_regions.empty or restored_grid.empty:
        logger.warning("*** Grid restoration failed: Restored grid points are out of image bounds. Using second method for grid fitting.")
        filtered_regions, rotated_zoom_grid, rotation_angle, best_row, best_col, best_zoom = grid_fitting_and_optimization(
            filtered_regions,
            expected_rows=config.expected_rows,
            expected_cols=config.expected_cols,
            x_spacing_ref=config.average_x_spacing,
            y_spacing_ref=config.average_y_spacing,
            spacing_tolerance=config.spacing_tolerance
        )
    else:
        grid_spacing = np.median(np.sqrt(np.diff(restored_grid["expected_x"])**2 + np.diff(restored_grid["expected_y"])**2))
        restored_grid_left, restored_grid_top = restored_grid["expected_x"].min()-10, restored_grid["expected_y"].min()-10
        restored_grid_right, restored_grid_bottom = restored_grid["expected_x"].max()+10, restored_grid["expected_y"].max()+10

        if restored_grid_left < -20 or restored_grid_top < -20 or restored_grid_right > w + 20 or restored_grid_bottom > h + 20:
            logger.warning("*** Grid restoration failed: Restored grid points are out of image bounds. Using second method for grid fitting.")

            filtered_regions, rotated_zoom_grid, rotation_angle, best_row, best_col, best_zoom = grid_fitting_and_optimization(
                filtered_regions,
                expected_rows=config.expected_rows,
                expected_cols=config.expected_cols,
                x_spacing_ref=config.average_x_spacing,
                y_spacing_ref=config.average_y_spacing,
                spacing_tolerance=config.spacing_tolerance
            )

        else:
            filtered_regions.dropna(subset=["row", "col"], inplace=True)
            filtered_regions.set_index(["row", "col"], inplace=True, drop=True)
            
            # Use reindex to ensure all grid points are present and sorted
            grid_df = restored_grid.set_index(["row", "col"])
            filtered_regions = filtered_regions.reindex(grid_df.index)
            
            filtered_regions["grid_point_x"] = grid_df["expected_x"]
            filtered_regions["grid_point_y"] = grid_df["expected_y"]
            
            filtered_regions["area"] = filtered_regions["area"].fillna(0)
    return filtered_regions

@logger.catch
def colony_grid_table(
    image: np.ndarray,
    config: Configuration,
    image_notes: tuple | None = None
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Create a grid table of colonies."""

    # Binarize image
    binary_image = binarize_image(
        image,
        gray_method=config.tetrad_gray_method,
        sigma=config.tetrad_gaussian_sigma
    )

    # Detect colonies
    detected_regions = detect_colonies(
        binary_image
    )

    # Filter colonies
    filtered_regions = filter_colonies(
        binary_image,
        detected_regions,
        config,
        is_marker=False
    )

    grid = filtered_regions[["grid_point_x", "grid_point_y"]].to_numpy().reshape(config.expected_rows, config.expected_cols, 2)

    return binary_image, filtered_regions, grid

@logger.catch
def marker_plate_point_matching(
    colony_regions: pd.DataFrame,
    marker_plate_image: np.ndarray,
    config: Configuration,
    image_notes: tuple | None = None
) -> tuple[np.ndarray | None, pd.DataFrame, pd.DataFrame, float, float, float, float, list[np.ndarray], list[np.ndarray]]:
    """Match marker points to plate points using the Hungarian algorithm."""
    # Detect marker colonies
    gray = convert_to_grayscale(marker_plate_image, channel=config.hyg_gray_channel)
    bg_removed = remove_background(gray, threshold_percentile=40)
    blur = gaussian(bg_removed, sigma=2)
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    blur_clahe = equalize_adapthist(blur, clip_limit=0.01)
    blur = (blur_clahe * 255).astype(np.uint8)
    marker_binary = blur > max(threshold_otsu(blur[blur < 100]), 30)
    marker_dilate = apply_morphology(marker_binary, disk_size=3, operation="dilate")
    marker_fill = ndimage.binary_fill_holes(marker_dilate)
    if marker_fill is None:
        marker_fill = marker_dilate
    marker_erode = apply_morphology(marker_fill, disk_size=3, operation="erode")
    marker_regions = detect_colonies(marker_erode, segmentation=True, min_distance=config.hyg_segmentation_min_distance)
    marker_regions = filter_colonies(
        marker_erode,
        marker_regions,
        config,
        is_marker=True
    )
    # marker_centroids = marker_regions[["centroid_x", "centroid_y"]].dropna().to_numpy()
    # tetrad_centroids = colony_regions[["centroid_x", "centroid_y"]].dropna().to_numpy()
    # tetrad_centroids_indices = colony_regions[["centroid_x", "centroid_y"]].dropna().index
    marker_centroids = marker_regions[["grid_point_x", "grid_point_y"]].dropna().to_numpy()
    tetrad_centroids = colony_regions[["grid_point_x", "grid_point_y"]].dropna().to_numpy()
    tetrad_centroids_indices = colony_regions[["grid_point_x", "grid_point_y"]].dropna().index
    
    if len(marker_centroids) == 0 or len(tetrad_centroids) == 0:
        logger.error(f"*** {' '.join(map(str, image_notes)) if image_notes else ''}: No centroids detected in marker or tetrad images for matching.")
        return None, colony_regions, 1.0, 0.0, 0.0, 0.0, [], []

    # Match marker centroids to tetrad centroids
    center_marker = np.mean(marker_centroids, axis=0)
    center_tetrad = np.mean(tetrad_centroids, axis=0)
    shift_vector = center_tetrad - center_marker
    marker_centroids_aligned = marker_centroids + shift_vector

    dist_matrix = cdist(marker_centroids_aligned, tetrad_centroids)
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

    return marker_aligned, colony_regions, marker_regions, scale, angle, tx, ty, matched_tetrad_centroids, matched_marker_centroids

@logger.catch
def genotyping(
    binary_tetrad_image: np.ndarray,
    aligned_marker_image_gray: np.ndarray,
    colony_regions: pd.DataFrame,
    radius: int = 15,
    image_notes: tuple | None = None
) -> pd.DataFrame:
    """Genotype colonies based on tetrad binary image."""
    
    # Calculate background as 15th percentile of non-colony regions
    background_signal = np.percentile(aligned_marker_image_gray[aligned_marker_image_gray>0], 15)
    # logger.info(f"Background signal estimated: {background_signal:.2f}")
    
    # Subtract background and clip negative values
    aligned_marker_image_gray = np.clip(aligned_marker_image_gray.astype(float) - background_signal, 0, 255).astype(np.uint8)

    for idx, region in colony_regions.iterrows():
        cx, cy = int(region["grid_point_x"]), int(region["grid_point_y"])
        x1, x2 = max(0, cx - radius), min(binary_tetrad_image.shape[1], cx + radius)
        y1, y2 = max(0, cy - radius), min(binary_tetrad_image.shape[0], cy + radius)
        tetrad_patch = binary_tetrad_image[y1:y2, x1:x2]
        marker_patch = aligned_marker_image_gray[y1:y2, x1:x2]
        if marker_patch.size == 0:
            logger.warning(f"*** {' '.join(map(str, image_notes)) if image_notes else ''}: Empty marker patch for colony at index {idx}. Skipping genotype assignment.")
            colony_regions.loc[idx, "genotype"] = "Unknown"
            continue
        mean_tetrad_intensity = np.mean(tetrad_patch)
        mean_marker_intensity = np.mean(marker_patch)
        median_tetrad_intensity = np.median(tetrad_patch)
        median_marker_intensity = np.median(marker_patch)
        quantile_80_marker_intensity = np.percentile(marker_patch, 80)
        otsu_threshold = threshold_otsu(aligned_marker_image_gray)
        positive_signal_median = np.median(aligned_marker_image_gray[aligned_marker_image_gray > otsu_threshold])

        marker_intensity_threshold = max(50, otsu_threshold)
        # logger.debug(f"Using marker intensity threshold: {marker_intensity_threshold}")
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
    marker_plate_image: np.ndarray,
    marker_regions: pd.DataFrame,
    radius: int = 15
):
    """Plot genotyping results with colony annotations."""
    n_days = len(tetrad_results)
    last_day = max(tetrad_results.keys())
    n_cols = n_days + 1 + 2 + 1 + 1 # +1 for marker plate, +1 for colony area plot

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
    
    axes[-5].imshow(marker_plate_image, rasterized=True)
    axes[-5].set_title("Original Marker Plate")
    axes[-5].axis('off')
    for idx, region in marker_regions.iterrows():
        cx, cy = region["grid_point_x"], region["grid_point_y"]
        square = Rectangle((cx - radius, cy - radius), 2*radius, 2*radius, edgecolor='blue', facecolor='none', linewidth=2, alpha=0.4)
        axes[-5].add_patch(square)

    axes[-4].imshow(aligned_marker_image, rasterized=True)
    axes[-4].set_title("Aligned Marker Plate")
    axes[-4].axis('off')
    for idx, region in colony_regions.iterrows():
        cx, cy = region[f"grid_point_x_day{last_day}"], region[f"grid_point_y_day{last_day}"]
        genotype = region["genotype"]
        color = 'green' if genotype == "WT" else 'red'
        square = Rectangle((cx - radius, cy - radius), 2*radius, 2*radius, edgecolor=color, facecolor='none', linewidth=2, alpha=0.4)
        axes[-4].add_patch(square)

    # remove background for better visualization
    aligned_marker_image_gray = convert_to_grayscale(aligned_marker_image, channel=0)
    background_signal = np.percentile(aligned_marker_image_gray[aligned_marker_image_gray>0], 15)
    aligned_marker_image_gray2 = np.clip(aligned_marker_image_gray.astype(float) - background_signal, 0, 255).astype(np.uint8)
    axes[-3].imshow(aligned_marker_image_gray2, cmap='gray', rasterized=True)
    axes[-3].set_title("Background Subtracted Marker Plate")
    axes[-3].axis('off')

    pixel_bins = np.arange(0, 256, 1)
    axes[-2].hist(aligned_marker_image_gray.ravel(), bins=pixel_bins, color='blue', alpha=0.7)
    axes[-2].hist(aligned_marker_image_gray2.ravel(), bins=pixel_bins, color='red', alpha=0.7)
    otsu_thresh = threshold_otsu(aligned_marker_image_gray)
    otsu_thresh2 = threshold_otsu(aligned_marker_image_gray2)
    axes[-2].axvline(otsu_thresh, color='darkblue', linestyle='--', label='Otsu Threshold')
    axes[-2].axvline(otsu_thresh2, color='darkred', linestyle='--', label='Otsu Threshold (BG Subtracted)')
    axes[-2].set_title("Marker Plate Intensity Histogram")
    axes[-2].set_xlabel("Intensity")
    axes[-2].set_ylabel("Frequency")
    axes[-2].set_xlim(0, 255)


    # Colony area plot
    ax_area = axes[-1]
    colony_regions = colony_regions.groupby("col").filter(lambda x: x.query("genotype == 'WT'").shape[0]/x.shape[0] == 0.5)
    area_table = colony_regions.set_index("genotype", append=True).filter(like="area_day")
    area_table["area_day0"] = 0
    area_table = area_table.rename_axis("day", axis=1).stack().reset_index().rename(columns={0: "area"})
    area_table["day_num"] = area_table["day"].str.extract(r'day(\d+)').astype(int)
    # area_table = area_table.groupby(["col", "day"]).filter(lambda x: x.query("genotype == 'WT'").shape[0]/x.shape[0] == 0.5)
    last_day_WT_colonies_area_mean = area_table.query("genotype == 'WT' and day_num == @last_day")["area"].median()
    area_table["area[normalized]"] = area_table["area"] / last_day_WT_colonies_area_mean
    area_table = area_table.query("genotype in ['WT', 'Deletion']")
    sns.lineplot(x="day_num", y="area[normalized]", hue="genotype", data=area_table, ax=ax_area, palette={"WT": "green", "Deletion": "red"}, errorbar=("pi", 50), estimator="median")
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
            config,
            image_notes=day_colonies[day]["image_notes"]
        )
        if day == last_day and last_day_binary is None and last_day_colony_regions is None:
            last_day_binary = day_colonies[day]["binary_image"]
            last_day_colony_regions = day_colonies[day]["table"]
        day_colonies[day]["table"] = day_colonies[day]["table"].add_suffix(f"_day{day}")
    
    if last_day_binary is None or last_day_colony_regions is None:
        raise ValueError("No tetrad images were processed.")

    marker_plate_image = io.imread(config.marker_image_path)
    marker_aligned, colony_regions, marker_regions, scale, angle, tx, ty, matched_tetrad_centroids, matched_marker_centroids = marker_plate_point_matching(
        last_day_colony_regions,
        marker_plate_image,
        config,
        image_notes=image_info
    )

    if marker_aligned is None or len(matched_tetrad_centroids) < 3 or marker_aligned.size == 0:
        logger.error(f"*** {' '.join(map(str, image_info)) if image_info else ''}: Marker plate alignment failed. No aligned marker image available.")
        marker_aligned = marker_plate_image

    marker_aligned_gray = convert_to_grayscale(marker_aligned, channel=config.hyg_gray_channel)
    genotyping_colony_regions = genotyping(last_day_binary, marker_aligned_gray, last_day_colony_regions, radius=config.signal_detection_radius, image_notes=image_info)
    all_colony_regions = pd.concat(
        [day_colonies[day]["table"] for day in sorted(day_colonies.keys())] + [genotyping_colony_regions[["genotype", "tetrad_intensity", "marker_intensity", "median_tetrad_intensity", "median_marker_intensity", "otsu_threshold", "positive_signal_median"]]],
        axis=1
    ).sort_index(level=[1,0], axis=0)
    fig = plot_genotype_results(day_colonies, marker_aligned, all_colony_regions, marker_plate_image, marker_regions, radius=config.signal_detection_radius)
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

# tetrad_image_paths={
#     3: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/7th_round/3d/99_any1_3d_#3_202411.cropped.png"),
#     4: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/7th_round/4d/99_any1_4d_#3_202411.cropped.png"),
#     5: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/7th_round/5d/99_any1_5d_#3_202411.cropped.png"),
#     6: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/7th_round/6d/99_any1_6d_#3_202411.cropped.png")
# }
# marker_image_path=Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/7th_round/replica/99_any1_HYG_#3_202411.cropped.png")

# # tetrad_image_paths={
# #     3: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/6th_round/3d/88_rpl1801_3d_#3_202411.cropped.png"),
# #     4: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/6th_round/4d/88_rpl1801_4d_#3_202411.cropped.png"),
# #     5: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/6th_round/5d/88_rpl1801_5d_#3_202411.cropped.png"),
# #     6: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/6th_round/6d/88_rpl1801_6d_#3_202411.cropped.png")
# # }
# # marker_image_path=Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/6th_round/replica/88_rpl1801_HYG_#3_202411.cropped.png")

# config = Configuration(
#     tetrad_image_paths=tetrad_image_paths,
#     marker_image_path=marker_image_path
# )

# last_day = max(config.tetrad_image_paths.keys())
# day_colonies = {}
# last_day_binary = None
# last_day_colony_regions = None

# # %%
# for day, day_image_path in config.tetrad_image_paths.items():
#     day_colonies[day] = {}
#     image = io.imread(day_image_path)
#     day_colonies[day]["image"] = image
#     day_colonies[day]["image_notes"] = (day_image_path.stem, day)
#     day_colonies[day]["binary_image"], day_colonies[day]["table"], day_colonies[day]["grids"] = colony_grid_table(
#         image,
#         config,
#         day_colonies[day]["image_notes"]
#     )
#     if day == last_day and last_day_binary is None and last_day_colony_regions is None:
#         last_day_binary = day_colonies[day]["binary_image"]
#         last_day_colony_regions = day_colonies[day]["table"]
#     day_colonies[day]["table"] = day_colonies[day]["table"].add_suffix(f"_day{day}")

# if last_day_binary is None or last_day_colony_regions is None:
#     raise ValueError("No tetrad images were processed.")

# # %%
# marker_plate_image = io.imread(config.marker_image_path)
# marker_aligned, colony_regions, scale, angle, tx, ty, matched_tetrad_centroids, matched_marker_centroids = marker_plate_point_matching(
#     last_day_colony_regions,
#     marker_plate_image,
#     config
# )

# if marker_aligned is None or len(matched_tetrad_centroids) < 3 or marker_aligned.size == 0:
#     logger.error("Marker plate alignment failed. No aligned marker image available.")
#     marker_aligned = marker_plate_image

# marker_aligned_gray = convert_to_grayscale(marker_aligned, channel=config.hyg_gray_channel)
# # %%
# genotyping_colony_regions = genotyping(last_day_binary, marker_aligned_gray, last_day_colony_regions, radius=config.signal_detection_radius)

# all_colony_regions = pd.concat(
#     [day_colonies[day]["table"] for day in sorted(day_colonies.keys())] + [genotyping_colony_regions[["genotype", "tetrad_intensity", "marker_intensity", "median_tetrad_intensity", "median_marker_intensity", "otsu_threshold", "positive_signal_median"]]],
#     axis=1
# ).sort_index(level=[1,0], axis=0)
# fig = plot_genotype_results(day_colonies, marker_aligned, all_colony_regions, radius=config.signal_detection_radius)

# # %%
# image = io.imread(tetrad_image_paths[3])

# binary_image = binarize_image(
#         image,
#         gray_method=config.tetrad_gray_method,
#         sigma=config.tetrad_gaussian_sigma
# )

# # Detect colonies
# detected_regions = detect_colonies(
#     binary_image
# )
# %%
