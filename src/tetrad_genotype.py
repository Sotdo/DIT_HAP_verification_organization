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
    average_x_spacing: float = 57.0 # averaged x spacing in pixels for 4x12 grid from pilot experiments
    average_y_spacing: float = 60.0 # averaged y spacing in pixels for 4x12 grid from pilot experiments
    spacing_tolerance: float = 8.0  # tolerance for spacing deviation in pixels
    
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
def solve_grid_dataframe(
    df: pd.DataFrame, 
    approx_spacing: float, 
    match_tolerance: float = 0.25, 
    img_coords: bool = True,
    expected_rows: int = 4,
    expected_cols: int = 12,
    x_spacing_ref: float | None = None,
    y_spacing_ref: float | None = None,
    spacing_tolerance: float = 10.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Robust grid restoration for tetrad plate colony detection.
    
    Uses multiple strategies to handle missing colonies:
    1. RANSAC with multiple candidate basis vectors
    2. Validation using prior spacing knowledge
    3. Robust row/column direction determination using physical coordinates
    4. Outlier rejection and re-fitting
    5. Image boundary constraints to prevent grid offset
    
    Input:
        df: DataFrame containing 'centroid_x', 'centroid_y'
        approx_spacing: approximate spacing between points
        match_tolerance: tolerance for grid alignment (relative to spacing)
        img_coords: True for image coordinate system (Y increases downward)
        expected_rows: expected number of rows (default 4)
        expected_cols: expected number of columns (default 12)
        x_spacing_ref: reference x spacing for validation
        y_spacing_ref: reference y spacing for validation  
        spacing_tolerance: tolerance for spacing validation in pixels
    
    Output:
        tagged_df: Original DataFrame with row, col labels
        ideal_grid: Perfect grid point coordinates
    """
    points = df[['centroid_x', 'centroid_y']].values
    n_points = len(points)
    
    if n_points < 4:
        logger.warning("Not enough points for grid detection (need at least 4)")
        return df, pd.DataFrame()
    
    # Get image bounds from points for later validation
    pts_x_min, pts_x_max = points[:, 0].min(), points[:, 0].max()
    pts_y_min, pts_y_max = points[:, 1].min(), points[:, 1].max()
    
    # ---------------- 1. Find candidate basis vectors ----------------
    candidates = []
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            vec = points[j] - points[i]
            dist = np.linalg.norm(vec)
            
            # Filter by approximate spacing (allow single or double spacing)
            for multiplier in [1, 2]:
                expected_dist = approx_spacing * multiplier
                if abs(dist - expected_dist) < approx_spacing * 0.25:
                    norm_vec = vec / multiplier
                    candidates.append((points[i], norm_vec, i, j))
                    break
    
    if len(candidates) == 0:
        logger.warning("No valid spacing candidates found")
        return df, pd.DataFrame()
    
    # ---------------- 2. RANSAC with multiple strategies ----------------
    best_score = -1
    best_result = None
    
    # Adaptive iterations based on number of points
    iterations = min(3000, len(candidates) * 30) if n_points > 6 else 150
    np.random.seed(42)
    
    # Early termination threshold
    early_stop_score = expected_rows * expected_cols * 0.95
    
    for iteration in range(iterations):
        cand_idx = np.random.randint(len(candidates))
        origin, u_vec, _, _ = candidates[cand_idx]
        
        v_vec = np.array([-u_vec[1], u_vec[0]], dtype=u_vec.dtype)
        basis = np.column_stack((u_vec, v_vec))
        
        # Check determinant before inversion (faster than try-catch)
        det = u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0]
        if abs(det) < 1e-10:
            continue
        
        basis_inv = np.linalg.inv(basis)
        
        rel_pos = points - origin
        coords = rel_pos @ basis_inv.T
        coords_round = np.round(coords).astype(int)
        # Faster squared norm calculation
        residuals = np.sqrt(np.sum((coords - coords_round)**2, axis=1))
        
        inlier_mask = residuals < match_tolerance
        n_inliers = np.sum(inlier_mask)
        
        if n_inliers < 4:
            continue
        
        inlier_indices = np.where(inlier_mask)[0]
        inlier_coords = coords_round[inlier_mask]
        
        # Try both orientations
        for is_transposed in [False, True]:
            if is_transposed:
                logical_cols = inlier_coords[:, 1]
                logical_rows = inlier_coords[:, 0]
            else:
                logical_cols = inlier_coords[:, 0]
                logical_rows = inlier_coords[:, 1]
            
            col_min, col_max = logical_cols.min(), logical_cols.max()
            row_min, row_max = logical_rows.min(), logical_rows.max()
            
            # Search for the best window position
            # Expand search range by ±1 to allow for grid shifts
            best_window_score = -1
            best_window_offset = (0, 0)
            
            for c_start in range(col_min - expected_cols, col_max + 2):
                for r_start in range(row_min - expected_rows, row_max + 2):
                    in_window = (
                        (logical_cols >= c_start) & (logical_cols < c_start + expected_cols) &
                        (logical_rows >= r_start) & (logical_rows < r_start + expected_rows)
                    )
                    window_score = np.sum(in_window)
                    
                    if window_score > best_window_score:
                        best_window_score = window_score
                        best_window_offset = (c_start, r_start)
            
            c_start, r_start = best_window_offset
            in_window = (
                (logical_cols >= c_start) & (logical_cols < c_start + expected_cols) &
                (logical_rows >= r_start) & (logical_rows < r_start + expected_rows)
            )
            
            if np.sum(in_window) < 4:
                continue
                
            window_cols = logical_cols[in_window] - c_start
            window_rows = logical_rows[in_window] - r_start
            
            n_unique_cols = len(np.unique(window_cols))
            n_unique_rows = len(np.unique(window_rows))
            
            # Score with coverage bonus
            coverage_bonus = (n_unique_cols / expected_cols + n_unique_rows / expected_rows) * 5
            score = np.sum(in_window) + coverage_bonus
            
            if score > best_score:
                best_score = score
                best_result = {
                    'origin': origin,
                    'basis_inv': basis_inv,
                    'u_vec': u_vec,
                    'v_vec': v_vec,
                    'is_transposed': is_transposed,
                    'inlier_indices': inlier_indices,
                    'inlier_coords': inlier_coords,
                    'window_offset': best_window_offset,
                    'in_window_mask': in_window
                }
                # Early termination if we found a very good fit
                if best_score > early_stop_score:
                    break
        
        # Also break outer loop if early stop achieved
        if best_score > early_stop_score:
            break
    
    if best_result is None:
        logger.warning("Failed to detect grid structure")
        return df, pd.DataFrame()
    
    # ---------------- 3. Extract mapping ----------------
    inlier_indices = best_result['inlier_indices']
    inlier_coords = best_result['inlier_coords']
    is_transposed = best_result['is_transposed']
    c_start, r_start = best_result['window_offset']
    in_window = best_result['in_window_mask']
    
    if is_transposed:
        logical_cols = inlier_coords[:, 1]
        logical_rows = inlier_coords[:, 0]
    else:
        logical_cols = inlier_coords[:, 0]
        logical_rows = inlier_coords[:, 1]
    
    temp_mapping = {}
    valid_indices = []
    valid_phys_pts = []
    
    for i, (idx, in_win) in enumerate(zip(inlier_indices, in_window)):
        if in_win:
            col = int(logical_cols[i] - c_start)
            row = int(logical_rows[i] - r_start)
            temp_mapping[idx] = [col, row]
            valid_indices.append(idx)
            valid_phys_pts.append(points[idx])
    
    valid_indices = np.array(valid_indices)
    valid_phys_pts = np.array(valid_phys_pts)
    
    # ---------------- 4. Direction correction using PHYSICAL coordinates ----------------
    # This is key: use actual physical positions to determine correct orientation
    
    current_cols = np.array([temp_mapping[i][0] for i in valid_indices])
    current_rows = np.array([temp_mapping[i][1] for i in valid_indices])
    
    # A. Determine row direction by comparing physical Y of different logical rows
    # For each logical row, calculate mean physical Y
    row_to_y = {}
    for r in range(expected_rows):
        mask = (current_rows == r)
        if np.any(mask):
            row_to_y[r] = np.median(valid_phys_pts[mask, 1])
    
    # Check if we need to flip rows
    # In image coords: row 0 should have SMALLEST Y (top of image)
    if len(row_to_y) >= 2:
        rows_sorted_by_y = sorted(row_to_y.keys(), key=lambda r: row_to_y[r])
        
        # If the first row (smallest Y) is not row 0, we need to flip
        if img_coords:
            # row 0 should be at top (smallest Y)
            should_flip_row = (rows_sorted_by_y[0] != 0)
        else:
            # row 0 should be at bottom (largest Y) 
            should_flip_row = (rows_sorted_by_y[-1] != 0)
        
        if should_flip_row:
            for idx in valid_indices:
                temp_mapping[idx][1] = (expected_rows - 1) - temp_mapping[idx][1]
    
    # B. Determine column direction by comparing physical X
    col_to_x = {}
    for c in range(expected_cols):
        mask = (current_cols == c)
        if np.any(mask):
            col_to_x[c] = np.median(valid_phys_pts[mask, 0])
    
    if len(col_to_x) >= 2:
        cols_sorted_by_x = sorted(col_to_x.keys(), key=lambda c: col_to_x[c])
        # col 0 should always have smallest X (leftmost)
        should_flip_col = (cols_sorted_by_x[0] != 0)
        
        if should_flip_col:
            for idx in valid_indices:
                temp_mapping[idx][0] = (expected_cols - 1) - temp_mapping[idx][0]
    
    # ---------------- 5. Least squares fitting ----------------
    df['row'] = np.nan
    df['col'] = np.nan
    
    A, b = [], []
    for idx, (c, r) in temp_mapping.items():
        df.at[df.index[idx], 'row'] = r
        df.at[df.index[idx], 'col'] = c
        A.append([1, 0, c, 0, r, 0])
        b.append(points[idx][0])
        A.append([0, 1, 0, c, 0, r])
        b.append(points[idx][1])
    
    A = np.array(A)
    b = np.array(b)
    
    if len(A) == 0:
        return df, pd.DataFrame()
    
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Calculate residuals and reject outliers
    predicted = A @ params
    point_residuals = []
    for i in range(0, len(b), 2):
        res = np.sqrt((b[i] - predicted[i])**2 + (b[i+1] - predicted[i+1])**2)
        point_residuals.append(res)
    point_residuals = np.array(point_residuals)
    
    if len(point_residuals) > 6:
        residual_threshold = np.percentile(point_residuals, 90) * 1.5
        residual_threshold = max(residual_threshold, approx_spacing * 0.3)
        
        A_clean, b_clean = [], []
        outlier_indices = []
        for i, (idx, (c, r)) in enumerate(temp_mapping.items()):
            if point_residuals[i] < residual_threshold:
                A_clean.append([1, 0, c, 0, r, 0])
                b_clean.append(points[idx][0])
                A_clean.append([0, 1, 0, c, 0, r])
                b_clean.append(points[idx][1])
            else:
                # Don't discard outliers yet, just track them
                outlier_indices.append((idx, c, r))
        
        if len(A_clean) >= 8:
            A = np.array(A_clean)
            b = np.array(b_clean)
            params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    O_fit = params[0:2]
    u_fit = params[2:4]  # col vector
    v_fit = params[4:6]  # row vector
    
    # ---------------- 6. Validate and adjust grid position ----------------
    # Check if the fitted grid covers the observed points properly
    # The grid should not extend too far outside the observed point range
    
    grid_corners = [
        O_fit,  # (row=0, col=0)
        O_fit + (expected_cols - 1) * u_fit,  # (row=0, col=11)
        O_fit + (expected_rows - 1) * v_fit,  # (row=3, col=0)
        O_fit + (expected_cols - 1) * u_fit + (expected_rows - 1) * v_fit  # (row=3, col=11)
    ]
    
    grid_x_min = min(c[0] for c in grid_corners)
    grid_x_max = max(c[0] for c in grid_corners)
    grid_y_min = min(c[1] for c in grid_corners)
    grid_y_max = max(c[1] for c in grid_corners)
    
    # Check if grid is shifted - the grid should roughly match the observed points extent
    margin = approx_spacing * 0.5
    
    # If grid minimum is much smaller than point minimum, grid might be shifted
    x_shift_needed = 0
    y_shift_needed = 0
    
    if grid_x_min < pts_x_min - margin:
        x_shift_needed = pts_x_min - grid_x_min - margin * 0.5
    if grid_y_min < pts_y_min - margin:
        y_shift_needed = pts_y_min - grid_y_min - margin * 0.5
        
    # Apply shift if needed (this adjusts O_fit)
    if abs(x_shift_needed) > margin or abs(y_shift_needed) > margin:
        logger.debug(f"Adjusting grid origin by ({x_shift_needed:.1f}, {y_shift_needed:.1f})")
        O_fit = O_fit + np.array([x_shift_needed, y_shift_needed])
    
    # ---------------- 7. Validate spacing ----------------
    u_spacing = np.linalg.norm(u_fit)
    v_spacing = np.linalg.norm(v_fit)
    
    if x_spacing_ref is not None and abs(u_spacing - x_spacing_ref) > spacing_tolerance:
        logger.warning(f"X spacing mismatch: {u_spacing:.2f} vs expected {x_spacing_ref:.2f}")
    if y_spacing_ref is not None and abs(v_spacing - y_spacing_ref) > spacing_tolerance:
        logger.warning(f"Y spacing mismatch: {v_spacing:.2f} vs expected {y_spacing_ref:.2f}")
    
    # ---------------- 8. Generate ideal grid ----------------
    ideal_data = []
    for r in range(expected_rows):
        for c in range(expected_cols):
            pt = O_fit + c * u_fit + r * v_fit
            ideal_data.append({
                'row': r, 
                'col': c, 
                'expected_x': pt[0], 
                'expected_y': pt[1]
            })
    ideal_df = pd.DataFrame(ideal_data)
    
    # ---------------- 8.5 Detect and correct systematic column offset ----------------
    # Check if there's a systematic offset suggesting the grid is shifted by one column
    # This can happen when edge colonies are missing, causing RANSAC to mis-align
    
    assigned_mask = ~df['row'].isna() & ~df['col'].isna()
    if assigned_mask.sum() >= 6:
        assigned_indices = df.index[assigned_mask].tolist()
        
        # Calculate the offset from each point to its assigned grid position
        x_offsets = []
        for idx in assigned_indices:
            pt_x = df.at[idx, 'centroid_x']
            r, c = int(df.at[idx, 'row']), int(df.at[idx, 'col'])
            expected_x = O_fit[0] + c * u_fit[0] + r * v_fit[0]
            x_offsets.append(pt_x - expected_x)
        
        x_offsets = np.array(x_offsets)
        median_x_offset = np.median(x_offsets)
        u_spacing = np.linalg.norm(u_fit)
        
        # If median offset is close to ±1 column spacing, the grid is likely shifted
        # Use 0.3 * spacing as threshold (more sensitive to catch subtle shifts)
        col_shift = 0
        if median_x_offset > 0.3 * u_spacing:
            # Points are systematically to the RIGHT of their grid positions
            # This means grid is shifted too far LEFT, need to shift columns by -1
            col_shift = -1
        elif median_x_offset < -0.3 * u_spacing:
            # Points are systematically to the LEFT of their grid positions  
            # This means grid is shifted too far RIGHT, need to shift columns by +1
            col_shift = 1
        
        if col_shift != 0:
            logger.info(f"Detected systematic column offset (median_x_offset={median_x_offset:.2f}, u_spacing={u_spacing:.2f}), shifting columns by {col_shift}")
            
            # Adjust the origin to compensate
            O_fit = O_fit - col_shift * u_fit
            
            # Regenerate ideal grid with corrected origin
            ideal_data = []
            for r in range(expected_rows):
                for c in range(expected_cols):
                    pt = O_fit + c * u_fit + r * v_fit
                    ideal_data.append({
                        'row': r, 
                        'col': c, 
                        'expected_x': pt[0], 
                        'expected_y': pt[1]
                    })
            ideal_df = pd.DataFrame(ideal_data)
            
            # Re-assign all points to nearest grid position after correction
            ideal_grid = np.column_stack([ideal_df['expected_x'].values, ideal_df['expected_y'].values])
            
            for idx in df.index:
                pt = np.array([df.at[idx, 'centroid_x'], df.at[idx, 'centroid_y']])
                distances = np.linalg.norm(ideal_grid - pt, axis=1)
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                
                if min_dist < approx_spacing * 1.5:
                    df.at[idx, 'row'] = ideal_df.at[min_idx, 'row']
                    df.at[idx, 'col'] = ideal_df.at[min_idx, 'col']
                else:
                    df.at[idx, 'row'] = np.nan
                    df.at[idx, 'col'] = np.nan
    
    # ---------------- 8.6 Detect missing column among unassigned points ----------------
    # Check if there's a column of regular points that were missed
    # Lowered threshold to detect even partially missing columns (>= 2 points)
    unassigned_mask = df['row'].isna()
    if unassigned_mask.sum() >= 2:
        unassigned_indices = df.index[unassigned_mask].tolist()
        unassigned_points = np.array([[df.at[idx, 'centroid_x'], df.at[idx, 'centroid_y']] 
                                      for idx in unassigned_indices])
        
        # Cluster unassigned points by X coordinate (column direction)
        if len(unassigned_points) >= 2:
            x_coords = unassigned_points[:, 0]
            # Use u_spacing to group points into potential columns
            x_sorted_indices = np.argsort(x_coords)
            
            # Find columns with at least 2 points (lowered from expected_rows)
            potential_columns = []
            i = 0
            while i < len(x_sorted_indices):
                col_group = [x_sorted_indices[i]]
                j = i + 1
                while j < len(x_sorted_indices) and \
                      abs(x_coords[x_sorted_indices[j]] - x_coords[x_sorted_indices[i]]) < u_spacing * 0.4:
                    col_group.append(x_sorted_indices[j])
                    j += 1
                
                if len(col_group) >= 2:
                    # Check if Y coordinates are regularly spaced
                    y_coords = unassigned_points[col_group, 1]
                    y_sorted = np.sort(y_coords)
                    y_diffs = np.diff(y_sorted)
                    
                    # If Y spacing is regular, this is likely a missed column
                    is_regular = False
                    if len(col_group) >= 3:
                        # For 3+ points, check if each interval is close to n * v_spacing (n = 1, 2, or 3)
                        # This handles non-consecutive rows better than standard deviation
                        all_valid = True
                        for diff in y_diffs:
                            # Check if this diff matches any multiplier of v_spacing
                            valid_diff = False
                            for multiplier in [1, 2, 3]:
                                if abs(diff - multiplier * v_spacing) < v_spacing * 0.4:
                                    valid_diff = True
                                    break
                            if not valid_diff:
                                all_valid = False
                                break
                        is_regular = all_valid
                    elif len(col_group) == 2:
                        # For 2 points, check if distance is close to n * v_spacing (n = 1, 2, or 3)
                        # This handles cases where points are not in adjacent rows
                        dist = y_diffs[0]
                        for multiplier in [1, 2, 3]:
                            if abs(dist - multiplier * v_spacing) < v_spacing * 0.4:
                                is_regular = True
                                break
                    
                    if is_regular:
                        potential_columns.append(col_group)
                        logger.info(f"Found potential missed column with {len(col_group)} regular points")
                
                i = j
            
            # If we found a regular column, the grid is likely shifted
            if potential_columns:
                # Calculate the X position of this missed column
                missed_col_x = np.median([x_coords[idx] for idx in potential_columns[0]])
                
                # Determine which direction to shift
                # Compare with the leftmost and rightmost expected columns
                leftmost_expected_x = O_fit[0]
                rightmost_expected_x = O_fit[0] + (expected_cols - 1) * u_fit[0]
                
                col_shift = 0
                if missed_col_x < leftmost_expected_x - u_spacing * 0.3:
                    # Missed column is to the left, shift grid left
                    col_shift = 1
                elif missed_col_x > rightmost_expected_x + u_spacing * 0.3:
                    # Missed column is to the right, shift grid right
                    col_shift = -1
                
                if col_shift != 0:
                    logger.info(f"Detected missed regular column (x={missed_col_x:.1f}), shifting grid by {col_shift} columns")
                    
                    # Adjust the origin
                    O_fit = O_fit - col_shift * u_fit
                    
                    # Regenerate ideal grid
                    ideal_data = []
                    for r in range(expected_rows):
                        for c in range(expected_cols):
                            pt = O_fit + c * u_fit + r * v_fit
                            ideal_data.append({
                                'row': r, 
                                'col': c, 
                                'expected_x': pt[0], 
                                'expected_y': pt[1]
                            })
                    ideal_df = pd.DataFrame(ideal_data)
                    
                    # Re-assign all points
                    ideal_grid = np.column_stack([ideal_df['expected_x'].values, ideal_df['expected_y'].values])
                    for idx in df.index:
                        pt = np.array([df.at[idx, 'centroid_x'], df.at[idx, 'centroid_y']])
                        distances = np.linalg.norm(ideal_grid - pt, axis=1)
                        min_idx = np.argmin(distances)
                        min_dist = distances[min_idx]
                        
                        if min_dist < approx_spacing * 1.5:
                            df.at[idx, 'row'] = ideal_df.at[min_idx, 'row']
                            df.at[idx, 'col'] = ideal_df.at[min_idx, 'col']
                        else:
                            df.at[idx, 'row'] = np.nan
                            df.at[idx, 'col'] = np.nan
    
    # ---------------- 9. Reassign points outside window to nearest grid point ----------------
    # For points that weren't assigned to the grid, find their nearest grid position
    unassigned_mask = df['row'].isna()
    if unassigned_mask.any():
        unassigned_indices = df.index[unassigned_mask].tolist()
        
        # Create ideal grid array for vectorized distance calculation
        ideal_grid = np.column_stack([ideal_df['expected_x'].values, ideal_df['expected_y'].values])
        
        # Calculate distance to all ideal grid points
        for idx in unassigned_indices:
            # Get point coordinates
            pt = np.array([df.at[idx, 'centroid_x'], df.at[idx, 'centroid_y']])
            
            # Vectorized distance calculation
            distances = np.linalg.norm(ideal_grid - pt, axis=1)
            
            # Find nearest grid point
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            nearest_row = ideal_df.at[min_idx, 'row']
            nearest_col = ideal_df.at[min_idx, 'col']
            
            # Only assign if distance is reasonable (within 2x approx_spacing)
            if min_dist < approx_spacing * 2.0:
                df.at[idx, 'row'] = nearest_row
                df.at[idx, 'col'] = nearest_col
                logger.debug(f"Reassigned outlier point (label={df.at[idx, 'label']}) to grid position ({nearest_row}, {nearest_col}), distance: {min_dist:.2f}")
    
    # ---------------- 9.5 Final validation: detect empty columns ----------------
    # Check if there are columns with no assigned points while there are unassigned points
    # This indicates a potential grid shift error
    assigned_mask = ~df['row'].isna() & ~df['col'].isna()
    if assigned_mask.sum() >= 3 and (~assigned_mask).sum() >= 2:
        # Count how many points are assigned to each column
        assigned_cols = df.loc[assigned_mask, 'col'].astype(int)
        col_counts = np.bincount(assigned_cols, minlength=expected_cols)
        
        # Find empty columns (columns with 0 points)
        empty_cols = np.where(col_counts == 0)[0]
        
        if len(empty_cols) > 0:
            # Check if all empty columns are at edges (first or last column)
            edge_empty = all(c == 0 or c == expected_cols - 1 for c in empty_cols)
            
            if edge_empty and len(empty_cols) <= 2:
                # Get unassigned points
                unassigned_indices = df.index[~assigned_mask].tolist()
                unassigned_points = np.array([[df.at[idx, 'centroid_x'], df.at[idx, 'centroid_y']] 
                                              for idx in unassigned_indices])
                
                if len(unassigned_points) >= 2:
                    # Calculate which direction the unassigned points are
                    unassigned_x = np.median(unassigned_points[:, 0])
                    assigned_x = np.median([df.at[idx, 'centroid_x'] for idx in df.index[assigned_mask]])
                    
                    # Determine shift direction
                    col_shift = 0
                    if 0 in empty_cols and unassigned_x < assigned_x:
                        # Empty at left, unassigned points are to the left -> shift right
                        col_shift = 1
                        logger.warning(f"Detected empty column 0 with {len(unassigned_points)} unassigned points to the left, correcting grid")
                    elif (expected_cols - 1) in empty_cols and unassigned_x > assigned_x:
                        # Empty at right, unassigned points are to the right -> shift left
                        col_shift = -1
                        logger.warning(f"Detected empty column {expected_cols-1} with {len(unassigned_points)} unassigned points to the right, correcting grid")
                    
                    if col_shift != 0:
                        # Adjust the origin
                        O_fit = O_fit - col_shift * u_fit
                        
                        # Regenerate ideal grid
                        ideal_data = []
                        for r in range(expected_rows):
                            for c in range(expected_cols):
                                pt = O_fit + c * u_fit + r * v_fit
                                ideal_data.append({
                                    'row': r, 
                                    'col': c, 
                                    'expected_x': pt[0], 
                                    'expected_y': pt[1]
                                })
                        ideal_df = pd.DataFrame(ideal_data)
                        
                        # Re-assign all points
                        ideal_grid = np.column_stack([ideal_df['expected_x'].values, ideal_df['expected_y'].values])
                        for idx in df.index:
                            pt = np.array([df.at[idx, 'centroid_x'], df.at[idx, 'centroid_y']])
                            distances = np.linalg.norm(ideal_grid - pt, axis=1)
                            min_idx = np.argmin(distances)
                            min_dist = distances[min_idx]
                            
                            if min_dist < approx_spacing * 1.5:
                                df.at[idx, 'row'] = ideal_df.at[min_idx, 'row']
                                df.at[idx, 'col'] = ideal_df.at[min_idx, 'col']
                            else:
                                df.at[idx, 'row'] = np.nan
                                df.at[idx, 'col'] = np.nan
                        
                        logger.info(f"Grid corrected: reassigned points after detecting empty edge column")
    
    # ---------------- 9.6 Column density analysis for shift detection ----------------
    # Check for characteristic pattern: edge column empty + adjacent column over-dense
    # This is a strong indicator of 1-column shift
    assigned_mask = ~df['row'].isna() & ~df['col'].isna()
    if assigned_mask.sum() >= 4:
        assigned_cols = df.loc[assigned_mask, 'col'].astype(int)
        col_counts = np.bincount(assigned_cols, minlength=expected_cols)
        
        # Expected average points per column
        total_assigned = assigned_mask.sum()
        expected_per_col = total_assigned / expected_cols
        
        # Check for shift pattern
        col_shift = 0
        
        # Left shift pattern: column 0 empty, column 1 has 2x expected density
        if col_counts[0] == 0 and col_counts[1] >= expected_per_col * 1.5 and total_assigned >= expected_rows * 2:
            # Verify there are unassigned points to the left
            unassigned_indices = df.index[~assigned_mask].tolist()
            if len(unassigned_indices) >= 2:
                unassigned_x = [df.at[idx, 'centroid_x'] for idx in unassigned_indices]
                assigned_x = [df.at[idx, 'centroid_x'] for idx in df.index[assigned_mask]]
                if np.median(unassigned_x) < np.median(assigned_x):
                    col_shift = 1
                    logger.warning(f"Column density pattern detected: col 0 empty ({col_counts[0]}), col 1 over-dense ({col_counts[1]:.1f} vs {expected_per_col:.1f}), shifting right")
        
        # Right shift pattern: last column empty, second-to-last has 2x expected density
        elif col_counts[-1] == 0 and col_counts[-2] >= expected_per_col * 1.5 and total_assigned >= expected_rows * 2:
            # Verify there are unassigned points to the right
            unassigned_indices = df.index[~assigned_mask].tolist()
            if len(unassigned_indices) >= 2:
                unassigned_x = [df.at[idx, 'centroid_x'] for idx in unassigned_indices]
                assigned_x = [df.at[idx, 'centroid_x'] for idx in df.index[assigned_mask]]
                if np.median(unassigned_x) > np.median(assigned_x):
                    col_shift = -1
                    logger.warning(f"Column density pattern detected: col {expected_cols-1} empty ({col_counts[-1]}), col {expected_cols-2} over-dense ({col_counts[-2]:.1f} vs {expected_per_col:.1f}), shifting left")
        
        if col_shift != 0:
            # Adjust the origin
            O_fit = O_fit - col_shift * u_fit
            
            # Regenerate ideal grid
            ideal_data = []
            for r in range(expected_rows):
                for c in range(expected_cols):
                    pt = O_fit + c * u_fit + r * v_fit
                    ideal_data.append({
                        'row': r, 
                        'col': c, 
                        'expected_x': pt[0], 
                        'expected_y': pt[1]
                    })
            ideal_df = pd.DataFrame(ideal_data)
            
            # Re-assign all points
            ideal_grid = np.column_stack([ideal_df['expected_x'].values, ideal_df['expected_y'].values])
            for idx in df.index:
                pt = np.array([df.at[idx, 'centroid_x'], df.at[idx, 'centroid_y']])
                distances = np.linalg.norm(ideal_grid - pt, axis=1)
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                
                if min_dist < approx_spacing * 1.5:
                    df.at[idx, 'row'] = ideal_df.at[min_idx, 'row']
                    df.at[idx, 'col'] = ideal_df.at[min_idx, 'col']
                else:
                    df.at[idx, 'row'] = np.nan
                    df.at[idx, 'col'] = np.nan
            
            logger.info(f"Grid corrected: reassigned points after detecting column density anomaly")
    
    # ---------------- 9.7 Boundary constraint validation ----------------
    # Check if the grid extends too far beyond the actual point range
    # This catches cases where the grid is shifted but all points still get assigned
    assigned_mask = ~df['row'].isna() & ~df['col'].isna()
    if assigned_mask.sum() >= 6:
        # Get the actual X range of assigned points
        assigned_x = df.loc[assigned_mask, 'centroid_x'].values
        actual_x_min = np.min(assigned_x)
        actual_x_max = np.max(assigned_x)
        
        # Get the expected X range from the grid
        expected_x_min = O_fit[0]  # Column 0
        expected_x_max = O_fit[0] + (expected_cols - 1) * u_fit[0]  # Column 11
        
        # Calculate how far the grid extends beyond actual points
        left_overhang = actual_x_min - expected_x_min  # Positive if grid extends left of points
        right_overhang = expected_x_max - actual_x_max  # Positive if grid extends right of points
        
        col_shift = 0
        # If grid extends too far to the left (column 0 is way left of actual points)
        # and not enough to the right, we may need to shift right
        if left_overhang > u_spacing * 0.6 and right_overhang < u_spacing * 0.3:
            col_shift = -1
            logger.warning(f"Boundary validation: grid extends {left_overhang:.1f}px left of points, shifting left")
        # If grid extends too far to the right and not enough to the left
        elif right_overhang > u_spacing * 0.6 and left_overhang < u_spacing * 0.3:
            col_shift = 1
            logger.warning(f"Boundary validation: grid extends {right_overhang:.1f}px right of points, shifting right")
        
        if col_shift != 0:
            # Adjust the origin
            O_fit = O_fit - col_shift * u_fit
            
            # Regenerate ideal grid
            ideal_data = []
            for r in range(expected_rows):
                for c in range(expected_cols):
                    pt = O_fit + c * u_fit + r * v_fit
                    ideal_data.append({
                        'row': r, 
                        'col': c, 
                        'expected_x': pt[0], 
                        'expected_y': pt[1]
                    })
            ideal_df = pd.DataFrame(ideal_data)
            
            # Re-assign all points
            ideal_grid = np.column_stack([ideal_df['expected_x'].values, ideal_df['expected_y'].values])
            for idx in df.index:
                pt = np.array([df.at[idx, 'centroid_x'], df.at[idx, 'centroid_y']])
                distances = np.linalg.norm(ideal_grid - pt, axis=1)
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                
                if min_dist < approx_spacing * 1.5:
                    df.at[idx, 'row'] = ideal_df.at[min_idx, 'row']
                    df.at[idx, 'col'] = ideal_df.at[min_idx, 'col']
                else:
                    df.at[idx, 'row'] = np.nan
                    df.at[idx, 'col'] = np.nan
            
            logger.info(f"Grid corrected: reassigned points after boundary validation")
    
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
            # Average across color channels - using faster np.mean with keepdims=False
            return image.mean(axis=2, dtype=np.float32).astype(image.dtype)
        elif method == 'luminosity':
            # Weighted average for luminosity - using @ operator for faster matrix multiplication
            weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
            return (image[..., :3] @ weights).astype(image.dtype)
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
    # Use np.percentile with linear interpolation for faster computation
    threshold_value = np.percentile(gray_image[gray_image > 0], threshold_percentile, method='linear')
    # Vectorized operation: subtract and clip in one step
    bg_removed = np.clip(gray_image.astype(np.float32) - threshold_value, 0, 255).astype(np.uint8)
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
    
    # fill small holes
    binary = ndi.binary_fill_holes(binary)

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

    # Calculate x and y spacing using vectorized operations
    # Create pairwise difference matrices
    x_diff_matrix = np.abs(centroids[:, np.newaxis, 0] - centroids[np.newaxis, :, 0])
    y_diff_matrix = np.abs(centroids[:, np.newaxis, 1] - centroids[np.newaxis, :, 1])
    
    # Create masks for same row and same column
    same_row_mask = (y_diff_matrix <= 10) & (x_diff_matrix > 30)
    same_col_mask = (x_diff_matrix <= 10) & (y_diff_matrix > 30)
    
    # Extract relevant differences
    x_diffs = x_diff_matrix[same_row_mask]
    y_diffs = y_diff_matrix[same_col_mask]

    # Filter differences
    filtered_x_diffs = x_diffs[(x_diffs > 30) & (x_diffs < 70)]
    filtered_y_diffs = y_diffs[(y_diffs > 30) & (y_diffs < 70)]

    if filtered_x_diffs.size == 0:
        filtered_x_diffs = x_diffs[(x_diffs > 55) & (x_diffs < 135)] / 2
    if filtered_y_diffs.size == 0:
        filtered_y_diffs = y_diffs[(y_diffs > 55) & (y_diffs < 135)] / 2
        
    x_spacing = np.mean(filtered_x_diffs) if filtered_x_diffs.size > 0 else 57.0
    y_spacing = np.mean(filtered_y_diffs) if filtered_y_diffs.size > 0 else 60.0
    
    # Use median of smallest values for robust min calculation
    x_min = centroids[:, 0].min()
    if len(centroids) > 18:
        y_min = np.median(np.partition(centroids[:, 1], 3)[:3])  # Faster than sort
    else:
        y_min = centroids[:, 1].min()

    # Vectorized grid creation
    col_offsets = np.arange(expected_cols) * x_spacing + x_min
    row_offsets = np.arange(expected_rows) * y_spacing + y_min
    fitted_grid = np.stack(np.meshgrid(col_offsets, row_offsets, indexing='xy'), axis=-1)

    # Vectorized colony to grid point assignment
    if x_spacing > 0 and y_spacing > 0:
        col_indices = np.clip(np.round((centroids[:, 0] - x_min) / x_spacing).astype(int), 0, expected_cols - 1)
        row_indices = np.clip(np.round((centroids[:, 1] - y_min) / y_spacing).astype(int), 0, expected_rows - 1)
    else:
        col_indices = np.zeros(len(centroids), dtype=int)
        row_indices = np.zeros(len(centroids), dtype=int)
    
    grid_points = fitted_grid[row_indices, col_indices]
    distances = np.sqrt(np.sum((centroids - grid_points)**2, axis=1))
    
    colony_regions = colony_regions.copy()
    colony_regions["row"] = row_indices
    colony_regions["col"] = col_indices
    colony_regions["grid_point_x"] = grid_points[:, 0]
    colony_regions["grid_point_y"] = grid_points[:, 1]
    colony_regions["distance"] = distances

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
    """Find optimal zoom factor using coarse-to-fine grid search with vectorized operations."""
    
    # Phase 1: Coarse search with larger step
    coarse_step = zoom_step * 5
    x_zoom_coarse = np.arange(zoom_range[0], zoom_range[1] + coarse_step, coarse_step)
    y_zoom_coarse = np.arange(zoom_range[0], zoom_range[1] + coarse_step, coarse_step)
    
    best_error = np.inf
    best_zoom_coarse = None
    
    # Precompute grid points extraction
    grid_points_base = grid_centered[rows, cols]
    
    for x_zoom in x_zoom_coarse:
        for y_zoom in y_zoom_coarse:
            zoom_factor = np.array([x_zoom, y_zoom], dtype=np.float32)
            zoomed_grid = grid_centered * zoom_factor + grid_center_point
            
            # Vectorized spacing check
            if x_spacing_ref is not None and spacing_tolerance is not None:
                grid_x_spacing = np.diff(zoomed_grid, axis=1)[:, :, 0]
                if np.any(np.abs(grid_x_spacing - x_spacing_ref) > spacing_tolerance):
                    continue
            if y_spacing_ref is not None and spacing_tolerance is not None:
                grid_y_spacing = np.diff(zoomed_grid, axis=0)[:, :, 1]
                if np.any(np.abs(grid_y_spacing - y_spacing_ref) > spacing_tolerance):
                    continue
            
            # Vectorized distance calculation
            grid_points = zoomed_grid[rows, cols]
            distances = np.sqrt(np.sum((centroids - grid_points)**2, axis=1))
            error = np.sum(distances)
            
            if error < best_error:
                best_error = error
                best_zoom_coarse = (x_zoom, y_zoom)
    
    if best_zoom_coarse is None:
        return (1.0, 1.0)
    
    # Phase 2: Fine search around best coarse result
    fine_range_x = (max(zoom_range[0], best_zoom_coarse[0] - coarse_step), 
                    min(zoom_range[1], best_zoom_coarse[0] + coarse_step))
    fine_range_y = (max(zoom_range[0], best_zoom_coarse[1] - coarse_step), 
                    min(zoom_range[1], best_zoom_coarse[1] + coarse_step))
    
    x_zoom_fine = np.arange(fine_range_x[0], fine_range_x[1] + zoom_step, zoom_step)
    y_zoom_fine = np.arange(fine_range_y[0], fine_range_y[1] + zoom_step, zoom_step)
    
    for x_zoom in x_zoom_fine:
        for y_zoom in y_zoom_fine:
            zoom_factor = np.array([x_zoom, y_zoom], dtype=np.float32)
            zoomed_grid = grid_centered * zoom_factor + grid_center_point
            
            if x_spacing_ref is not None and spacing_tolerance is not None:
                grid_x_spacing = np.diff(zoomed_grid, axis=1)[:, :, 0]
                if np.any(np.abs(grid_x_spacing - x_spacing_ref) > spacing_tolerance):
                    continue
            if y_spacing_ref is not None and spacing_tolerance is not None:
                grid_y_spacing = np.diff(zoomed_grid, axis=0)[:, :, 1]
                if np.any(np.abs(grid_y_spacing - y_spacing_ref) > spacing_tolerance):
                    continue
            
            grid_points = zoomed_grid[rows, cols]
            distances = np.sqrt(np.sum((centroids - grid_points)**2, axis=1))
            error = np.sum(distances)
            
            if error < best_error:
                best_error = error
                best_zoom_coarse = (x_zoom, y_zoom)
    
    return (float(best_zoom_coarse[0]), float(best_zoom_coarse[1]))

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
    is_marker: bool = False,
    reference_grid: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Filter colonies and fit grid.
    
    Args:
        binary_image: Binary image of colonies
        colony_regions: DataFrame with detected colony regions
        config: Configuration object
        is_marker: Whether this is a marker plate image
        reference_grid: Optional reference grid from tetrad image (used for marker plates)
    """
    
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
    
    region, restored_grid = solve_grid_dataframe(
        filtered_regions, 
        approx_spacing=config.average_x_spacing,
        expected_rows=config.expected_rows,
        expected_cols=config.expected_cols,
        x_spacing_ref=config.average_x_spacing,
        y_spacing_ref=config.average_y_spacing,
        spacing_tolerance=config.spacing_tolerance
    )

    # For marker images: if grid fitting fails badly, use reference grid from tetrad
    if is_marker and reference_grid is not None:
        # Check if current grid is reasonable by comparing coverage
        assigned_count = (~region['row'].isna()).sum() if not region.empty else 0
        total_points = len(filtered_regions)
        coverage = assigned_count / max(total_points, 1)
        
        # If coverage is too low (< 50%), use reference grid directly
        if coverage < 0.5 or region.empty or restored_grid.empty:
            logger.warning(f"Marker grid fitting poor (coverage={coverage:.2f}), using tetrad reference grid")
            # Use reference grid to assign points
            ref_grid_points = reference_grid[['expected_x', 'expected_y']].values
            ref_grid_positions = reference_grid[['row', 'col']].values
            
            for idx in filtered_regions.index:
                pt = np.array([filtered_regions.at[idx, 'centroid_x'], filtered_regions.at[idx, 'centroid_y']])
                distances = np.linalg.norm(ref_grid_points - pt, axis=1)
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                
                # Use larger threshold for marker (colonies may have shifted)
                if min_dist < config.average_x_spacing * 2.5:
                    filtered_regions.at[idx, 'row'] = ref_grid_positions[min_idx, 0]
                    filtered_regions.at[idx, 'col'] = ref_grid_positions[min_idx, 1]
                else:
                    filtered_regions.at[idx, 'row'] = np.nan
                    filtered_regions.at[idx, 'col'] = np.nan
            
            region = filtered_regions
            restored_grid = reference_grid.copy()


    if region.empty or restored_grid.empty:
        logger.warning("*** Grid restoration failed: No colonies detected or grid fitting failed. Filter colonies at the boundaries and try again.")
        left, right = w * 0.015, w * 0.985
        top, bottom = h * 0.015, h * 0.985
        filtered_regions = filtered_regions.query(
            f"centroid_x >= {left} and centroid_x <= {right} and centroid_y >= {top} and centroid_y <= {bottom}"
        ).copy()
        region, restored_grid = solve_grid_dataframe(
            filtered_regions, 
            approx_spacing=config.average_x_spacing,
            expected_rows=config.expected_rows,
            expected_cols=config.expected_cols,
            x_spacing_ref=config.average_x_spacing,
            y_spacing_ref=config.average_y_spacing,
            spacing_tolerance=config.spacing_tolerance
        )
    
        if region.empty or restored_grid.empty:
            logger.warning("*** Grid restoration failed: Restored grid points are out of image bounds. Using second method for grid fitting.")
            region, rotated_zoom_grid, rotation_angle, best_row, best_col, best_zoom = grid_fitting_and_optimization(
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

            region, rotated_zoom_grid, rotation_angle, best_row, best_col, best_zoom = grid_fitting_and_optimization(
                filtered_regions,
                expected_rows=config.expected_rows,
                expected_cols=config.expected_cols,
                x_spacing_ref=config.average_x_spacing,
                y_spacing_ref=config.average_y_spacing,
                spacing_tolerance=config.spacing_tolerance
            )

        else:
            region.dropna(subset=["row", "col"], inplace=True)
            
            # Handle duplicate (row, col) assignments - keep the one closest to grid point
            if region.duplicated(subset=["row", "col"]).any():
                logger.debug("Found duplicate grid assignments, keeping closest points")
                # For each (row, col), calculate distance to expected position and keep minimum
                region = region.copy()
                region["row"] = region["row"].astype(int)
                region["col"] = region["col"].astype(int)
                
                # Merge with expected positions
                grid_df_temp = restored_grid.copy()
                grid_df_temp["row"] = grid_df_temp["row"].astype(int)
                grid_df_temp["col"] = grid_df_temp["col"].astype(int)
                region = region.merge(grid_df_temp[["row", "col", "expected_x", "expected_y"]], on=["row", "col"], how="left")
                
                # Calculate distance to expected position
                region["_dist_to_grid"] = np.sqrt(
                    (region["centroid_x"] - region["expected_x"])**2 + 
                    (region["centroid_y"] - region["expected_y"])**2
                )
                
                # Keep the closest point for each (row, col)
                region = region.loc[region.groupby(["row", "col"])["_dist_to_grid"].idxmin()]
                region = region.drop(columns=["_dist_to_grid", "expected_x", "expected_y"])
            
            region["row"] = region["row"].astype(int)
            region["col"] = region["col"].astype(int)
            region.set_index(["row", "col"], inplace=True, drop=True)
            
            # Use reindex to ensure all grid points are present and sorted
            grid_df = restored_grid.copy()
            grid_df["row"] = grid_df["row"].astype(int)
            grid_df["col"] = grid_df["col"].astype(int)
            grid_df = grid_df.set_index(["row", "col"])
            region = region.reindex(grid_df.index)
            
            region["grid_point_x"] = grid_df["expected_x"]
            region["grid_point_y"] = grid_df["expected_y"]
            
            region["area"] = region["area"].fillna(0)
    return region
@logger.catch
def colony_grid_table(
    image: np.ndarray,
    config: Configuration,
    image_notes: tuple | None = None
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
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

    return binary_image, detected_regions, filtered_regions, grid

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
    
    # Create reference grid from tetrad colony_regions for marker grid fitting
    # This helps when marker colonies are merged/irregular
    reference_grid = None
    if 'grid_point_x' in colony_regions.columns and 'grid_point_y' in colony_regions.columns:
        # Extract grid information from tetrad results
        tetrad_grid_info = colony_regions[['grid_point_x', 'grid_point_y']].dropna()
        if len(tetrad_grid_info) > 0:
            # Reset index to get row/col as columns
            tetrad_with_index = colony_regions.reset_index()
            if 'row' in tetrad_with_index.columns and 'col' in tetrad_with_index.columns:
                reference_grid = tetrad_with_index[['row', 'col', 'grid_point_x', 'grid_point_y']].dropna()
                reference_grid = reference_grid.rename(columns={'grid_point_x': 'expected_x', 'grid_point_y': 'expected_y'})
    
    marker_regions = filter_colonies(
        marker_erode,
        marker_regions,
        config,
        is_marker=True,
        reference_grid=reference_grid
    )
    # marker_centroids = marker_regions[["centroid_x", "centroid_y"]].dropna().to_numpy()
    # tetrad_centroids = colony_regions[["centroid_x", "centroid_y"]].dropna().to_numpy()
    # tetrad_centroids_indices = colony_regions[["centroid_x", "centroid_y"]].dropna().index
    marker_centroids = marker_regions[["grid_point_x", "grid_point_y"]].dropna().to_numpy()
    tetrad_centroids = colony_regions[["grid_point_x", "grid_point_y"]].dropna().to_numpy()
    tetrad_centroids_indices = colony_regions[["grid_point_x", "grid_point_y"]].dropna().index
    
    if len(marker_centroids) == 0 or len(tetrad_centroids) == 0:
        logger.error(f"*** {' '.join(map(str, image_notes)) if image_notes else ''}: No centroids detected in marker or tetrad images for matching.")
        return None, colony_regions, marker_regions, 1.0, 0.0, 0.0, 0.0, [], []

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
    
    # Subtract background and clip negative values
    aligned_marker_image_gray = np.clip(aligned_marker_image_gray.astype(np.float32) - background_signal, 0, 255).astype(np.uint8)
    
    # Precompute Otsu threshold and positive signal median
    otsu_threshold = threshold_otsu(aligned_marker_image_gray)
    positive_signal_median = np.median(aligned_marker_image_gray[aligned_marker_image_gray > otsu_threshold])
    marker_intensity_threshold = max(50, otsu_threshold)
    
    img_height, img_width = binary_tetrad_image.shape

    for idx, region in colony_regions.iterrows():
        cx, cy = int(region["grid_point_x"]), int(region["grid_point_y"])
        x1, x2 = max(0, cx - radius), min(img_width, cx + radius)
        y1, y2 = max(0, cy - radius), min(img_height, cy + radius)
        
        tetrad_patch = binary_tetrad_image[y1:y2, x1:x2]
        marker_patch = aligned_marker_image_gray[y1:y2, x1:x2]
        
        if marker_patch.size == 0:
            logger.warning(f"*** {' '.join(map(str, image_notes)) if image_notes else ''}: Empty marker patch for colony at index {idx}. Skipping genotype assignment.")
            colony_regions.loc[idx, "genotype"] = "Unknown"
            continue
            
        # Vectorized computations
        mean_tetrad_intensity = tetrad_patch.mean()
        mean_marker_intensity = marker_patch.mean()
        median_tetrad_intensity = np.median(tetrad_patch)
        median_marker_intensity = np.median(marker_patch)
        quantile_80_marker_intensity = np.percentile(marker_patch, 80)

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
def genotyping_by_colony_size(
    colony_regions: pd.DataFrame,
) -> pd.DataFrame:
    """Genotype colonies based on colony size per column."""
    colony_regions = colony_regions.copy()
    colony_regions["genotype"] = "Deletion"
    for col, col_data in colony_regions.groupby("col"):
        if col_data.shape[0] == 0:
            continue
        median_area = col_data["area"].median()
        for idx, region in col_data.iterrows():
            if region["area"] >= 0.75 * median_area:
                colony_regions.loc[idx, "genotype"] = "WT"
            else:
                colony_regions.loc[idx, "genotype"] = "Deletion"
    return colony_regions

@logger.catch
def plot_genotype_results(
    tetrad_results: dict[int, dict],
    colony_regions: pd.DataFrame,
    radius: int = 15,
    aligned_marker_image: np.ndarray | None = None,
    marker_plate_image: np.ndarray | None = None,
    marker_regions: pd.DataFrame | None = None,
):
    """Plot genotyping results with colony annotations."""
    n_days = len(tetrad_results)
    last_day = max(tetrad_results.keys())
    if aligned_marker_image is None or marker_plate_image is None or marker_regions is None:
        n_cols = n_days + 1 # +1 for colony area plot
    else:
        n_cols = n_days + 1 + 2 + 1 + 1 # +1 for marker plate, +1 for colony area plot
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 4))
    for day_idx, (day, day_data) in enumerate(sorted(tetrad_results.items())):
        ax = axes[day_idx]
        ax.imshow(day_data["image"], rasterized=True, cmap='gray')
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
    if aligned_marker_image is None or marker_plate_image is None or marker_regions is None:
        pass
    else:
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
    marker_image_path: Path | None = None,
    image_info: tuple | None = None,
    config: Configuration | None = None,
    colony_columns: int | None = None
) -> tuple[pd.DataFrame, Figure]:
    if colony_columns is None:
        config = Configuration(
            tetrad_image_paths=tetrad_image_paths,
            marker_image_path=marker_image_path
        )
    else:
        config = Configuration(
            tetrad_image_paths=tetrad_image_paths,
            marker_image_path=marker_image_path,
            expected_cols=colony_columns
        )

    first_day = min(config.tetrad_image_paths.keys())
    last_day = max(config.tetrad_image_paths.keys())
    day_colonies = {}
    first_day_colony_regions = None
    last_day_binary = None
    last_day_colony_regions = None
    for day, day_image_path in config.tetrad_image_paths.items():
        day_colonies[day] = {}
        image = io.imread(day_image_path)
        day_colonies[day]["image_notes"] = (*image_info, day) if image_info is not None else (day_image_path.stem, day)
        day_colonies[day]["image"] = image
        day_colonies[day]["binary_image"], day_colonies[day]["detected_regions"], day_colonies[day]["table"], day_colonies[day]["grids"] = colony_grid_table(
            image,
            config,
            image_notes=day_colonies[day]["image_notes"]
        )
        if day == first_day:
            first_day_colony_regions = day_colonies[day]["table"]
        if day == last_day and last_day_binary is None and last_day_colony_regions is None:
            last_day_binary = day_colonies[day]["binary_image"]
            last_day_colony_regions = day_colonies[day]["table"]
        day_colonies[day]["table"] = day_colonies[day]["table"].add_suffix(f"_day{day}")
    
    if last_day_binary is None or last_day_colony_regions is None or first_day_colony_regions is None:
        raise ValueError("No tetrad images were processed.")

    if config.marker_image_path is None:
        genotyping_colony_regions = genotyping_by_colony_size(first_day_colony_regions)
        all_colony_regions = pd.concat(
            [day_colonies[day]["table"] for day in sorted(day_colonies.keys())] + [genotyping_colony_regions[["genotype"]]],
            axis=1
        ).sort_index(level=[1,0], axis=0)
        fig = plot_genotype_results(day_colonies, all_colony_regions, radius=config.signal_detection_radius)
        return all_colony_regions, fig
    else:
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
        fig = plot_genotype_results(day_colonies, all_colony_regions, radius=config.signal_detection_radius, aligned_marker_image=marker_aligned, marker_plate_image=marker_plate_image, marker_regions=marker_regions)
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
#     3: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/16th_round/3d/201_alm1_3d_#3_202503.cropped.png"),
#     4: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/16th_round/4d/201_alm1_4d_#3_202503.cropped.png"),
#     5: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/16th_round/5d/201_alm1_5d_#3_202503.cropped.png"),
#     6: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/16th_round/6d/201_alm1_6d_#3_202503.cropped.png")
# }
# marker_image_path=Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/16th_round/replica/201_alm1_HYG_#3_202503.cropped.png")

# tetrad_image_paths={
#     3: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/16th_round/3d/201_alm1_3d_#3_202503.cropped.png"),
#     4: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/16th_round/4d/201_alm1_4d_#3_202503.cropped.png"),
#     5: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/16th_round/5d/201_alm1_5d_#3_202503.cropped.png"),
#     6: Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/16th_round/6d/201_alm1_6d_#3_202503.cropped.png")
# }
# marker_image_path=None

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
#     day_colonies[day]["binary_image"], day_colonies[day]["detected_regions"], day_colonies[day]["table"], day_colonies[day]["grids"] = colony_grid_table(
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
# marker_aligned, colony_regions, marker_regions, scale, angle, tx, ty, matched_tetrad_centroids, matched_marker_centroids = marker_plate_point_matching(
#     last_day_colony_regions,
#     marker_plate_image,
#     config
# )

# # if marker_aligned is None or len(matched_tetrad_centroids) < 3 or marker_aligned.size == 0:
# #     logger.error(f"*** {' '.join(map(str, image_info)) if image_info else ''}: Marker plate alignment failed. No aligned marker image available.")
# #     marker_aligned = marker_plate_image

# marker_aligned_gray = convert_to_grayscale(marker_aligned, channel=config.hyg_gray_channel)
# # %%
# genotyping_colony_regions = genotyping(last_day_binary, marker_aligned_gray, last_day_colony_regions, radius=config.signal_detection_radius)
# all_colony_regions = pd.concat(
#     [day_colonies[day]["table"] for day in sorted(day_colonies.keys())] + [genotyping_colony_regions[["genotype", "tetrad_intensity", "marker_intensity", "median_tetrad_intensity", "median_marker_intensity", "otsu_threshold", "positive_signal_median"]]],
#     axis=1
# ).sort_index(level=[1,0], axis=0)
# fig = plot_genotype_results(day_colonies, all_colony_regions, radius=config.signal_detection_radius, aligned_marker_image=marker_aligned, marker_plate_image=marker_plate_image, marker_regions=marker_regions)

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
