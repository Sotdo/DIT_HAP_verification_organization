"""
Tetrad dissection HYG selection analysis with grid-based genotyping.

This module analyzes replica plating results from tetrad dissection experiments
to distinguish deletion (HYG-resistant) from WT (HYG-sensitive) colonies.

Key Features:
- Grid-based colony detection with rotation and translation optimization
- Signal-to-noise ratio (SNR) based genotyping using Otsu threshold
- Robust image alignment using centroid matching
- Watershed segmentation for fused colony splitting

Usage:
    from src.align_and_measure_colony import analyze_replica_plating, AnalysisConfig
    
    config = AnalysisConfig(
        expected_cols=12,
        expected_rows=4,
    )
    
    del_count, wt_count = analyze_replica_plating(
        tetrad_path='path/to/tetrad.png',
        hyg_path='path/to/hyg.png',
        config=configm
    )
"""

# %% ------------------------------------ Imports ------------------------------------ #
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from loguru import logger

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment, minimize
from scipy import ndimage
from skimage import io, filters, measure, morphology
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


# %% ------------------------------------ Dataclasses ------------------------------------ #
@dataclass
class AnalysisConfig:
    """Configuration for tetrad-HYG replica plating analysis."""
    # Grid configuration for tetrad dissection
    expected_cols: int = 12  # Number of tetrads
    expected_rows: int = 4   # Spores per tetrad
    
    # Grid fitting parameters
    max_rotation_angle: float = 15.0  # Maximum rotation angle in degrees
    max_translation: float = 30.0  # Maximum translation in pixels
    
    # Colony detection - Tetrad plate
    tetrad_gaussian_sigma: float = 1.0
    tetrad_min_area: int = 10
    tetrad_split_min_distance: int = 20
    
    # Colony detection - HYG plate  
    hyg_gaussian_sigma: float = 0.5
    hyg_min_area: int = 20
    hyg_split_min_distance: int = 5
    
    # Watershed splitting parameters
    split_fused_colonies: bool = True
    min_area_for_split: int = 50  # Only attempt to split regions larger than this
    
    # Alignment parameters
    max_match_dist_fraction: float = 0.15  # 15% of image size
    ransac_reproj_threshold: float = 5.0
    ransac_max_iters: int = 2000
    ransac_confidence: float = 0.99
    
    # Signal measurement
    measure_radius_fraction: float = 0.35  # Fraction of y_spacing for measurement radius
    
    # Output
    output_filename: str = "replica_plating_analysis_result.png"
    output_dpi: int = 300


# %% ------------------------------------ Grid Fitting ------------------------------------ #
def detect_rotation_angle_robust(centroids: list, n_rows: int = 4, n_cols: int = 12) -> tuple:
    """
    Detect rotation angle using robust row-based estimation.
    
    Algorithm:
    1. Sort centroids by y-coordinate to identify rows
    2. For each row, fit a line and calculate slope
    3. Use median slope as the rotation angle (robust to outliers)
    
    Args:
        centroids: List of (x, y) centroid coordinates
        n_rows: Expected number of rows
        n_cols: Expected number of columns
        
    Returns:
        angle_deg: Rotation angle in degrees
        row_angles: Individual angle for each row
    """
    pts = np.array(centroids)
    n_pts = len(pts)
    
    if n_pts < n_cols:
        return 0.0, []
    
    # Sort by y-coordinate
    sorted_indices = np.argsort(pts[:, 1])
    pts_sorted = pts[sorted_indices]
    
    # Divide into rows
    pts_per_row = n_pts // n_rows
    row_angles = []
    
    for i in range(n_rows):
        start_idx = i * pts_per_row
        end_idx = start_idx + pts_per_row if i < n_rows - 1 else n_pts
        row_pts = pts_sorted[start_idx:end_idx]
        
        if len(row_pts) >= 2:
            row_pts = row_pts[np.argsort(row_pts[:, 0])]
            
            if len(row_pts) >= 3:
                slopes = []
                for j in range(len(row_pts) - 1):
                    dx = row_pts[j+1, 0] - row_pts[j, 0]
                    dy = row_pts[j+1, 1] - row_pts[j, 1]
                    if abs(dx) > 1:
                        slopes.append(dy / dx)
                if slopes:
                    angle = np.degrees(np.arctan(np.median(slopes)))
                    row_angles.append(angle)
            else:
                dx = row_pts[-1, 0] - row_pts[0, 0]
                dy = row_pts[-1, 1] - row_pts[0, 1]
                if abs(dx) > 1:
                    angle = np.degrees(np.arctan(dy / dx))
                    row_angles.append(angle)
    
    if not row_angles:
        return 0.0, []
    
    angle_deg = np.median(row_angles)
    angle_deg = np.clip(angle_deg, -15, 15)
    
    return angle_deg, row_angles


def rotate_points(points: np.ndarray, angle_deg: float, center: np.ndarray) -> np.ndarray:
    """Rotate points around a center by given angle."""
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    pts = np.array(points)
    pts_centered = pts - center
    
    rotated = np.zeros_like(pts_centered)
    rotated[:, 0] = pts_centered[:, 0] * cos_a - pts_centered[:, 1] * sin_a
    rotated[:, 1] = pts_centered[:, 0] * sin_a + pts_centered[:, 1] * cos_a
    
    return rotated + center


def calculate_grid_error(grid: np.ndarray, centroids: list) -> float:
    """Calculate average distance from grid points to nearest centroids."""
    pts = np.array(centroids)
    grid_flat = grid.reshape(-1, 2)
    
    total_error = 0
    for gp in grid_flat:
        dists = np.linalg.norm(pts - gp, axis=1)
        total_error += np.min(dists)
    
    return total_error / len(grid_flat)


def fit_grid_with_rotation(
    centroids: list, 
    n_rows: int = 4, 
    n_cols: int = 12
) -> tuple:
    """
    Fit grid using rotation correction only (no translation).
    
    Args:
        centroids: List of (x, y) centroid coordinates
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        
    Returns:
        grid: Grid positions (n_rows, n_cols, 2)
        params: Dictionary with grid parameters
    """
    pts = np.array(centroids)
    center = pts.mean(axis=0)
    
    # Detect rotation angle
    angle_deg, row_angles = detect_rotation_angle_robust(centroids, n_rows, n_cols)
    
    # Rotate centroids to align with axes
    pts_aligned = rotate_points(pts, -angle_deg, center)
    
    # Fit grid on aligned points
    x_min, x_max = pts_aligned[:, 0].min(), pts_aligned[:, 0].max()
    y_min, y_max = pts_aligned[:, 1].min(), pts_aligned[:, 1].max()
    
    x_spacing = (x_max - x_min) / (n_cols - 1) if n_cols > 1 else 0
    y_spacing = (y_max - y_min) / (n_rows - 1) if n_rows > 1 else 0
    
    # Create aligned grid
    grid_aligned = np.zeros((n_rows, n_cols, 2))
    for row in range(n_rows):
        for col in range(n_cols):
            grid_aligned[row, col, 0] = x_min + col * x_spacing
            grid_aligned[row, col, 1] = y_min + row * y_spacing
    
    # Rotate grid back
    grid_flat = grid_aligned.reshape(-1, 2)
    grid_rotated = rotate_points(grid_flat, angle_deg, center)
    grid = grid_rotated.reshape(n_rows, n_cols, 2)
    
    return grid, {
        'x_spacing': x_spacing, 
        'y_spacing': y_spacing,
        'rotation_angle': angle_deg,
        'row_angles': row_angles,
        'center': center
    }


def fit_grid_optimized(
    centroids: list, 
    n_rows: int = 4, 
    n_cols: int = 12, 
    max_translation: float = 30.0,
    verbose: bool = True
) -> tuple:
    """
    Two-step optimization: First rotation, then translation.
    
    Strategy:
    1. First fit grid with rotation correction (proven to work well)
    2. Then optimize translation using grid search + local optimization
    
    This avoids the joint optimization getting stuck in local optima.
    
    Args:
        centroids: List of (x, y) centroid coordinates
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        max_translation: Maximum translation in pixels
        verbose: Whether to print optimization details
        
    Returns:
        grid: Optimized grid positions (n_rows, n_cols, 2)
        params: Dictionary with optimization parameters
    """
    # Step 1: Get rotation-corrected grid (already works well)
    grid_rotated, params_rot = fit_grid_with_rotation(centroids, n_rows, n_cols)
    error_after_rotation = calculate_grid_error(grid_rotated, centroids)
    
    if verbose:
        logger.info("   Step 1 - Rotation correction:")
        logger.info(f"      Rotation angle: {params_rot['rotation_angle']:.2f}°")
        logger.info(f"      Error after rotation: {error_after_rotation:.2f} px")
    
    # Step 2: Optimize translation with grid search initialization
    if verbose:
        logger.info("   Step 2 - Translation optimization:")
    
    # Grid search for best translation starting point
    best_tx, best_ty = 0.0, 0.0
    best_error = error_after_rotation
    
    # Search in a grid pattern: -20 to +20 pixels in steps of 5
    search_range = np.arange(-20, 21, 5)
    
    for tx in search_range:
        for ty in search_range:
            grid_shifted = grid_rotated + np.array([tx, ty])
            err = calculate_grid_error(grid_shifted, centroids)
            if err < best_error:
                best_error = err
                best_tx, best_ty = tx, ty
    
    if verbose:
        logger.info(f"      Grid search best: tx={best_tx:.0f}, ty={best_ty:.0f}, error={best_error:.2f}")
    
    # Fine-tune with local optimization around the best grid search result
    def objective(params):
        tx, ty = params
        grid_shifted = grid_rotated + np.array([tx, ty])
        return calculate_grid_error(grid_shifted, centroids)
    
    # Start from grid search result
    result = minimize(objective, [best_tx, best_ty], method='L-BFGS-B',
                      bounds=[(-max_translation, max_translation), 
                              (-max_translation, max_translation)],
                      options={'ftol': 1e-8})
    
    tx_opt, ty_opt = result.x
    grid_final = grid_rotated + np.array([tx_opt, ty_opt])
    error_final = calculate_grid_error(grid_final, centroids)
    
    if verbose:
        logger.info(f"      Fine-tuned: tx={tx_opt:.1f}, ty={ty_opt:.1f}, error={error_final:.2f}")
    
    # Only use translation if it actually improves the result
    if error_final >= error_after_rotation:
        if verbose:
            logger.info("      Translation did not improve, using rotation-only result")
        return grid_rotated, {
            **params_rot,
            'translation': (0.0, 0.0),
            'translation_applied': False,
            'error_after_rotation': error_after_rotation,
            'error_final': error_after_rotation
        }
    
    improvement = (error_after_rotation - error_final) / error_after_rotation * 100
    if verbose:
        logger.info(f"      Translation improvement: {improvement:.1f}%")
    
    return grid_final, {
        **params_rot,
        'translation': (tx_opt, ty_opt),
        'translation_applied': True,
        'error_after_rotation': error_after_rotation,
        'error_final': error_final,
        'improvement_percent': improvement
    }


# %% ------------------------------------ Signal Measurement ------------------------------------ #
def measure_grid_signal(
    image: np.ndarray, 
    grid: np.ndarray, 
    radius: int = 15
) -> np.ndarray:
    """
    Measure signal intensity at each grid position.
    
    Uses the red channel since HYG plate has green background 
    (colonies appear bright in red channel).
    
    Args:
        image: HYG plate image (RGB)
        grid: Grid positions (n_rows, n_cols, 2)
        radius: Measurement radius in pixels
        
    Returns:
        signals: Signal intensity at each grid position (n_rows, n_cols)
    """
    n_rows, n_cols = grid.shape[:2]
    h, w = image.shape[:2]
    
    # Use red channel (HYG green background - colonies appear bright in red)
    gray = image[:, :, 0].astype(float) if image.ndim == 3 else image.astype(float)
    
    signals = np.zeros((n_rows, n_cols))
    
    for row in range(n_rows):
        for col in range(n_cols):
            cx, cy = int(grid[row, col, 0]), int(grid[row, col, 1])
            
            x1, x2 = max(0, cx - radius), min(w, cx + radius + 1)
            y1, y2 = max(0, cy - radius), min(h, cy + radius + 1)
            
            if x2 > x1 and y2 > y1:
                signals[row, col] = np.mean(gray[y1:y2, x1:x2])
    
    return signals


def genotype_by_signal(
    signals: np.ndarray, 
    threshold: float = None
) -> tuple:
    """
    Genotype colonies based on signal intensity using Otsu threshold.
    
    Args:
        signals: Signal intensity array (n_rows, n_cols)
        threshold: Optional fixed threshold. If None, use Otsu.
        
    Returns:
        genotypes: Genotype array ('DEL' or 'WT')
        threshold: Threshold used
        del_count: Number of deletion colonies
        wt_count: Number of WT colonies
    """
    if threshold is None:
        threshold = filters.threshold_otsu(signals.flatten())
    
    genotypes = np.where(signals > threshold, 'DEL', 'WT')
    del_count = np.sum(genotypes == 'DEL')
    wt_count = np.sum(genotypes == 'WT')
    
    return genotypes, threshold, del_count, wt_count


# %% ------------------------------------ Image Alignment ------------------------------------ #
def align_images_by_centroids(
    img_ref: np.ndarray,
    centroids_ref: list,
    img_target: np.ndarray,
    centroids_target: list,
    config: AnalysisConfig
) -> tuple:
    """
    Align target image to reference using centroid-based affine transformation.
    
    Uses Hungarian algorithm for optimal centroid matching, then estimates
    an affine transformation (rotation + translation + scale) via RANSAC.
    """
    h, w = img_ref.shape[:2]
    
    # Need at least 3 points for affine estimation
    if len(centroids_ref) < 3 or len(centroids_target) < 3:
        logger.warning("Insufficient colonies for alignment, using simple resize.")
        return cv2.resize(img_target, (w, h)), None, None
    
    pts_ref = np.array(centroids_ref, dtype=np.float32)
    pts_target = np.array(centroids_target, dtype=np.float32)
    
    # Match centroids using Hungarian algorithm
    dist_matrix = distance.cdist(pts_ref, pts_target)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    # Filter by distance threshold
    max_dist = max(h, w) * config.max_match_dist_fraction
    matched_ref, matched_target = [], []
    
    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] < max_dist:
            matched_ref.append(pts_ref[r])
            matched_target.append(pts_target[c])
    
    if len(matched_ref) < 3:
        logger.warning(f"Only matched {len(matched_ref)} points, using simple resize.")
        return cv2.resize(img_target, (w, h)), None, None
    
    matched_ref = np.array(matched_ref, dtype=np.float32)
    matched_target = np.array(matched_target, dtype=np.float32)
    
    logger.info(f"Matched {len(matched_ref)} colony positions for image alignment.")
    
    # Estimate affine transformation
    M, inliers = cv2.estimateAffinePartial2D(
        matched_target, matched_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_reproj_threshold,
        maxIters=config.ransac_max_iters,
        confidence=config.ransac_confidence
    )
    
    if M is None:
        logger.warning("Failed to estimate transform matrix, using simple resize.")
        return cv2.resize(img_target, (w, h)), None, None
    
    # Apply transformation
    aligned = cv2.warpAffine(
        img_target, M, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # Report transformation
    scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
    angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    tx, ty = M[0, 2], M[1, 2]
    inlier_count = np.sum(inliers) if inliers is not None else len(matched_ref)
    
    logger.info(f"Alignment: scale={scale:.3f}, rotation={angle:.2f}deg, translation=({tx:.1f}, {ty:.1f})px")
    logger.info(f"RANSAC inliers: {inlier_count}/{len(matched_ref)}")
    
    return aligned, M, inliers


def align_images_by_features(
    img_ref: np.ndarray,
    img_target: np.ndarray
) -> tuple:
    """
    Align images using ORB feature matching (fallback method).
    """
    h, w = img_ref.shape[:2]
    
    # Convert to grayscale
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY) if img_ref.ndim == 3 else img_ref
    gray_target = cv2.cvtColor(img_target, cv2.COLOR_RGB2GRAY) if img_target.ndim == 3 else img_target
    
    # Detect ORB features
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_target, None)
    
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        logger.warning("Feature detection failed, using simple resize.")
        return cv2.resize(img_target, (w, h)), None
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    
    n_matches = min(50, len(matches))
    if n_matches < 4:
        logger.warning("Insufficient matched features, using simple resize.")
        return cv2.resize(img_target, (w, h)), None
    
    # Extract matched points
    pts_ref = np.float32([kp1[m.queryIdx].pt for m in matches[:n_matches]])
    pts_target = np.float32([kp2[m.trainIdx].pt for m in matches[:n_matches]])
    
    # Estimate transformation
    M, _ = cv2.estimateAffinePartial2D(pts_target, pts_ref, method=cv2.RANSAC)
    
    if M is None:
        return cv2.resize(img_target, (w, h)), None
    
    aligned = cv2.warpAffine(img_target, M, (w, h))
    logger.info(f"Feature-based alignment: used {n_matches} features")
    
    return aligned, M


# %% ------------------------------------ Colony Detection ------------------------------------ #
def detect_colonies(
    img: np.ndarray,
    is_hyg_plate: bool = False,
    config: AnalysisConfig = None,
    mask_zeros: bool = False
) -> tuple:
    """
    Detect colony centroids in plate image.
    
    Args:
        img: Input image (RGB or grayscale)
        is_hyg_plate: True for HYG plate (green background), False for tetrad
        config: Analysis configuration
        mask_zeros: Exclude black pixels (for aligned images with borders)
        
    Returns:
        Tuple of (centroids_list, binary_mask)
    """
    if config is None:
        config = AnalysisConfig()
    
    # Convert to grayscale
    if is_hyg_plate:
        # Use red channel for HYG (green background appears dark in R channel)
        gray = img[:, :, 0] if img.ndim == 3 else img
        sigma = config.hyg_gaussian_sigma
        min_area = config.hyg_min_area
        min_distance = config.hyg_split_min_distance
    else:
        # Average for tetrad plate
        gray = np.mean(img, axis=2) if img.ndim == 3 else img
        sigma = config.tetrad_gaussian_sigma
        min_area = config.tetrad_min_area
        min_distance = config.tetrad_split_min_distance
    
    # Gaussian blur
    blurred = filters.gaussian(gray, sigma=sigma, preserve_range=True)
    
    # Thresholding
    if mask_zeros:
        valid_mask = gray > 10
        valid_pixels = blurred[valid_mask]
        if len(valid_pixels) > 100:
            try:
                thresh_val = filters.threshold_otsu(valid_pixels)
                mask = (blurred > thresh_val) & valid_mask
            except Exception:
                mask = (blurred > 100) & valid_mask
        else:
            mask = blurred > 100
    else:
        try:
            thresh_val = filters.threshold_otsu(blurred)
            mask = blurred > thresh_val
        except Exception:
            mask = blurred > 100
    
    # Morphological opening to remove noise
    mask = morphology.binary_opening(mask, morphology.disk(3))
    
    # Split fused colonies
    if config.split_fused_colonies:
        mask = split_fused_colonies(mask, min_distance, config.min_area_for_split)
    
    # Extract centroids
    label_img = measure.label(mask)
    props = measure.regionprops(label_img)
    
    centroids = []
    for prop in props:
        if prop.area > min_area:
            y, x = prop.centroid  # regionprops returns (row, col)
            centroids.append((x, y))
    
    return centroids, mask


def split_fused_colonies(
    binary_mask: np.ndarray,
    min_distance: int = 15,
    min_area_for_split: int = 150
) -> np.ndarray:
    """
    Split fused colonies using watershed segmentation.
    """
    # Label connected components
    labeled = measure.label(binary_mask)
    split_mask = np.zeros_like(binary_mask, dtype=bool)
    
    for region in measure.regionprops(labeled):
        region_mask = labeled == region.label
        
        # Small regions: keep as-is
        if region.area < min_area_for_split:
            split_mask |= region_mask
            continue
        
        # Compute distance transform
        dist_map = ndimage.distance_transform_edt(region_mask)
        
        if dist_map.max() == 0:
            split_mask |= region_mask
            continue
        
        # Find local maxima as markers
        coords = peak_local_max(
            dist_map,
            min_distance=min_distance,
            labels=region_mask.astype(int),
            exclude_border=False
        )
        
        if len(coords) < 2:
            # Can't split - only one peak found
            split_mask |= region_mask
            continue
        
        # Create markers
        markers = np.zeros(dist_map.shape, dtype=int)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
        
        # Dilate markers
        markers = ndimage.label(
            morphology.binary_dilation(markers > 0, morphology.disk(2))
        )[0]
        markers = measure.label(markers > 0)
        
        # Apply watershed
        ws_labels = watershed(-dist_map, markers, mask=region_mask)
        split_mask |= ws_labels > 0
    
    return split_mask


# %% ------------------------------------ Main Analysis ------------------------------------ #
def analyze_replica_plating(
    tetrad_path: str,
    hyg_path: str,
    config: AnalysisConfig = None,
    pic_result_path: str = None
) -> tuple:
    """
    Analyze replica plating results using grid-based genotyping.
    
    This method uses grid fitting with rotation and translation optimization
    to map colony positions, then measures signal intensity at each grid
    position to determine genotypes using Otsu thresholding.
    
    Args:
        tetrad_path: Path to tetrad plate image (reference)
        hyg_path: Path to HYG selection plate image
        config: Analysis configuration
        pic_result_path: Path to save result visualization
        
    Returns:
        Tuple of (deletion_count, wt_count)
    """
    if config is None:
        config = AnalysisConfig()
    
    # 1. Load images
    logger.info("=" * 60)
    logger.info("Tetrad Dissection HYG Selection Analysis (Grid-Based)")
    logger.info("=" * 60)
    
    try:
        img_tetrad = io.imread(str(tetrad_path))
        img_hyg = io.imread(str(hyg_path))
    except Exception as e:
        logger.error(f"Failed to read images. {e}")
        return None, None
    
    # Convert to uint8
    for img in [img_tetrad, img_hyg]:
        if img.dtype != np.uint8:
            if img.max() <= 1:
                img[:] = (img * 255).astype(np.uint8)
            else:
                img[:] = img.astype(np.uint8)
    
    logger.info(f"Tetrad image size: {img_tetrad.shape}")
    logger.info(f"HYG image size: {img_hyg.shape}")
    
    # 2. Initial colony detection
    logger.info("\n=== Phase 1: Colony Detection ===")
    
    logger.info("Detecting tetrad plate colonies...")
    tetrad_centroids, tetrad_mask = detect_colonies(
        img_tetrad, is_hyg_plate=False, config=config
    )
    logger.info(f"Detected {len(tetrad_centroids)} spores.")
    
    # Resize HYG to match tetrad
    h_ref, w_ref = img_tetrad.shape[:2]
    img_hyg_resized = cv2.resize(img_hyg, (w_ref, h_ref))
    
    logger.info("Detecting HYG plate colonies (initial)...")
    hyg_centroids_initial, _ = detect_colonies(
        img_hyg_resized, is_hyg_plate=True, config=config
    )
    logger.info(f"Detected {len(hyg_centroids_initial)} resistant clones.")
    
    # 3. Image alignment
    logger.info("\n=== Phase 2: Image Alignment ===")
    
    if len(tetrad_centroids) >= 6 and len(hyg_centroids_initial) >= 3:
        logger.info("Using centroid-based alignment...")
        img_hyg_aligned, transform_matrix, _ = align_images_by_centroids(
            img_tetrad, tetrad_centroids,
            img_hyg_resized, hyg_centroids_initial,
            config
        )
        alignment_method = "centroid"
    else:
        logger.info("Insufficient colonies, trying feature-based alignment...")
        img_hyg_aligned, transform_matrix = align_images_by_features(
            img_tetrad, img_hyg_resized
        )
        alignment_method = "feature" if transform_matrix is not None else "resize"
    
    # 4. Grid fitting with rotation and translation optimization
    logger.info("\n=== Phase 3: Grid Fitting ===")
    
    grid, grid_params = fit_grid_optimized(
        tetrad_centroids, 
        n_rows=config.expected_rows, 
        n_cols=config.expected_cols,
        max_translation=config.max_translation,
        verbose=True
    )
    
    logger.info("\nGrid fitting summary:")
    logger.info(f"   X spacing: {grid_params['x_spacing']:.1f} px")
    logger.info(f"   Y spacing: {grid_params['y_spacing']:.1f} px")
    logger.info(f"   Rotation angle: {grid_params['rotation_angle']:.2f} deg")
    if grid_params.get('translation_applied', False):
        tx, ty = grid_params['translation']
        logger.info(f"   Translation: ({tx:.1f}, {ty:.1f}) px")
        logger.info(f"   Improvement: {grid_params.get('improvement_percent', 0):.1f}%")
    else:
        logger.info("   Translation: not applied (no improvement)")
    logger.info(f"   Final grid error: {grid_params['error_final']:.2f} px")
    
    # 5. Measure signal at grid positions
    logger.info("\n=== Phase 4: Signal Measurement & Genotyping ===")
    
    measure_radius = int(grid_params['y_spacing'] * config.measure_radius_fraction)
    signals = measure_grid_signal(img_hyg_aligned, grid, radius=measure_radius)
    
    # Genotype using Otsu threshold
    genotypes, threshold, del_count, wt_count = genotype_by_signal(signals)
    
    logger.info(f"Measurement radius: {measure_radius} px")
    logger.info(f"Signal threshold (Otsu): {threshold:.1f}")
    logger.info("Genotyping result:")
    logger.info(f"   DEL (HYG-R): {del_count}")
    logger.info(f"   WT (HYG-S):  {wt_count}")
    
    # 6. Visualization
    logger.info("\n=== Generating Visualization ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Panel 1: Tetrad + detected centroids + grid
    axes[0, 0].imshow(img_tetrad)
    for (x, y) in tetrad_centroids:
        axes[0, 0].plot(x, y, 'r+', markersize=6)
    for row in range(config.expected_rows):
        for col in range(config.expected_cols):
            gx, gy = grid[row, col]
            axes[0, 0].plot(gx, gy, 'co', markersize=8, fillstyle='none', markeredgewidth=1.5)
    # Draw grid lines
    for row in range(config.expected_rows):
        xs = [grid[row, col, 0] for col in range(config.expected_cols)]
        ys = [grid[row, col, 1] for col in range(config.expected_cols)]
        axes[0, 0].plot(xs, ys, 'c-', alpha=0.3, linewidth=1)
    for col in range(config.expected_cols):
        xs = [grid[row, col, 0] for row in range(config.expected_rows)]
        ys = [grid[row, col, 1] for row in range(config.expected_rows)]
        axes[0, 0].plot(xs, ys, 'c-', alpha=0.3, linewidth=1)
    rot_str = f"rot={grid_params['rotation_angle']:.1f}deg"
    trans_str = ""
    if grid_params.get('translation_applied', False):
        tx, ty = grid_params['translation']
        trans_str = f", trans=({tx:.0f},{ty:.0f})"
    axes[0, 0].set_title(f'Tetrad + Grid ({rot_str}{trans_str})')
    axes[0, 0].axis('off')
    
    # Panel 2: Aligned HYG + measurement areas
    axes[0, 1].imshow(img_hyg_aligned)
    for row in range(config.expected_rows):
        for col in range(config.expected_cols):
            gx, gy = grid[row, col]
            circ = Circle((gx, gy), radius=measure_radius, color='yellow', fill=False, linewidth=1.5)
            axes[0, 1].add_patch(circ)
    axes[0, 1].set_title(f'Aligned HYG + Measurement Areas\n(radius={measure_radius}px, method={alignment_method})')
    axes[0, 1].axis('off')
    
    # Panel 3: Signal heatmap
    im = axes[0, 2].imshow(signals, cmap='hot', aspect='auto')
    axes[0, 2].set_title('Signal Heatmap')
    axes[0, 2].set_xticks(range(config.expected_cols))
    axes[0, 2].set_yticks(range(config.expected_rows))
    axes[0, 2].set_xticklabels([str(i+1) for i in range(config.expected_cols)])
    axes[0, 2].set_yticklabels(['A', 'B', 'C', 'D'][:config.expected_rows])
    plt.colorbar(im, ax=axes[0, 2])
    
    # Panel 4: Signal distribution
    axes[1, 0].hist(signals.flatten(), bins=20, color='steelblue', edgecolor='white')
    axes[1, 0].axvline(threshold, color='red', linestyle='--', linewidth=2, 
                       label=f'Threshold={threshold:.1f}')
    axes[1, 0].set_xlabel('Signal Intensity')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Signal Distribution')
    axes[1, 0].legend()
    
    # Panel 5: Genotype table
    gt_colors = np.where(genotypes == 'DEL', 1, 0)
    axes[1, 1].imshow(gt_colors, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    for row in range(config.expected_rows):
        for col in range(config.expected_cols):
            color = 'white' if genotypes[row, col] == 'DEL' else 'black'
            axes[1, 1].text(col, row, genotypes[row, col], ha='center', va='center', 
                           fontsize=8, color=color, fontweight='bold')
    axes[1, 1].set_title(f'Genotypes: DEL={del_count}, WT={wt_count}')
    axes[1, 1].set_xticks(range(config.expected_cols))
    axes[1, 1].set_yticks(range(config.expected_rows))
    axes[1, 1].set_xticklabels([str(i+1) for i in range(config.expected_cols)])
    axes[1, 1].set_yticklabels(['A', 'B', 'C', 'D'][:config.expected_rows])
    
    # Panel 6: Final result overlay
    axes[1, 2].imshow(img_tetrad)
    for row in range(config.expected_rows):
        for col in range(config.expected_cols):
            gx, gy = grid[row, col]
            color = 'lime' if genotypes[row, col] == 'DEL' else 'red'
            circ = Circle((gx, gy), radius=measure_radius, color=color, fill=False, linewidth=2)
            axes[1, 2].add_patch(circ)
            axes[1, 2].text(gx, gy-measure_radius-5, genotypes[row, col], 
                           color=color, fontsize=6, ha='center', fontweight='bold')
    axes[1, 2].set_title('Final Result\n(Green=DEL/HYG-R, Red=WT/HYG-S)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if pic_result_path:
        plt.savefig(pic_result_path, dpi=config.output_dpi, bbox_inches='tight')
        logger.success(f"Result saved to: {pic_result_path}")
    
    plt.show()
    plt.close()
    
    # 7. Summary
    logger.info("\n=== Analysis Summary ===")
    logger.info(f"Total spores detected: {len(tetrad_centroids)}")
    logger.info(f"Grid positions: {config.expected_rows} x {config.expected_cols} = {config.expected_rows * config.expected_cols}")
    logger.info(f"Deletion (HYG-R): {del_count}")
    logger.info(f"WT (HYG-S): {wt_count}")
    
    if del_count + wt_count > 0:
        expected = (config.expected_rows * config.expected_cols) // 2
        logger.info(f"DEL:WT ratio: {del_count}:{wt_count}")
        if abs(del_count - expected) <= 3 and abs(wt_count - expected) <= 3:
            logger.success("Consistent with expected 1:1 segregation ratio")
    
    return del_count, wt_count


# %% ------------------------------------ Main ------------------------------------ #
if __name__ == "__main__":
    import os
    
    # Example: Process from Excel file
    data = pd.read_excel("./results/all_rounds_combined_verification_summary.xlsx")
    
    # Configuration
    config = AnalysisConfig(
        expected_cols=12,
        expected_rows=4,
        split_fused_colonies=True,
        max_rotation_angle=15.0,
        max_translation=30.0
    )
    
    for idx, row in data.iloc[-10:,:].iterrows():
        try:
            tetrad_path = Path(row['6d_image_path'])
            hyg_path = Path(row['HYG_image_path'])
        except Exception as e:
            logger.error(f"Path read error: {e}")
            continue
        
        logger.info(f"\nProcessing sample: {tetrad_path.stem} and {hyg_path.stem}")
        
        if os.path.exists(tetrad_path) and os.path.exists(hyg_path):
            del_count, wt_count = analyze_replica_plating(
                tetrad_path, hyg_path, config=config, 
                pic_result_path=f"./results/selection_marker_test/{tetrad_path.stem}_result.png"
            )
        else:
            logger.warning("Please check file paths.")
            logger.warning(f"Tetrad: {tetrad_path}")
            logger.warning(f"HYG: {hyg_path}")
