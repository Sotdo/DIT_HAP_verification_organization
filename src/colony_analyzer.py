#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colony Analyzer - Tetrad Dissection HYG Selection Analysis

This script analyzes tetrad dissection plates with HYG selection to determine
genotypes (DEL vs WT) based on colony detection, image alignment, grid fitting
with rotation/translation optimization, and SNR-based signal measurement.

Author: Auto-generated from colony_analysis_tutorial.ipynb
Date: 2024
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.patches import Circle
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial.distance import cdist
from skimage import io, measure, morphology
from skimage.filters import gaussian, threshold_otsu


# ============================================================
# Configuration
# ============================================================

@dataclass
class AnalysisConfig:
    """Configuration for colony analysis."""
    
    # Grid parameters
    expected_rows: int = 4
    expected_cols: int = 12
    
    # Colony detection parameters
    tetrad_gaussian_sigma: float = 2.0
    hyg_gaussian_sigma: float = 1.0
    tetrad_min_area: int = 100
    hyg_min_area: int = 50
    morphology_disk_size: int = 3
    
    # Alignment parameters
    max_match_distance_ratio: float = 0.15  # Relative to image size
    min_matched_pairs: int = 4
    ransac_threshold: float = 5.0
    
    # Grid fitting parameters
    max_rotation_angle: float = 15.0  # degrees
    max_translation: float = 30.0  # pixels
    translation_search_step: int = 5  # pixels
    translation_search_range: int = 20  # pixels
    
    # Signal measurement parameters
    measure_radius_ratio: float = 0.35  # Relative to y_spacing
    snr_threshold: float = 2.0
    background_sample_size: int = 3  # pixels
    
    # Output parameters
    output_dpi: int = 150
    figure_size: tuple[int, int] = field(default_factory=lambda: (16, 12))


# ============================================================
# Image I/O
# ============================================================

def load_image(path: Path | str) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        path: Path to the image file.
        
    Returns:
        Image as numpy array (RGB format).
        
    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the image cannot be read.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    try:
        img = io.imread(str(path))
        logger.debug(f"Loaded image: {path.name}, shape={img.shape}, dtype={img.dtype}")
        return img
    except Exception as e:
        raise IOError(f"Failed to read image {path}: {e}") from e


def save_figure(fig: plt.Figure, path: Path | str, dpi: int = 150) -> None:
    """Save a matplotlib figure to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    logger.success(f"Figure saved: {path}")


# ============================================================
# Image Preprocessing
# ============================================================

def to_grayscale(image: np.ndarray, is_hyg_plate: bool = False) -> np.ndarray:
    """
    Convert image to grayscale.
    
    For tetrad plates (black background): Use mean of all channels.
    For HYG plates (green background): Use red channel for better contrast.
    
    Args:
        image: RGB image array.
        is_hyg_plate: Whether this is a HYG plate.
        
    Returns:
        Grayscale image as float array.
    """
    if image.ndim == 2:
        return image.astype(float)
    
    if is_hyg_plate:
        # Red channel gives better contrast on green background
        return image[:, :, 0].astype(float)
    else:
        # Mean of all channels for black background
        return np.mean(image, axis=2)


def preprocess_for_detection(
    image: np.ndarray,
    sigma: float,
    disk_size: int
) -> np.ndarray:
    """
    Preprocess image for colony detection.
    
    Pipeline: Grayscale → Gaussian blur → Otsu threshold → Morphological opening
    
    Args:
        image: Input image (grayscale).
        sigma: Gaussian blur sigma.
        disk_size: Morphological disk size.
        
    Returns:
        Binary mask after preprocessing.
    """
    blurred = gaussian(image, sigma=sigma, preserve_range=True)
    
    try:
        threshold = threshold_otsu(blurred)
        binary = blurred > threshold
    except ValueError:
        # Fallback if Otsu fails
        binary = blurred > np.median(blurred)
    
    # Morphological opening to remove noise
    cleaned = morphology.binary_opening(binary, morphology.disk(disk_size))
    
    return cleaned


# ============================================================
# Colony Detection
# ============================================================

def detect_colonies(
    image: np.ndarray,
    is_hyg_plate: bool = False,
    config: AnalysisConfig | None = None
) -> tuple[list[tuple[float, float]], np.ndarray]:
    """
    Detect colony centroids in an image.
    
    Args:
        image: RGB image array.
        is_hyg_plate: Whether this is a HYG plate.
        config: Analysis configuration.
        
    Returns:
        Tuple of (centroids list, binary mask).
        Centroids are in (x, y) format.
    """
    if config is None:
        config = AnalysisConfig()
    
    # Parameters based on plate type
    sigma = config.hyg_gaussian_sigma if is_hyg_plate else config.tetrad_gaussian_sigma
    min_area = config.hyg_min_area if is_hyg_plate else config.tetrad_min_area
    
    # Convert to grayscale
    gray = to_grayscale(image, is_hyg_plate)
    
    # Preprocess
    binary = preprocess_for_detection(gray, sigma, config.morphology_disk_size)
    
    # Label connected regions
    labeled = measure.label(binary)
    props = measure.regionprops(labeled)
    
    # Extract centroids (filter by area)
    centroids = [
        (prop.centroid[1], prop.centroid[0])  # Convert (row, col) to (x, y)
        for prop in props
        if prop.area > min_area
    ]
    
    return centroids, binary


# ============================================================
# Image Alignment
# ============================================================

def align_images_by_centroids(
    img_ref: np.ndarray,
    img_target: np.ndarray,
    centroids_ref: list[tuple[float, float]],
    centroids_target: list[tuple[float, float]],
    config: AnalysisConfig | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Align target image to reference using centroid matching.
    
    Uses Hungarian algorithm for optimal point matching, then RANSAC
    for robust affine transformation estimation.
    
    Args:
        img_ref: Reference image.
        img_target: Target image to align.
        centroids_ref: Reference centroids.
        centroids_target: Target centroids.
        config: Analysis configuration.
        
    Returns:
        Tuple of (aligned image, transformation matrix or None).
    """
    if config is None:
        config = AnalysisConfig()
    
    h, w = img_ref.shape[:2]
    
    if len(centroids_ref) < config.min_matched_pairs or \
       len(centroids_target) < config.min_matched_pairs:
        logger.warning(f"Insufficient points for alignment: "
                      f"ref={len(centroids_ref)}, target={len(centroids_target)}")
        return img_target.copy(), None
    
    # Convert to arrays
    pts_ref = np.array(centroids_ref, dtype=np.float32)
    pts_target = np.array(centroids_target, dtype=np.float32)
    
    # Compute distance matrix and find optimal matching
    dist_matrix = cdist(pts_ref, pts_target)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    # Filter matches with large distances
    max_dist = max(h, w) * config.max_match_distance_ratio
    matched_ref, matched_target = [], []
    
    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] < max_dist:
            matched_ref.append(pts_ref[r])
            matched_target.append(pts_target[c])
    
    logger.info(f"Matched {len(matched_ref)} point pairs for alignment")
    
    if len(matched_ref) < config.min_matched_pairs:
        logger.warning("Too few valid matches for alignment")
        return img_target.copy(), None
    
    # Estimate transformation using RANSAC
    matched_ref = np.array(matched_ref, dtype=np.float32)
    matched_target = np.array(matched_target, dtype=np.float32)
    
    transform_matrix, inliers = cv2.estimateAffinePartial2D(
        matched_target, matched_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_threshold
    )
    
    if transform_matrix is None:
        logger.warning("RANSAC alignment failed")
        return img_target.copy(), None
    
    # Apply transformation
    aligned = cv2.warpAffine(img_target, transform_matrix, (w, h))
    
    # Log transformation parameters
    scale = np.sqrt(transform_matrix[0, 0]**2 + transform_matrix[0, 1]**2)
    angle = np.degrees(np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0]))
    tx, ty = transform_matrix[0, 2], transform_matrix[1, 2]
    
    logger.info(f"Alignment: scale={scale:.4f}, rotation={angle:.2f}°, "
               f"translation=({tx:.1f}, {ty:.1f})px")
    
    return aligned, transform_matrix


# ============================================================
# Grid Fitting with Rotation and Translation Optimization
# ============================================================

def detect_rotation_angle_robust(
    centroids: list[tuple[float, float]],
    n_rows: int = 4,
    n_cols: int = 12,
    max_angle: float = 15.0
) -> tuple[float, list[float]]:
    """
    Detect rotation angle using robust row-based estimation.
    
    Algorithm:
    1. Sort centroids by y-coordinate to identify rows
    2. For each row, fit a line and calculate slope
    3. Use median slope as the rotation angle (robust to outliers)
    
    Args:
        centroids: List of (x, y) centroid coordinates.
        n_rows: Expected number of rows.
        n_cols: Expected number of columns.
        max_angle: Maximum allowed rotation angle in degrees.
        
    Returns:
        Tuple of (rotation angle in degrees, list of per-row angles).
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
        
        if len(row_pts) < 2:
            continue
            
        # Sort by x within row
        row_pts = row_pts[np.argsort(row_pts[:, 0])]
        
        if len(row_pts) >= 3:
            # Calculate slopes between consecutive points
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
            # Use endpoints for two-point rows
            dx = row_pts[-1, 0] - row_pts[0, 0]
            dy = row_pts[-1, 1] - row_pts[0, 1]
            if abs(dx) > 1:
                angle = np.degrees(np.arctan(dy / dx))
                row_angles.append(angle)
    
    if not row_angles:
        return 0.0, []
    
    # Use median and clip to max angle
    angle_deg = np.median(row_angles)
    angle_deg = np.clip(angle_deg, -max_angle, max_angle)
    
    return float(angle_deg), row_angles


def rotate_points(
    points: np.ndarray,
    angle_deg: float,
    center: np.ndarray
) -> np.ndarray:
    """Rotate points around a center by given angle."""
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    pts_centered = points - center
    
    rotated = np.zeros_like(pts_centered)
    rotated[:, 0] = pts_centered[:, 0] * cos_a - pts_centered[:, 1] * sin_a
    rotated[:, 1] = pts_centered[:, 0] * sin_a + pts_centered[:, 1] * cos_a
    
    return rotated + center


def calculate_grid_error(grid: np.ndarray, centroids: list[tuple[float, float]]) -> float:
    """Calculate average distance from grid points to nearest centroids."""
    pts = np.array(centroids)
    grid_flat = grid.reshape(-1, 2)
    
    total_error = 0.0
    for gp in grid_flat:
        dists = np.linalg.norm(pts - gp, axis=1)
        total_error += np.min(dists)
    
    return total_error / len(grid_flat)


def fit_grid_simple(
    centroids: list[tuple[float, float]],
    n_rows: int = 4,
    n_cols: int = 12
) -> tuple[np.ndarray, dict]:
    """
    Fit a simple grid from centroid bounding box.
    
    Args:
        centroids: List of (x, y) centroid coordinates.
        n_rows: Expected number of rows.
        n_cols: Expected number of columns.
        
    Returns:
        Tuple of (grid array [n_rows, n_cols, 2], parameters dict).
    """
    pts = np.array(centroids)
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    
    x_spacing = (x_max - x_min) / (n_cols - 1) if n_cols > 1 else 0
    y_spacing = (y_max - y_min) / (n_rows - 1) if n_rows > 1 else 0
    
    grid = np.zeros((n_rows, n_cols, 2))
    for row in range(n_rows):
        for col in range(n_cols):
            grid[row, col, 0] = x_min + col * x_spacing
            grid[row, col, 1] = y_min + row * y_spacing
    
    return grid, {'x_spacing': x_spacing, 'y_spacing': y_spacing}


def fit_grid_with_rotation(
    centroids: list[tuple[float, float]],
    n_rows: int = 4,
    n_cols: int = 12,
    max_angle: float = 15.0
) -> tuple[np.ndarray, dict]:
    """
    Fit grid using rotation correction only.
    
    Args:
        centroids: List of (x, y) centroid coordinates.
        n_rows: Expected number of rows.
        n_cols: Expected number of columns.
        max_angle: Maximum allowed rotation angle.
        
    Returns:
        Tuple of (grid array, parameters dict).
    """
    pts = np.array(centroids)
    center = pts.mean(axis=0)
    
    # Detect rotation angle
    angle_deg, row_angles = detect_rotation_angle_robust(
        centroids, n_rows, n_cols, max_angle
    )
    
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
    
    # Rotate grid back to original orientation
    grid_flat = grid_aligned.reshape(-1, 2)
    grid_rotated = rotate_points(grid_flat, angle_deg, center)
    grid = grid_rotated.reshape(n_rows, n_cols, 2)
    
    return grid, {
        'x_spacing': x_spacing,
        'y_spacing': y_spacing,
        'rotation_angle': angle_deg,
        'row_angles': row_angles,
        'center': center.tolist()
    }


def fit_grid_optimized(
    centroids: list[tuple[float, float]],
    n_rows: int = 4,
    n_cols: int = 12,
    config: AnalysisConfig | None = None
) -> tuple[np.ndarray, dict]:
    """
    Two-step optimization: First rotation, then translation.
    
    Strategy:
    1. First fit grid with rotation correction (robust)
    2. Then optimize translation using grid search + local optimization
    3. Only apply translation if it improves the result
    
    Args:
        centroids: List of (x, y) centroid coordinates.
        n_rows: Expected number of rows.
        n_cols: Expected number of columns.
        config: Analysis configuration.
        
    Returns:
        Tuple of (optimized grid array, parameters dict).
    """
    if config is None:
        config = AnalysisConfig()
    
    pts = np.array(centroids)
    
    # Step 1: Get rotation-corrected grid
    grid_rotated, params_rot = fit_grid_with_rotation(
        centroids, n_rows, n_cols, config.max_rotation_angle
    )
    error_after_rotation = calculate_grid_error(grid_rotated, centroids)
    
    logger.info(f"Step 1 - Rotation correction:")
    logger.info(f"   Rotation angle: {params_rot['rotation_angle']:.2f}°")
    logger.info(f"   Error after rotation: {error_after_rotation:.2f} px")
    
    # Step 2: Optimize translation with grid search initialization
    logger.info("Step 2 - Translation optimization:")
    
    # Grid search for best translation starting point
    best_tx, best_ty = 0.0, 0.0
    best_error = error_after_rotation
    
    search_range = np.arange(
        -config.translation_search_range,
        config.translation_search_range + 1,
        config.translation_search_step
    )
    
    for tx in search_range:
        for ty in search_range:
            grid_shifted = grid_rotated + np.array([tx, ty])
            err = calculate_grid_error(grid_shifted, centroids)
            if err < best_error:
                best_error = err
                best_tx, best_ty = float(tx), float(ty)
    
    logger.debug(f"   Grid search best: tx={best_tx:.0f}, ty={best_ty:.0f}, "
                f"error={best_error:.2f}")
    
    # Fine-tune with local optimization
    def objective(params):
        tx, ty = params
        grid_shifted = grid_rotated + np.array([tx, ty])
        return calculate_grid_error(grid_shifted, centroids)
    
    result = minimize(
        objective, [best_tx, best_ty],
        method='L-BFGS-B',
        bounds=[
            (-config.max_translation, config.max_translation),
            (-config.max_translation, config.max_translation)
        ],
        options={'ftol': 1e-8}
    )
    
    tx_opt, ty_opt = result.x
    grid_final = grid_rotated + np.array([tx_opt, ty_opt])
    error_final = calculate_grid_error(grid_final, centroids)
    
    logger.debug(f"   Fine-tuned: tx={tx_opt:.1f}, ty={ty_opt:.1f}, "
                f"error={error_final:.2f}")
    
    # Only use translation if it actually improves the result
    if error_final >= error_after_rotation:
        logger.info("   Translation did not improve, using rotation-only result")
        return grid_rotated, {
            **params_rot,
            'translation': (0.0, 0.0),
            'translation_applied': False,
            'error_after_rotation': error_after_rotation,
            'error_final': error_after_rotation
        }
    
    improvement = (error_after_rotation - error_final) / error_after_rotation * 100
    logger.info(f"   Translation: ({tx_opt:.1f}, {ty_opt:.1f}) px")
    logger.info(f"   Improvement: {improvement:.1f}%")
    logger.info(f"   Final error: {error_final:.2f} px")
    
    return grid_final, {
        **params_rot,
        'translation': (tx_opt, ty_opt),
        'translation_applied': True,
        'error_after_rotation': error_after_rotation,
        'error_final': error_final,
        'improvement_percent': improvement
    }


# ============================================================
# Signal Measurement and SNR
# ============================================================

def estimate_background(
    image: np.ndarray,
    grid: np.ndarray,
    x_spacing: float,
    y_spacing: float,
    sample_size: int = 3
) -> tuple[float, float]:
    """
    Estimate background intensity by sampling between grid points.
    
    Args:
        image: Grayscale image.
        grid: Grid array [n_rows, n_cols, 2].
        x_spacing: Grid x spacing.
        y_spacing: Grid y spacing.
        sample_size: Sample window half-size.
        
    Returns:
        Tuple of (background mean, background std).
    """
    h, w = image.shape[:2]
    n_rows, n_cols = grid.shape[:2]
    
    bg_samples = []
    s = sample_size
    
    # Sample between columns
    for row in range(n_rows):
        for col in range(n_cols - 1):
            x1, y1 = grid[row, col]
            x2, y2 = grid[row, col + 1]
            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)
            
            if s < mid_x < w - s and s < mid_y < h - s:
                region = image[mid_y-s:mid_y+s+1, mid_x-s:mid_x+s+1]
                bg_samples.append(np.mean(region))
    
    # Sample between rows
    for row in range(n_rows - 1):
        for col in range(n_cols):
            x1, y1 = grid[row, col]
            x2, y2 = grid[row + 1, col]
            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)
            
            if s < mid_x < w - s and s < mid_y < h - s:
                region = image[mid_y-s:mid_y+s+1, mid_x-s:mid_x+s+1]
                bg_samples.append(np.mean(region))
    
    if not bg_samples:
        logger.warning("No valid background samples, using image corners")
        corners = [
            image[:20, :20],
            image[:20, -20:],
            image[-20:, :20],
            image[-20:, -20:]
        ]
        bg_samples = [np.mean(c) for c in corners]
    
    return float(np.median(bg_samples)), float(np.std(bg_samples))


def measure_grid_signals(
    image: np.ndarray,
    grid: np.ndarray,
    radius: int
) -> np.ndarray:
    """
    Measure signal intensity at each grid position.
    
    Args:
        image: Grayscale image.
        grid: Grid array [n_rows, n_cols, 2].
        radius: Measurement radius in pixels.
        
    Returns:
        Signal array [n_rows, n_cols].
    """
    n_rows, n_cols = grid.shape[:2]
    h, w = image.shape[:2]
    
    signals = np.zeros((n_rows, n_cols))
    
    for row in range(n_rows):
        for col in range(n_cols):
            cx, cy = int(grid[row, col, 0]), int(grid[row, col, 1])
            
            x1, x2 = max(0, cx - radius), min(w, cx + radius + 1)
            y1, y2 = max(0, cy - radius), min(h, cy + radius + 1)
            
            if x2 > x1 and y2 > y1:
                signals[row, col] = np.mean(image[y1:y2, x1:x2])
    
    return signals


def calculate_snr(
    signals: np.ndarray,
    bg_mean: float,
    bg_std: float
) -> np.ndarray:
    """
    Calculate Signal-to-Noise Ratio.
    
    SNR = (signal - background_mean) / background_std
    
    Args:
        signals: Signal array.
        bg_mean: Background mean.
        bg_std: Background standard deviation.
        
    Returns:
        SNR array with same shape as signals.
    """
    if bg_std <= 0:
        logger.warning("Background std is zero, returning raw signals")
        return signals - bg_mean
    
    return (signals - bg_mean) / bg_std


def genotype_by_snr(
    snr: np.ndarray,
    threshold: float = 2.0
) -> np.ndarray:
    """
    Determine genotypes based on SNR.
    
    For Essential Genes:
    - High SNR (signal present) = WT (survives)
    - Low SNR (no signal) = DEL (lethal, no growth)
    
    Args:
        snr: SNR array.
        threshold: SNR threshold for WT classification.
        
    Returns:
        Genotype array with 'DEL' or 'WT' strings.
    """
    return np.where(snr > threshold, 'WT', 'DEL')


def genotype_by_otsu(signals: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Determine genotypes using Otsu thresholding.
    
    For Essential Genes:
    - High signal = WT (survives)
    - Low signal = DEL (lethal)
    
    Args:
        signals: Signal array.
        
    Returns:
        Tuple of (genotype array, threshold value).
    """
    threshold = threshold_otsu(signals.flatten())
    genotypes = np.where(signals > threshold, 'WT', 'DEL')
    return genotypes, threshold


# ============================================================
# Main Analysis Function
# ============================================================

@dataclass
class AnalysisResult:
    """Result container for colony analysis."""
    
    # Genotyping results
    genotypes: np.ndarray
    del_count: int
    wt_count: int
    
    # Signal data
    signals: np.ndarray
    snr: np.ndarray
    snr_threshold: float
    
    # Grid data
    grid: np.ndarray
    grid_params: dict
    
    # Background
    bg_mean: float
    bg_std: float
    
    # Alignment
    transform_matrix: np.ndarray | None = None
    
    # Colony detection
    tetrad_centroids: list[tuple[float, float]] = field(default_factory=list)
    hyg_centroids: list[tuple[float, float]] = field(default_factory=list)


def analyze_tetrad_pair(
    tetrad_path: Path | str,
    hyg_path: Path | str,
    config: AnalysisConfig | None = None,
    output_path: Path | str | None = None,
    show_plot: bool = False
) -> AnalysisResult | None:
    """
    Analyze a tetrad and HYG image pair for genotyping.
    
    Pipeline:
    1. Load and preprocess images
    2. Detect colonies on both plates
    3. Align HYG to tetrad
    4. Fit grid with rotation and translation optimization
    5. Measure signals and calculate SNR
    6. Genotype using SNR threshold
    7. Generate visualization
    
    Args:
        tetrad_path: Path to tetrad plate image.
        hyg_path: Path to HYG plate image.
        config: Analysis configuration.
        output_path: Path to save result figure (optional).
        show_plot: Whether to display the plot.
        
    Returns:
        AnalysisResult object or None if analysis fails.
    """
    if config is None:
        config = AnalysisConfig()
    
    logger.info("=" * 60)
    logger.info("Tetrad Dissection HYG Selection Analysis")
    logger.info("=" * 60)
    
    # 1. Load images
    try:
        img_tetrad = load_image(tetrad_path)
        img_hyg = load_image(hyg_path)
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Failed to load images: {e}")
        return None
    
    logger.info(f"Tetrad image: {img_tetrad.shape}")
    logger.info(f"HYG image: {img_hyg.shape}")
    
    # 2. Detect colonies
    logger.info("\n=== Phase 1: Colony Detection ===")
    
    logger.info("Detecting tetrad plate colonies...")
    tetrad_centroids, tetrad_mask = detect_colonies(
        img_tetrad, is_hyg_plate=False, config=config
    )
    logger.info(f"Detected {len(tetrad_centroids)} spores")
    
    # Resize HYG to match tetrad
    h_ref, w_ref = img_tetrad.shape[:2]
    img_hyg_resized = cv2.resize(img_hyg, (w_ref, h_ref))
    
    logger.info("Detecting HYG plate colonies...")
    hyg_centroids, hyg_mask = detect_colonies(
        img_hyg_resized, is_hyg_plate=True, config=config
    )
    logger.info(f"Detected {len(hyg_centroids)} resistant clones")
    
    # 3. Image alignment
    logger.info("\n=== Phase 2: Image Alignment ===")
    
    if len(tetrad_centroids) >= config.min_matched_pairs and \
       len(hyg_centroids) >= config.min_matched_pairs:
        logger.info("Using centroid-based alignment...")
        img_hyg_aligned, transform_matrix = align_images_by_centroids(
            img_tetrad, img_hyg_resized,
            tetrad_centroids, hyg_centroids,
            config
        )
    else:
        logger.warning("Insufficient colonies for alignment, using resized image")
        img_hyg_aligned = img_hyg_resized
        transform_matrix = None
    
    # 4. Grid fitting
    logger.info("\n=== Phase 3: Grid Fitting ===")
    
    grid, grid_params = fit_grid_optimized(
        tetrad_centroids,
        n_rows=config.expected_rows,
        n_cols=config.expected_cols,
        config=config
    )
    
    logger.info("\nGrid fitting summary:")
    logger.info(f"   X spacing: {grid_params['x_spacing']:.1f} px")
    logger.info(f"   Y spacing: {grid_params['y_spacing']:.1f} px")
    logger.info(f"   Rotation angle: {grid_params['rotation_angle']:.2f}°")
    if grid_params.get('translation_applied', False):
        tx, ty = grid_params['translation']
        logger.info(f"   Translation: ({tx:.1f}, {ty:.1f}) px")
    else:
        logger.info("   Translation: not applied")
    logger.info(f"   Final error: {grid_params['error_final']:.2f} px")
    
    # 5. Signal measurement and SNR
    logger.info("\n=== Phase 4: Signal Measurement (SNR Method) ===")
    
    # Convert to grayscale
    gray_hyg = to_grayscale(img_hyg_aligned, is_hyg_plate=True)
    
    # Calculate measurement radius
    measure_radius = int(grid_params['y_spacing'] * config.measure_radius_ratio)
    logger.info(f"Measurement radius: {measure_radius} px")
    
    # Estimate background
    bg_mean, bg_std = estimate_background(
        gray_hyg, grid,
        grid_params['x_spacing'], grid_params['y_spacing'],
        config.background_sample_size
    )
    logger.info(f"Background: mean={bg_mean:.1f}, std={bg_std:.1f}")
    
    # Measure signals
    signals = measure_grid_signals(gray_hyg, grid, measure_radius)
    
    # Calculate SNR
    snr = calculate_snr(signals, bg_mean, bg_std)
    logger.info(f"SNR range: {snr.min():.2f} ~ {snr.max():.2f}")
    
    # 6. Genotyping
    logger.info("\n=== Phase 5: Genotyping ===")
    
    genotypes = genotype_by_snr(snr, config.snr_threshold)
    del_count = int(np.sum(genotypes == 'DEL'))
    wt_count = int(np.sum(genotypes == 'WT'))
    
    logger.info(f"SNR threshold: {config.snr_threshold}")
    logger.info(f"DEL (HYG-R): {del_count}")
    logger.info(f"WT (HYG-S):  {wt_count}")
    
    # Also show Otsu result for comparison
    genotypes_otsu, otsu_threshold = genotype_by_otsu(signals)
    del_otsu = int(np.sum(genotypes_otsu == 'DEL'))
    wt_otsu = int(np.sum(genotypes_otsu == 'WT'))
    logger.debug(f"[Otsu comparison] threshold={otsu_threshold:.1f}, "
                f"DEL={del_otsu}, WT={wt_otsu}")
    
    # 7. Visualization
    if output_path or show_plot:
        logger.info("\n=== Generating Visualization ===")
        
        fig, axes = plt.subplots(2, 3, figsize=config.figure_size)
        
        # Tetrad + centroids + grid
        axes[0, 0].imshow(img_tetrad)
        for x, y in tetrad_centroids:
            axes[0, 0].plot(x, y, 'r+', markersize=6, markeredgewidth=1)
        for row in range(config.expected_rows):
            for col in range(config.expected_cols):
                gx, gy = grid[row, col]
                axes[0, 0].plot(gx, gy, 'co', markersize=8, fillstyle='none')
        # Draw grid lines
        for row in range(config.expected_rows):
            xs = [grid[row, col, 0] for col in range(config.expected_cols)]
            ys = [grid[row, col, 1] for col in range(config.expected_cols)]
            axes[0, 0].plot(xs, ys, 'c-', alpha=0.3, linewidth=1)
        axes[0, 0].set_title(f'Tetrad + Grid\n({len(tetrad_centroids)} colonies)')
        axes[0, 0].axis('off')
        
        # HYG aligned + measurement circles
        axes[0, 1].imshow(img_hyg_aligned)
        for row in range(config.expected_rows):
            for col in range(config.expected_cols):
                gx, gy = grid[row, col]
                circ = Circle(
                    (gx, gy), radius=measure_radius,
                    color='yellow', fill=False, linewidth=1.5
                )
                axes[0, 1].add_patch(circ)
        axes[0, 1].set_title(f'HYG Aligned + Measurement Areas\n(radius={measure_radius}px)')
        axes[0, 1].axis('off')
        
        # SNR heatmap
        im = axes[0, 2].imshow(snr, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=5)
        for row in range(config.expected_rows):
            for col in range(config.expected_cols):
                color = 'white' if snr[row, col] >= config.snr_threshold else 'black'
                axes[0, 2].text(
                    col, row, f'{snr[row, col]:.1f}',
                    ha='center', va='center', fontsize=7, color=color
                )
        axes[0, 2].set_title(f'SNR (threshold={config.snr_threshold})')
        axes[0, 2].set_xticks(range(config.expected_cols))
        axes[0, 2].set_yticks(range(config.expected_rows))
        axes[0, 2].set_xticklabels([str(i+1) for i in range(config.expected_cols)])
        axes[0, 2].set_yticklabels(['A', 'B', 'C', 'D'][:config.expected_rows])
        plt.colorbar(im, ax=axes[0, 2])
        
        # Signal histogram with SNR threshold
        axes[1, 0].hist(snr.flatten(), bins=15, color='steelblue', edgecolor='white')
        axes[1, 0].axvline(
            config.snr_threshold, color='red', linestyle='--',
            linewidth=2, label=f'Threshold={config.snr_threshold}'
        )
        axes[1, 0].axvline(0, color='gray', linestyle=':', label='Background')
        axes[1, 0].set_xlabel('SNR')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('SNR Distribution')
        axes[1, 0].legend()
        
        # Genotype table
        gt_colors = np.where(genotypes == 'WT', 1, 0)  # Green=WT, Red=DEL
        axes[1, 1].imshow(gt_colors, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        for row in range(config.expected_rows):
            for col in range(config.expected_cols):
                color = 'white' if genotypes[row, col] == 'DEL' else 'black'
                axes[1, 1].text(
                    col, row, genotypes[row, col],
                    ha='center', va='center', fontsize=8,
                    color=color, fontweight='bold'
                )
        axes[1, 1].set_title(f'Genotypes: DEL={del_count}, WT={wt_count}')
        axes[1, 1].set_xticks(range(config.expected_cols))
        axes[1, 1].set_yticks(range(config.expected_rows))
        axes[1, 1].set_xticklabels([str(i+1) for i in range(config.expected_cols)])
        axes[1, 1].set_yticklabels(['A', 'B', 'C', 'D'][:config.expected_rows])
        
        # Final result on tetrad
        axes[1, 2].imshow(img_tetrad)
        for row in range(config.expected_rows):
            for col in range(config.expected_cols):
                gx, gy = grid[row, col]
                color = 'lime' if genotypes[row, col] == 'WT' else 'red'
                circ = Circle(
                    (gx, gy), radius=measure_radius,
                    color=color, fill=False, linewidth=2
                )
                axes[1, 2].add_patch(circ)
        axes[1, 2].set_title('Final Result\n(Green=WT, Red=DEL)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            save_figure(fig, output_path, config.output_dpi)
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
    
    # Summary
    logger.info("\n=== Analysis Summary ===")
    logger.info(f"Total spores detected: {len(tetrad_centroids)}")
    logger.info(f"Grid: {config.expected_rows} x {config.expected_cols}")
    logger.info(f"DEL (HYG-R): {del_count}")
    logger.info(f"WT (HYG-S): {wt_count}")
    
    if del_count + wt_count > 0:
        ratio = del_count / (del_count + wt_count)
        logger.info(f"DEL:WT ratio: {del_count}:{wt_count} ({ratio:.1%} DEL)")
        
        expected = (config.expected_rows * config.expected_cols) // 2
        if abs(del_count - expected) <= 3 and abs(wt_count - expected) <= 3:
            logger.success("Consistent with expected 1:1 segregation ratio")
    
    logger.info("=" * 60)
    
    return AnalysisResult(
        genotypes=genotypes,
        del_count=del_count,
        wt_count=wt_count,
        signals=signals,
        snr=snr,
        snr_threshold=config.snr_threshold,
        grid=grid,
        grid_params=grid_params,
        bg_mean=bg_mean,
        bg_std=bg_std,
        transform_matrix=transform_matrix,
        tetrad_centroids=tetrad_centroids,
        hyg_centroids=hyg_centroids
    )


# ============================================================
# Batch Processing
# ============================================================

def process_batch(
    data_file: Path | str,
    output_dir: Path | str,
    config: AnalysisConfig | None = None,
    tetrad_col: str = '6d_image_path',
    hyg_col: str = 'HYG_image_path'
) -> list[dict]:
    """
    Process a batch of image pairs from a CSV/TSV file.
    
    Args:
        data_file: Path to CSV/TSV file with image paths.
        output_dir: Directory to save results.
        config: Analysis configuration.
        tetrad_col: Column name for tetrad image paths.
        hyg_col: Column name for HYG image paths.
        
    Returns:
        List of result dictionaries.
    """
    import pandas as pd
    
    data_file = Path(data_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read data file
    if data_file.suffix == '.tsv':
        df = pd.read_csv(data_file, sep='\t')
    else:
        df = pd.read_csv(data_file)
    
    logger.info(f"Processing {len(df)} samples from {data_file.name}")
    
    results = []
    
    for idx, row in df.iterrows():
        try:
            tetrad_path = Path(row[tetrad_col])
            hyg_path = Path(row[hyg_col])
        except Exception as e:
            logger.error(f"Row {idx}: Failed to read paths: {e}")
            continue
        
        if not tetrad_path.exists() or not hyg_path.exists():
            logger.warning(f"Row {idx}: Files not found")
            logger.warning(f"   Tetrad: {tetrad_path}")
            logger.warning(f"   HYG: {hyg_path}")
            continue
        
        logger.info(f"\nProcessing: {tetrad_path.stem}")
        
        output_path = output_dir / f"{tetrad_path.stem}_result.png"
        
        result = analyze_tetrad_pair(
            tetrad_path, hyg_path,
            config=config,
            output_path=output_path,
            show_plot=False
        )
        
        if result:
            results.append({
                'sample': tetrad_path.stem,
                'tetrad_path': str(tetrad_path),
                'hyg_path': str(hyg_path),
                'del_count': result.del_count,
                'wt_count': result.wt_count,
                'rotation_angle': result.grid_params['rotation_angle'],
                'grid_error': result.grid_params['error_final'],
                'bg_mean': result.bg_mean,
                'bg_std': result.bg_std
            })
    
    # Save summary
    if results:
        summary_df = pd.DataFrame(results)
        summary_path = output_dir / 'analysis_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.success(f"Summary saved to {summary_path}")
    
    return results


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Tetrad Dissection HYG Selection Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single pair
  python colony_analyzer.py -t tetrad.png -h hyg.png -o result.png
  
  # Batch processing
  python colony_analyzer.py --batch data.csv --output-dir results/
  
  # Adjust SNR threshold
  python colony_analyzer.py -t tetrad.png -h hyg.png --snr-threshold 2.5
        """
    )
    
    # Single pair mode
    parser.add_argument('-t', '--tetrad', type=str, help='Tetrad plate image path')
    parser.add_argument('-y', '--hyg', type=str, help='HYG plate image path')
    parser.add_argument('-o', '--output', type=str, help='Output figure path')
    
    # Batch mode
    parser.add_argument('--batch', type=str, help='Batch CSV/TSV file')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for batch mode')
    
    # Parameters
    parser.add_argument('--snr-threshold', type=float, default=2.0,
                       help='SNR threshold for genotyping (default: 2.0)')
    parser.add_argument('--rows', type=int, default=4,
                       help='Expected number of rows (default: 4)')
    parser.add_argument('--cols', type=int, default=12,
                       help='Expected number of columns (default: 12)')
    parser.add_argument('--show', action='store_true',
                       help='Show plot interactively')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if not args.verbose:
        logger.remove()
        logger.add(
            lambda msg: print(msg, end=''),
            format="<level>{level: <8}</level> | {message}",
            level="INFO"
        )
    
    # Create config
    config = AnalysisConfig(
        snr_threshold=args.snr_threshold,
        expected_rows=args.rows,
        expected_cols=args.cols
    )
    
    # Batch mode
    if args.batch:
        process_batch(args.batch, args.output_dir, config)
        return
    
    # Single pair mode
    if not args.tetrad or not args.hyg:
        parser.error("Either --batch or both --tetrad and --hyg are required")
    
    result = analyze_tetrad_pair(
        args.tetrad, args.hyg,
        config=config,
        output_path=args.output,
        show_plot=args.show
    )
    
    if result is None:
        logger.error("Analysis failed")
        exit(1)


if __name__ == '__main__':
    main()
