"""
Tetrad dissection HYG selection analysis with image alignment.

This module analyzes replica plating results from tetrad dissection experiments
to distinguish deletion (HYG-resistant) from WT (HYG-sensitive) colonies.

Usage:
    from src.align_and_measure_colony import analyze_replica_plating, AnalysisConfig
    
    config = AnalysisConfig(
        expected_cols=12,
        expected_rows=4,
        match_dist_threshold=40
    )
    
    del_count, wt_count = analyze_replica_plating(
        tetrad_path='path/to/tetrad.png',
        hyg_path='path/to/hyg.png',
        config=config
    )
"""

# %% ------------------------------------ Imports ------------------------------------ #
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
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
    
    # Matching parameters
    match_dist_threshold: int = 40  # Default distance threshold (pixels)
    use_adaptive_threshold: bool = True  # Auto-compute threshold from spacing
    
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
    
    # Output
    output_filename: str = "replica_plating_analysis_result.png"
    output_dpi: int = 300


# %% ------------------------------------ Image Alignment ------------------------------------ #
def align_images_by_centroids(
    img_ref: np.ndarray,
    centroids_ref: list[tuple[float, float]],
    img_target: np.ndarray,
    centroids_target: list[tuple[float, float]],
    config: AnalysisConfig
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Align target image to reference using centroid-based affine transformation.
    
    Uses Hungarian algorithm for optimal centroid matching, then estimates
    an affine transformation (rotation + translation + scale) via RANSAC.
    
    Args:
        img_ref: Reference image (tetrad plate)
        centroids_ref: Colony centroids from reference image [(x, y), ...]
        img_target: Target image to align (HYG plate)
        centroids_target: Colony centroids from target image
        config: Analysis configuration
        
    Returns:
        Tuple of (aligned_image, transform_matrix, inliers_mask)
    """
    h, w = img_ref.shape[:2]
    
    # Need at least 3 points for affine estimation
    if len(centroids_ref) < 3 or len(centroids_target) < 3:
        print("警告: 菌落数量不足，使用简单缩放。")
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
        print(f"警告: 仅匹配到 {len(matched_ref)} 个点对，使用简单缩放。")
        return cv2.resize(img_target, (w, h)), None, None
    
    matched_ref = np.array(matched_ref, dtype=np.float32)
    matched_target = np.array(matched_target, dtype=np.float32)
    
    print(f"成功匹配 {len(matched_ref)} 个菌落位置用于图像对齐。")
    
    # Estimate affine transformation
    M, inliers = cv2.estimateAffinePartial2D(
        matched_target, matched_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_reproj_threshold,
        maxIters=config.ransac_max_iters,
        confidence=config.ransac_confidence
    )
    
    if M is None:
        print("警告: 无法估计变换矩阵，使用简单缩放。")
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
    
    print(f"对齐变换: 缩放={scale:.3f}, 旋转={angle:.2f}°, 平移=({tx:.1f}, {ty:.1f})px")
    print(f"RANSAC 内点数: {inlier_count}/{len(matched_ref)}")
    
    return aligned, M, inliers


def align_images_by_features(
    img_ref: np.ndarray,
    img_target: np.ndarray
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Align images using ORB feature matching (fallback method).
    
    Args:
        img_ref: Reference image
        img_target: Target image to align
        
    Returns:
        Tuple of (aligned_image, transform_matrix)
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
        print("警告: 特征点检测失败，使用简单缩放。")
        return cv2.resize(img_target, (w, h)), None
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    
    n_matches = min(50, len(matches))
    if n_matches < 4:
        print("警告: 匹配特征点不足，使用简单缩放。")
        return cv2.resize(img_target, (w, h)), None
    
    # Extract matched points
    pts_ref = np.float32([kp1[m.queryIdx].pt for m in matches[:n_matches]])
    pts_target = np.float32([kp2[m.trainIdx].pt for m in matches[:n_matches]])
    
    # Estimate transformation
    M, _ = cv2.estimateAffinePartial2D(pts_target, pts_ref, method=cv2.RANSAC)
    
    if M is None:
        return cv2.resize(img_target, (w, h)), None
    
    aligned = cv2.warpAffine(img_target, M, (w, h))
    print(f"特征匹配对齐: 使用 {n_matches} 个特征点")
    
    return aligned, M


# %% ------------------------------------ Colony Detection ------------------------------------ #
def detect_colonies(
    img: np.ndarray,
    is_hyg_plate: bool = False,
    config: AnalysisConfig = None,
    mask_zeros: bool = False
) -> tuple[list[tuple[float, float]], np.ndarray]:
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
    
    Args:
        binary_mask: Binary mask of detected colonies
        min_distance: Minimum distance between colony centers
        min_area_for_split: Only split regions larger than this
        
    Returns:
        Binary mask with fused colonies split
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


# %% ------------------------------------ Threshold Computation ------------------------------------ #
def compute_adaptive_threshold(
    centroids: list[tuple[float, float]],
    image_width: int,
    expected_cols: int
) -> float:
    """
    Compute adaptive matching threshold based on colony spacing.
    
    Args:
        centroids: List of colony centroids
        image_width: Width of the image
        expected_cols: Expected number of columns
        
    Returns:
        Distance threshold in pixels
    """
    if len(centroids) < 2:
        return image_width / expected_cols * 0.4
    
    # Sort by x coordinate
    sorted_x = sorted([c[0] for c in centroids])
    
    # Compute adjacent x differences
    x_diffs = [sorted_x[i+1] - sorted_x[i] for i in range(len(sorted_x)-1)]
    
    # Filter out small differences (within same column)
    min_col_spacing = image_width / (expected_cols * 2)
    col_spacings = [d for d in x_diffs if d > min_col_spacing]
    
    if col_spacings:
        threshold = np.median(col_spacings) * 0.35
    else:
        threshold = image_width / expected_cols * 0.35
    
    return max(threshold, 20)  # Minimum 20 pixels


# %% ------------------------------------ Main Analysis ------------------------------------ #
def analyze_replica_plating(
    tetrad_path: str | Path,
    hyg_path: str | Path,
    config: AnalysisConfig = None,
    threshold_dist: float = None,
    pic_result_path: str = None
) -> tuple[int, int]:
    """
    Analyze replica plating results with robust image alignment.
    
    Args:
        tetrad_path: Path to tetrad plate image (reference)
        hyg_path: Path to HYG selection plate image
        config: Analysis configuration
        threshold_dist: Fixed distance threshold (overrides adaptive)
        
    Returns:
        Tuple of (deletion_count, wt_count)
    """
    if config is None:
        config = AnalysisConfig()
    
    # 1. Load images
    try:
        img_tetrad = io.imread(str(tetrad_path))
        img_hyg = io.imread(str(hyg_path))
    except Exception as e:
        print(f"Error: 无法读取图片. {e}")
        return None, None
    
    # Convert to uint8
    for img in [img_tetrad, img_hyg]:
        if img.dtype != np.uint8:
            if img.max() <= 1:
                img[:] = (img * 255).astype(np.uint8)
            else:
                img[:] = img.astype(np.uint8)
    
    print(f"Tetrad 图像尺寸: {img_tetrad.shape}")
    print(f"HYG 图像尺寸: {img_hyg.shape}")
    
    # 2. Initial colony detection
    print("\n=== 第一阶段：初步检测菌落位置 ===")
    
    print("正在检测 Tetrad 平板...")
    tetrad_centroids, tetrad_mask = detect_colonies(
        img_tetrad, is_hyg_plate=False, config=config
    )
    print(f"检测到 {len(tetrad_centroids)} 个孢子。")
    
    # Resize HYG to match tetrad
    h_ref, w_ref = img_tetrad.shape[:2]
    img_hyg_resized = cv2.resize(img_hyg, (w_ref, h_ref))
    
    print("正在检测 HYG 平板 (初步)...")
    hyg_centroids_initial, _ = detect_colonies(
        img_hyg_resized, is_hyg_plate=True, config=config
    )
    print(f"检测到 {len(hyg_centroids_initial)} 个抗性克隆。")
    
    # 3. Image alignment
    print("\n=== 第二阶段：图像精确对齐 ===")
    
    if len(tetrad_centroids) >= 6 and len(hyg_centroids_initial) >= 3:
        print("使用菌落质心进行图像对齐...")
        img_hyg_aligned, transform_matrix, _ = align_images_by_centroids(
            img_tetrad, tetrad_centroids,
            img_hyg_resized, hyg_centroids_initial,
            config
        )
        alignment_method = "centroid"
    else:
        print("菌落检测不足，尝试特征点对齐...")
        img_hyg_aligned, transform_matrix = align_images_by_features(
            img_tetrad, img_hyg_resized
        )
        alignment_method = "feature" if transform_matrix is not None else "resize"
    
    # 4. Re-detect HYG colonies on aligned image
    print("\n=== 第三阶段：在对齐图像上重新检测 ===")
    
    print("正在检测对齐后的 HYG 平板...")
    hyg_centroids, hyg_mask = detect_colonies(
        img_hyg_aligned, is_hyg_plate=True, config=config, mask_zeros=True
    )
    print(f"检测到 {len(hyg_centroids)} 个抗性克隆。")
    
    # 5. Compute threshold
    if threshold_dist is not None:
        print(f"\n使用指定匹配阈值: {threshold_dist} 像素")
    elif config.use_adaptive_threshold:
        threshold_dist = compute_adaptive_threshold(
            tetrad_centroids, w_ref, config.expected_cols
        )
        print(f"\n自适应匹配阈值: {threshold_dist:.1f} 像素")
    else:
        threshold_dist = config.match_dist_threshold
        print(f"\n使用默认匹配阈值: {threshold_dist} 像素")
    
    # 6. Spatial matching
    if not hyg_centroids:
        hyg_centroids = [(-1000, -1000)]  # Dummy to avoid empty array
    
    dists = distance.cdist(tetrad_centroids, hyg_centroids)
    
    results = []
    del_count, wt_count = 0, 0
    
    for i, (cX, cY) in enumerate(tetrad_centroids):
        min_dist = np.min(dists[i])
        
        if min_dist < threshold_dist:
            genotype = "DEL"
            del_count += 1
        else:
            genotype = "WT"
            wt_count += 1
        
        results.append({'x': cX, 'y': cY, 'genotype': genotype, 'min_dist': min_dist})
    
    # 7. Visualization
    print("\n=== 生成结果可视化 ===")
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Original HYG
    ax[0].imshow(img_hyg_resized)
    ax[0].set_title("Original HYG Plate\n(Before Alignment)")
    ax[0].axis('off')
    
    # Panel 2: Aligned HYG with detection
    ax[1].imshow(img_hyg_aligned)
    ax[1].imshow(hyg_mask, alpha=0.3, cmap='spring')
    for (hx, hy) in hyg_centroids:
        if hx > 0:
            ax[1].plot(hx, hy, 'c+', markersize=10, markeredgewidth=2)
    ax[1].set_title(f"Aligned HYG Plate\n(Method: {alignment_method}, Detected: {len(hyg_centroids)})")
    ax[1].axis('off')
    
    # Panel 3: Genotyping result
    ax[2].imshow(img_tetrad)
    ax[2].imshow(tetrad_mask, alpha=0.2, cmap='gray')
    
    for res in results:
        cX, cY = res['x'], res['y']
        color = 'lime' if res['genotype'] == "DEL" else 'red'
        
        circ = Circle((cX, cY), radius=20, color=color, fill=False, linewidth=2)
        ax[2].text(cX, cY-25, res['genotype'], color=color, fontsize=7, 
                   ha='center', fontweight='bold')
        ax[2].add_patch(circ)
    
    ax[2].set_title(
        f"Genotyping Result\n"
        f"Deletion(HYG-R)={del_count}, WT(HYG-S)={wt_count}\n"
        f"Threshold={threshold_dist:.1f}px"
    )
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(pic_result_path, dpi=config.output_dpi, bbox_inches='tight')
    print(f"结果已保存至: {pic_result_path}")
    
    plt.show()
    plt.close()
    
    # 8. Summary
    print("\n=== 分析结果 ===")
    print(f"总检测孢子数: {len(tetrad_centroids)}")
    print(f"Deletion (HYG抗性): {del_count}")
    print(f"WT (HYG敏感): {wt_count}")
    
    if del_count + wt_count > 0:
        print(f"Del:WT 比例: {del_count}:{wt_count}")
        expected = len(tetrad_centroids) // 2
        if abs(del_count - expected) <= 2 and abs(wt_count - expected) <= 2:
            print("✓ 符合预期的 1:1 分离比例")
    
    return del_count, wt_count


# %% ------------------------------------ Main ------------------------------------ #
if __name__ == "__main__":
    import os
    
    # Example paths - modify these for your analysis
    # data_folder = "/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion"
    # tetrad_path = f"{data_folder}/17th_round/6d/303_omh3_6d_#2_202511.cropped.png"
    # hyg_path = f"{data_folder}/17th_round/replica/303_omh3_HYG_#2_202511.cropped.png"

    data = pd.read_excel("./results/all_rounds_combined_verification_summary.xlsx")
    
    # Configuration
    config = AnalysisConfig(
        expected_cols=12,
        expected_rows=4,
        use_adaptive_threshold=True,
        split_fused_colonies=True
    )
    
    for idx, row in data.iterrows():
        try:
            tetrad_path = Path(row['6d_image_path'])
            hyg_path = Path(row['HYG_image_path'])
        except Exception as e:
            print("路径读取错误:", e)
            continue
        print(f"\n处理样本: {tetrad_path.stem} 和 {hyg_path.stem}")
        if os.path.exists(tetrad_path) and os.path.exists(hyg_path):
            print("=" * 60)
            print("四分体解剖 HYG 筛选分析 (带图像对齐)")
            print("=" * 60)
            
            del_count, wt_count = analyze_replica_plating(
                tetrad_path, hyg_path, config=config, 
                pic_result_path=f"./results/selection_marker_test/{tetrad_path.stem}_result.png"
            )
        else:
            print("请检查文件路径是否正确。")
            print(f"Tetrad: {tetrad_path}")
            print(f"HYG: {hyg_path}")
