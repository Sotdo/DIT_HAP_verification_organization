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
import os

# ================= 配置区域 =================
# 请替换为你本地图片的实际路径
TETRAD_PATH = "/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/17th_round/3d/302_meu23_3d_#2_202511.cropped.png"  # 基准图 (黑底)
HYG_PATH = "/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/17th_round/replica/302_meu23_HYG_#2_202511.cropped.png"    # 筛选图 (绿底/胶底)
MATCH_DIST_THRESHOLD = 40  # 匹配半径(像素)，如果对其不准可适当调大

# Grid configuration for tetrad dissection (12 tetrads × 4 spores)
EXPECTED_COLS = 12  # Number of tetrads
EXPECTED_ROWS = 4   # Spores per tetrad
# ===========================================


def align_images_by_centroids(img_ref, centroids_ref, img_target, centroids_target):
    """
    Align target image to reference image using centroid-based affine transformation.
    
    This function computes an optimal affine transformation (rotation, translation, scale)
    that best maps the target centroids to reference centroids using RANSAC.
    
    Args:
        img_ref: Reference image (tetrad plate)
        centroids_ref: List of (x, y) centroids from reference image
        img_target: Target image to be aligned (HYG plate)
        centroids_target: List of (x, y) centroids from target image
        
    Returns:
        aligned_img: Warped target image aligned to reference coordinates
        transform_matrix: 2x3 affine transformation matrix
        inliers_mask: Boolean mask indicating which point pairs were inliers
    """
    if len(centroids_ref) < 3 or len(centroids_target) < 3:
        print("警告: 检测到的菌落数量不足，无法进行精确对齐。使用简单缩放。")
        h, w = img_ref.shape[:2]
        aligned = cv2.resize(img_target, (w, h))
        return aligned, None, None
    
    # Convert to numpy arrays
    pts_ref = np.array(centroids_ref, dtype=np.float32)
    pts_target = np.array(centroids_target, dtype=np.float32)
    
    # Match centroids using Hungarian algorithm (optimal assignment)
    # Compute distance matrix between all pairs
    dist_matrix = distance.cdist(pts_ref, pts_target)
    
    # Use Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    # Filter matches by distance threshold (reject outliers)
    max_match_dist = max(img_ref.shape[:2]) * 0.15  # 15% of image size
    valid_matches = []
    matched_ref = []
    matched_target = []
    
    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] < max_match_dist:
            valid_matches.append((r, c, dist_matrix[r, c]))
            matched_ref.append(pts_ref[r])
            matched_target.append(pts_target[c])
    
    if len(matched_ref) < 3:
        print(f"警告: 仅匹配到 {len(matched_ref)} 个点对，不足以计算变换。使用简单缩放。")
        h, w = img_ref.shape[:2]
        aligned = cv2.resize(img_target, (w, h))
        return aligned, None, None
    
    matched_ref = np.array(matched_ref, dtype=np.float32)
    matched_target = np.array(matched_target, dtype=np.float32)
    
    print(f"成功匹配 {len(matched_ref)} 个菌落位置用于图像对齐。")
    
    # Estimate affine transformation with RANSAC
    # estimateAffinePartial2D: rotation + translation + uniform scale (4 DOF)
    # This is more robust than full affine for replica plating alignment
    M, inliers = cv2.estimateAffinePartial2D(
        matched_target, matched_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=2000,
        confidence=0.99
    )
    
    if M is None:
        print("警告: 无法估计变换矩阵。使用简单缩放。")
        h, w = img_ref.shape[:2]
        aligned = cv2.resize(img_target, (w, h))
        return aligned, None, None
    
    # Apply transformation
    h, w = img_ref.shape[:2]
    aligned = cv2.warpAffine(img_target, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Report transformation parameters
    scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
    angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    tx, ty = M[0, 2], M[1, 2]
    print(f"对齐变换: 缩放={scale:.3f}, 旋转={angle:.2f}°, 平移=({tx:.1f}, {ty:.1f})px")
    
    inlier_count = np.sum(inliers) if inliers is not None else len(matched_ref)
    print(f"RANSAC 内点数: {inlier_count}/{len(matched_ref)}")
    
    return aligned, M, inliers


def align_images_by_features(img_ref, img_target):
    """
    Align images using ORB feature matching (fallback method).
    
    This method works when colony detection fails but images have
    similar texture/features that can be matched.
    
    Args:
        img_ref: Reference image (tetrad plate)
        img_target: Target image to be aligned (HYG plate)
        
    Returns:
        aligned_img: Warped target image
        transform_matrix: 2x3 affine transformation matrix
    """
    # Convert to grayscale
    if img_ref.ndim == 3:
        gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
    else:
        gray_ref = img_ref
        
    if img_target.ndim == 3:
        gray_target = cv2.cvtColor(img_target, cv2.COLOR_RGB2GRAY)
    else:
        gray_target = img_target
    
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=500)
    
    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_target, None)
    
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        print("警告: 特征点检测失败，使用简单缩放。")
        h, w = img_ref.shape[:2]
        aligned = cv2.resize(img_target, (w, h))
        return aligned, None
    
    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Take top matches
    n_matches = min(50, len(matches))
    if n_matches < 4:
        print("警告: 匹配特征点不足，使用简单缩放。")
        h, w = img_ref.shape[:2]
        aligned = cv2.resize(img_target, (w, h))
        return aligned, None
    
    good_matches = matches[:n_matches]
    
    # Extract matched points
    pts_ref = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts_target = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # Estimate transformation
    M, inliers = cv2.estimateAffinePartial2D(pts_target, pts_ref, method=cv2.RANSAC)
    
    if M is None:
        h, w = img_ref.shape[:2]
        aligned = cv2.resize(img_target, (w, h))
        return aligned, None
    
    h, w = img_ref.shape[:2]
    aligned = cv2.warpAffine(img_target, M, (w, h))
    
    print(f"特征匹配对齐: 使用 {n_matches} 个特征点")
    
    return aligned, M


def compute_adaptive_threshold(centroids, image_width, expected_cols=EXPECTED_COLS):
    """
    Compute adaptive matching threshold based on detected colony spacing.
    
    Args:
        centroids: List of (x, y) colony centroids
        image_width: Width of the image
        expected_cols: Expected number of columns (tetrads)
        
    Returns:
        threshold: Adaptive distance threshold for colony matching
    """
    if len(centroids) < 2:
        # Fallback to grid-based estimate
        expected_spacing = image_width / expected_cols
        return expected_spacing * 0.4
    
    # Sort centroids by x coordinate
    sorted_x = sorted([c[0] for c in centroids])
    
    # Compute pairwise distances between adjacent x coordinates
    x_diffs = [sorted_x[i+1] - sorted_x[i] for i in range(len(sorted_x)-1)]
    
    # Filter out very small differences (within same column)
    min_col_spacing = image_width / (expected_cols * 2)
    col_spacings = [d for d in x_diffs if d > min_col_spacing]
    
    if col_spacings:
        median_spacing = np.median(col_spacings)
        # Use 35% of column spacing as threshold
        threshold = median_spacing * 0.35
    else:
        # Fallback
        threshold = image_width / expected_cols * 0.35
    
    return max(threshold, 20)  # Minimum 20 pixels


def analyze_replica_plating(tetrad_path, hyg_path, threshold_dist=None, use_adaptive_threshold=True):
    """
    Analyze replica plating results with robust image alignment.
    
    Args:
        tetrad_path: Path to tetrad plate image (reference)
        hyg_path: Path to HYG selection plate image
        threshold_dist: Fixed distance threshold for matching (optional)
        use_adaptive_threshold: If True, compute threshold from colony spacing
        
    Returns:
        del_count: Number of deletion colonies (HYG resistant)
        wt_count: Number of WT colonies (HYG sensitive)
    """
    # 1. 读取图片
    try:
        img_tetrad = io.imread(tetrad_path)
        img_hyg = io.imread(hyg_path)
    except Exception as e:
        print(f"Error: 无法读取图片. {e}")
        return None, None

    # Convert to uint8 if needed
    if img_tetrad.dtype != np.uint8:
        img_tetrad = (img_tetrad * 255).astype(np.uint8) if img_tetrad.max() <= 1 else img_tetrad.astype(np.uint8)
    if img_hyg.dtype != np.uint8:
        img_hyg = (img_hyg * 255).astype(np.uint8) if img_hyg.max() <= 1 else img_hyg.astype(np.uint8)

    print(f"Tetrad 图像尺寸: {img_tetrad.shape}")
    print(f"HYG 图像尺寸: {img_hyg.shape}")

    # 2. 核心检测函数
    def detect_colonies(img, is_hyg_plate=False, mask_zeros=False, split_fused=True):
        """
        Detect colony centroids in plate image with optional watershed-based splitting.
        
        Args:
            img: Input image
            is_hyg_plate: If True, use HYG-specific detection (red channel)
            mask_zeros: If True, exclude black (zero) pixels from thresholding
                       (useful for aligned images with black borders)
            split_fused: If True, use watershed algorithm to split fused colonies
        """
        if is_hyg_plate:
            # === HYG 平板特殊处理 ===
            # 提取红色通道 (R=0)，因为绿色背景在 R 通道下是黑色的，对比度最高
            if img.ndim == 3:
                gray = img[:, :, 0] 
            else:
                gray = img
            
            # 高斯模糊：使用较小的sigma以保留菌落边界，便于分割融合菌落
            # sigma=1.5 而不是 3，这样融合的菌落不会被过度模糊成一个
            blurred = filters.gaussian(gray, sigma=1.5, preserve_range=True)
            
            # 创建有效区域掩码（排除黑色边界）
            if mask_zeros:
                valid_mask = gray > 10  # 排除接近黑色的像素
                valid_pixels = blurred[valid_mask]
                if len(valid_pixels) > 100:
                    # 只对有效像素计算 Otsu 阈值
                    try:
                        thresh_val = filters.threshold_otsu(valid_pixels)
                        mask = (blurred > thresh_val) & valid_mask
                    except Exception:
                        mask = (blurred > 100) & valid_mask
                else:
                    mask = blurred > 100
            else:
                # Otsu 阈值分割
                try:
                    thresh_val = filters.threshold_otsu(blurred)
                    mask = blurred > thresh_val
                except Exception:
                    mask = blurred > 100 # 如果自动阈值失败，使用固定值
            
            # 形态学操作：去除细小的噪点，但不要过度腐蚀
            mask = morphology.binary_opening(mask, morphology.disk(2))
            min_area = 60 # 降低以检测被分割后的较小菌落
            # HYG菌落分割距离 - 基于典型菌落大小
            min_distance_for_split = 6
            # 估计的单个菌落半径（像素）
            expected_radius = 18
            
        else:
            # === Tetrad 平板处理 ===
            # 普通灰度化
            if img.ndim == 3:
                gray = np.mean(img, axis=2)
            else:
                gray = img
                
            blurred = filters.gaussian(gray, sigma=1)
            thresh_val = filters.threshold_otsu(blurred)
            mask = blurred > thresh_val
            min_area = 30 # Tetrad 菌落很清晰但可能较小
            # 较大的分割距离阈值，因为tetrad菌落分开较远
            min_distance_for_split = 20
            expected_radius = 15
        
        # === Watershed 分割融合菌落 ===
        if split_fused:
            mask, split_centroids = split_fused_colonies(mask, gray if 'gray' in dir() else blurred, 
                                        min_distance=min_distance_for_split,
                                        expected_colony_radius=expected_radius)
            # Use centroids from split_fused_colonies directly
            if split_centroids is not None and len(split_centroids) > 0:
                return split_centroids, mask
            
        # 连通区域标记 (only if not using split centroids)
        label_img = measure.label(mask)
        props = measure.regionprops(label_img)
        
        centroids = []
        for prop in props:
            if prop.area > min_area:
                # prop.centroid 返回 (row, col)，我们需要 (x, y) 作图
                y, x = prop.centroid
                centroids.append((x, y))
                
        return centroids, mask
    
    def split_fused_colonies(binary_mask, intensity_img, min_distance=20, expected_colony_radius=15):
        """
        Split fused colonies using watershed algorithm based on distance transform.
        
        This function identifies fused colonies (large connected components that 
        likely contain multiple colonies) and splits them using watershed segmentation.
        
        The algorithm:
        1. Compute distance transform - pixels at colony centers have high values
        2. Find local maxima (peaks) - each peak represents a colony center  
        3. For large/elongated regions, use erosion to find multiple centers
        4. Use peaks as markers for watershed segmentation
        5. Watershed splits fused regions at the "valleys" between peaks
        
        Args:
            binary_mask: Binary mask of detected colonies
            intensity_img: Grayscale intensity image for intensity-based splitting
            min_distance: Minimum distance between peaks for watershed markers
            expected_colony_radius: Expected radius of individual colonies
            
        Returns:
            split_mask: Binary mask with fused colonies split
        """
        # Ensure binary mask is proper boolean/int type
        binary_mask = binary_mask.astype(bool)
        
        # Compute distance transform - each pixel gets distance to nearest background
        distance_map = ndimage.distance_transform_edt(binary_mask)
        
        if distance_map.max() == 0:
            return binary_mask
        
        # First, analyze each connected component separately
        label_img = measure.label(binary_mask)
        props = measure.regionprops(label_img)
        
        all_coords = []
        
        # Print region analysis for first detection only (reset on each detect_colonies call)
        print(f"\n=== Fusion Detection Analysis ({len(props)} regions) ===")
        print(f"Expected single colony area: {np.pi * expected_colony_radius**2:.1f} px²")
        
        for prop in props:
            # Get the distance map for this component only
            component_mask = (label_img == prop.label)
            component_dist = distance_map * component_mask
            
            # Estimate expected number of colonies based on area
            single_colony_area = np.pi * expected_colony_radius**2
            estimated_colonies = max(1, int(prop.area / single_colony_area + 0.5))
            
            # Check shape metrics for fusion detection
            # Eccentricity > 0.6 suggests elongated (potentially fused)
            # Solidity < 0.9 suggests irregular shape (potentially fused)  
            # Also check if area is significantly larger than expected single colony
            area_ratio = prop.area / single_colony_area
            is_likely_fused = (prop.eccentricity > 0.6 or 
                              prop.solidity < 0.9 or 
                              area_ratio >= 1.5)  # 50% larger than expected
            
            component_max_dist = component_dist.max()
            
            # Debug print for each region (only for fused candidates)
            if is_likely_fused:
                print(f"  Region {prop.label}: area={prop.area:.0f}, ecc={prop.eccentricity:.2f}, "
                      f"sol={prop.solidity:.2f}, area_ratio={area_ratio:.1f}, est={estimated_colonies}")
            if component_max_dist > 0:
                if is_likely_fused:
                    # === Strategy for likely fused colonies ===
                    # Try multiple methods to find colony centers
                    
                    # Method 1: Distance transform peaks with very low min_distance
                    # For fused colonies, the peaks might be very close
                    local_min_dist = max(2, int(component_max_dist * 0.25))
                    coords = peak_local_max(
                        component_dist, 
                        min_distance=local_min_dist,
                        threshold_rel=0.1,  # Very low threshold
                        exclude_border=False
                    )
                    print(f"    -> M1 (peaks): {len(coords)}")
                    
                    # Method 2: If still only one peak, use h-maxima approach
                    # This finds peaks that are higher than their surroundings by at least h
                    if len(coords) <= 1 and (estimated_colonies >= 2 or area_ratio >= 1.5):
                        # h-maxima: suppress peaks that don't rise at least h above surroundings
                        h_value = component_max_dist * 0.15  # 15% of max distance
                        h_maxima = morphology.reconstruction(
                            component_dist - h_value, 
                            component_dist,
                            method='dilation'
                        )
                        h_peaks = component_dist - h_maxima
                        
                        coords = peak_local_max(
                            h_peaks,
                            min_distance=max(2, int(component_max_dist * 0.2)),
                            threshold_rel=0.05,
                            exclude_border=False
                        )
                        if len(coords) > 1:
                            print(f"    -> M2 (h-maxima): {len(coords)}")
                    
                    # Method 3: If still only one peak, try erosion-based approach
                    if len(coords) <= 1 and (estimated_colonies >= 2 or area_ratio >= 1.5):
                        # Erode until we get separate components
                        erode_radius = max(2, int(component_max_dist * 0.4))
                        eroded = morphology.binary_erosion(
                            component_mask, 
                            morphology.disk(erode_radius)
                        )
                        
                        # Find centroids of eroded components
                        eroded_labels = measure.label(eroded)
                        eroded_props = measure.regionprops(eroded_labels)
                        
                        if len(eroded_props) >= 2:
                            # We successfully split by erosion
                            coords = np.array([[int(p.centroid[0]), int(p.centroid[1])] 
                                              for p in eroded_props])
                            print(f"    -> M3 (erosion): {len(coords)}")
                        elif len(eroded_props) <= 1:
                            # Still one component, try with more erosion
                            for erosion_factor in [0.5, 0.6, 0.7]:
                                erode_radius = max(3, int(component_max_dist * erosion_factor))
                                eroded = morphology.binary_erosion(
                                    component_mask, 
                                    morphology.disk(erode_radius)
                                )
                                eroded_labels = measure.label(eroded)
                                eroded_props = measure.regionprops(eroded_labels)
                                if len(eroded_props) >= 2:
                                    coords = np.array([[int(p.centroid[0]), int(p.centroid[1])] 
                                                      for p in eroded_props])
                                    print(f"    -> M3b (erosion): {len(coords)}")
                                    break
                    
                    # Method 4: Axis-based splitting for round but large regions
                    # When area_ratio suggests fusion but shape looks round (low eccentricity),
                    # split along the major axis based on estimated colony count
                    # Also trigger if we found fewer peaks than estimated colonies for very large regions
                    should_axis_split = (
                        (len(coords) <= 1 and area_ratio >= 1.5 and estimated_colonies >= 2) or
                        (len(coords) < estimated_colonies and area_ratio >= 2.5)  # Very large but underdetected
                    )
                    if should_axis_split:
                        # Split along major axis into estimated_colonies parts
                        centroid_y, centroid_x = prop.centroid
                        orientation = prop.orientation  # Angle in radians
                        major_axis = prop.major_axis_length
                        
                        # Calculate single colony expected diameter
                        single_diameter = 2 * expected_colony_radius
                        
                        # How many colonies could fit along major axis?
                        n_split = min(estimated_colonies, max(2, int(major_axis / single_diameter)))
                        
                        # Only use axis split if it would give us more colonies than current
                        if n_split > len(coords):
                            # Create split points along major axis
                            split_coords = []
                            for i in range(n_split):
                                # Position along axis (centered)
                                t = (i - (n_split - 1) / 2) * (major_axis * 0.8 / max(1, n_split - 1)) if n_split > 1 else 0
                                # Convert to image coordinates (orientation is counter-clockwise from horizontal)
                                dy = t * np.sin(orientation)
                                dx = t * np.cos(orientation)
                                new_y = centroid_y - dy  # negative because image y increases downward
                                new_x = centroid_x + dx
                                
                                # Only add if within component mask
                                in_bounds = (0 <= int(new_y) < component_mask.shape[0] and 
                                           0 <= int(new_x) < component_mask.shape[1])
                                in_mask = in_bounds and component_mask[int(new_y), int(new_x)]
                                
                                if in_mask:
                                    split_coords.append([int(new_y), int(new_x)])
                            
                            if len(split_coords) > len(coords):
                                coords = np.array(split_coords)
                                print(f"    -> M4 (axis): {len(coords)}")
                    
                    # Method 5: For very large complex regions, use k-means clustering
                    # This works for L-shaped or branching fused colonies where axis split fails
                    if len(coords) < estimated_colonies and area_ratio >= 3:
                        # Get all pixels in the component, weighted by distance from edge
                        pixel_coords = np.argwhere(component_mask)
                        pixel_weights = component_dist[component_mask]
                        
                        # Use top 50% of distance-weighted pixels (interior pixels)
                        weight_threshold = np.percentile(pixel_weights, 50)
                        interior_mask = pixel_weights >= weight_threshold
                        interior_coords = pixel_coords[interior_mask]
                        
                        if len(interior_coords) >= estimated_colonies * 5:
                            # Simple k-means using iterative centroid refinement
                            from scipy.cluster.vq import kmeans2
                            try:
                                # Run k-means
                                centroids_km, _ = kmeans2(
                                    interior_coords.astype(float), 
                                    estimated_colonies, 
                                    minit='points',
                                    iter=20
                                )
                                
                                # Verify each centroid is within the mask
                                valid_centroids = []
                                for cy, cx in centroids_km:
                                    iy, ix = int(cy), int(cx)
                                    if 0 <= iy < component_mask.shape[0] and 0 <= ix < component_mask.shape[1]:
                                        if component_mask[iy, ix]:
                                            valid_centroids.append([iy, ix])
                                
                                if len(valid_centroids) > len(coords):
                                    coords = np.array(valid_centroids)
                                    print(f"    -> M5 (k-means): {len(coords)}")
                            except Exception:
                                pass  # k-means failed silently
                        
                    # Method 6: For very large regions, try skeleton-based splitting
                    # If the region has significant concavities, split at the narrowest point
                    if len(coords) <= 1 and area_ratio >= 1.8:
                        # Find the skeleton and look for branch points or narrow sections
                        skeleton = morphology.skeletonize(component_mask)
                        skel_coords = np.argwhere(skeleton)
                        
                        if len(skel_coords) > 10:
                            # Use distance transform on skeleton to find narrow points
                            skel_dist = component_dist * skeleton
                            
                            # Find peaks along skeleton (centers of overlapping colonies)
                            skel_peaks = peak_local_max(
                                skel_dist,
                                min_distance=max(3, int(component_max_dist * 0.3)),
                                threshold_rel=0.1,
                                exclude_border=False
                            )
                            
                            if len(skel_peaks) >= 2:
                                coords = skel_peaks
                                print(f"    -> M6 (skeleton): {len(coords)}")
                    
                    # Print final result only if multiple colonies found
                    if len(coords) >= 2:
                        print(f"    -> FINAL: {len(coords)}")
                else:
                    # Single colony - just find the peak
                    local_min_dist = max(min_distance, int(component_max_dist * 0.6))
                    coords = peak_local_max(
                        component_dist, 
                        min_distance=local_min_dist,
                        threshold_rel=0.3,
                        exclude_border=False
                    )
                
                if len(coords) > 0:
                    all_coords.extend(coords.tolist())
        
        if len(all_coords) == 0:
            return binary_mask, None
        
        coords = np.array(all_coords)
        
        # Convert coords (row, col) to centroids (x, y) format
        centroids = [(float(c), float(r)) for r, c in coords]
        
        # Create marker image - each peak gets a unique label
        markers = np.zeros(distance_map.shape, dtype=np.int32)
        for i, (r, c) in enumerate(coords):
            markers[r, c] = i + 1
        
        # Dilate markers slightly to make them more robust
        struct = morphology.disk(2)
        dilated_markers = np.zeros_like(markers)
        for label_val in range(1, len(coords) + 1):
            single_marker = (markers == label_val)
            dilated = morphology.binary_dilation(single_marker, struct)
            dilated_markers[dilated & (dilated_markers == 0)] = label_val
        
        markers = dilated_markers
        
        # Apply watershed on negative distance
        labels = watershed(-distance_map, markers, mask=binary_mask)
        
        print(f"  => Total colonies: {len(centroids)} (from {len(props)} regions)")
        
        split_mask = labels > 0
        
        return split_mask, centroids

    # 3. 初步检测菌落（用于对齐）
    print("\n=== 第一阶段：初步检测菌落位置 ===")
    print("正在检测 Tetrad 平板...")
    tetrad_centroids_initial, tetrad_mask = detect_colonies(img_tetrad, is_hyg_plate=False)
    print(f"检测到 {len(tetrad_centroids_initial)} 个孢子。")
    
    # 先将 HYG 图像缩放到与 Tetrad 相同尺寸（粗对齐）
    h_ref, w_ref = img_tetrad.shape[:2]
    img_hyg_resized = cv2.resize(img_hyg, (w_ref, h_ref))
    
    print("正在检测 HYG 平板 (初步)...")
    hyg_centroids_initial, _ = detect_colonies(img_hyg_resized, is_hyg_plate=True)
    print(f"检测到 {len(hyg_centroids_initial)} 个抗性克隆。")

    # 4. 图像对齐
    print("\n=== 第二阶段：图像精确对齐 ===")
    
    # 尝试基于菌落质心的对齐
    if len(tetrad_centroids_initial) >= 6 and len(hyg_centroids_initial) >= 3:
        print("使用菌落质心进行图像对齐...")
        img_hyg_aligned, transform_matrix, _ = align_images_by_centroids(
            img_tetrad, tetrad_centroids_initial,
            img_hyg_resized, hyg_centroids_initial
        )
        alignment_method = "centroid"
    else:
        # 回退到特征点对齐
        print("菌落检测不足，尝试特征点对齐...")
        img_hyg_aligned, transform_matrix = align_images_by_features(img_tetrad, img_hyg_resized)
        alignment_method = "feature" if transform_matrix is not None else "resize"
    
    # 5. 获取对齐后的 HYG 菌落位置
    print("\n=== 第三阶段：获取对齐后的 HYG 菌落位置 ===")
    
    if transform_matrix is not None:
        # 使用变换矩阵将初始 HYG 质心转换到对齐坐标系
        # 这比在对齐图像上重新检测更可靠，不会丢失边缘的菌落
        print("使用变换矩阵转换 HYG 菌落坐标...")
        hyg_centroids_np = np.array(hyg_centroids_initial, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.transform(hyg_centroids_np, transform_matrix)
        hyg_centroids = [(float(pt[0][0]), float(pt[0][1])) for pt in transformed]
        print(f"转换后 HYG 菌落数: {len(hyg_centroids)}")
        
        # 同时也在对齐图像上重新检测，取两者中较多的结果
        print("同时在对齐图像上重新检测...")
        hyg_centroids_redetect, hyg_mask = detect_colonies(img_hyg_aligned, is_hyg_plate=True, mask_zeros=True)
        print(f"重新检测到: {len(hyg_centroids_redetect)} 个抗性克隆")
        
        # 使用检测到更多菌落的结果
        if len(hyg_centroids_redetect) > len(hyg_centroids):
            print(f"使用重新检测结果 ({len(hyg_centroids_redetect)} > {len(hyg_centroids)})")
            hyg_centroids = hyg_centroids_redetect
        else:
            print(f"使用变换坐标结果 ({len(hyg_centroids)} >= {len(hyg_centroids_redetect)})")
            # 需要重新生成 mask 用于可视化
            _, hyg_mask = detect_colonies(img_hyg_aligned, is_hyg_plate=True, mask_zeros=True)
    else:
        # 没有变换矩阵，直接在对齐图像上检测
        print("正在检测对齐后的 HYG 平板...")
        hyg_centroids, hyg_mask = detect_colonies(img_hyg_aligned, is_hyg_plate=True, mask_zeros=True)
        print(f"检测到 {len(hyg_centroids)} 个抗性克隆。")
    
    # 使用初步检测的 tetrad centroids
    tetrad_centroids = tetrad_centroids_initial

    # 6. 计算自适应阈值
    if use_adaptive_threshold and threshold_dist is None:
        threshold_dist = compute_adaptive_threshold(tetrad_centroids, w_ref, EXPECTED_COLS)
        print(f"\n自适应匹配阈值: {threshold_dist:.1f} 像素")
    elif threshold_dist is None:
        threshold_dist = MATCH_DIST_THRESHOLD
        print(f"\n使用默认匹配阈值: {threshold_dist} 像素")
    else:
        print(f"\n使用指定匹配阈值: {threshold_dist} 像素")

    # 7. 匹配逻辑 (Spatial Matching)
    if not hyg_centroids:
        hyg_centroids = [(-1000, -1000)] # 防止为空报错

    # 计算距离矩阵
    dists = distance.cdist(tetrad_centroids, hyg_centroids)
    
    wt_count = 0
    del_count = 0
    results = []  # Store detailed results
    
    # Debug: track matching statistics
    matched_hyg_indices = set()
    unmatched_tetrad_details = []
    tetrad_to_hyg = {}  # Track which tetrad matched to which HYG
    
    for i, (cX, cY) in enumerate(tetrad_centroids):
        # 寻找最近的 HYG 菌落
        min_dist = np.min(dists[i])
        nearest_hyg_idx = np.argmin(dists[i])
        
        if min_dist < threshold_dist:
            genotype = "DEL"
            del_count += 1
            matched_hyg_indices.add(nearest_hyg_idx)
            tetrad_to_hyg[i] = (nearest_hyg_idx, min_dist)
        else:
            genotype = "WT"
            wt_count += 1
            if min_dist < threshold_dist * 2:  # Close misses
                unmatched_tetrad_details.append((i, cX, cY, min_dist, nearest_hyg_idx))
        
        results.append({
            'x': cX, 'y': cY, 
            'genotype': genotype, 
            'min_dist': min_dist
        })
    
    # Check for unmatched HYG colonies
    unmatched_hyg = set(range(len(hyg_centroids))) - matched_hyg_indices
    
    # Debug output
    print(f"\n匹配结果: {len(hyg_centroids)} HYG菌落中 {len(matched_hyg_indices)} 个匹配到四分体位置")
    
    if len(unmatched_hyg) > 0:
        print(f"  (有 {len(unmatched_hyg)} 个HYG菌落未匹配，可能是过度分割或噪点)")
    
    # 8. 可视化绘图 (3 panels)
    print("\n=== 生成结果可视化 ===")
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # 左图：原始 HYG 平板
    ax[0].imshow(img_hyg_resized)
    ax[0].set_title("Original HYG Plate\n(Before Alignment)")
    ax[0].axis('off')
    
    # 中图：对齐后的 HYG 平板（带检测结果）
    ax[1].imshow(img_hyg_aligned)
    ax[1].imshow(hyg_mask, alpha=0.3, cmap='spring')
    for (hx, hy) in hyg_centroids:
        if hx > 0:  # Skip dummy centroid
            ax[1].plot(hx, hy, 'c+', markersize=10, markeredgewidth=2)
    ax[1].set_title(f"Aligned HYG Plate\n(Method: {alignment_method}, Detected: {len(hyg_centroids)})")
    ax[1].axis('off')
    
    # 右图：最终基因型结果
    ax[2].imshow(img_tetrad)
    ax[2].imshow(tetrad_mask, alpha=0.2, cmap='gray')
    
    for res in results:
        cX, cY = res['x'], res['y']
        genotype = res['genotype']
        
        if genotype == "DEL":
            color = 'lime'
        else:
            color = 'red'
            
        # 画圈标记
        circ = Circle((cX, cY), radius=20, color=color, fill=False, linewidth=2)
        ax[2].text(cX, cY-25, genotype, color=color, fontsize=7, ha='center', fontweight='bold')
        ax[2].add_patch(circ)
        
    ax[2].set_title(f"Genotyping Result\nDeletion(HYG-R)={del_count}, WT(HYG-S)={wt_count}\nThreshold={threshold_dist:.1f}px")
    ax[2].axis('off')
    
    plt.tight_layout()
    
    # Save with informative filename
    output_name = "replica_plating_analysis_result.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"结果已保存至: {output_name}")
    
    plt.show()
    plt.close()
    
    # Print summary
    print("\n=== 分析结果 ===")
    print(f"总检测孢子数: {len(tetrad_centroids)}")
    print(f"Deletion (HYG抗性): {del_count}")
    print(f"WT (HYG敏感): {wt_count}")
    if del_count + wt_count > 0:
        print(f"Del:WT 比例: {del_count}:{wt_count}")
        expected_ratio = len(tetrad_centroids) // 2
        if abs(del_count - expected_ratio) <= 2 and abs(wt_count - expected_ratio) <= 2:
            print("✓ 符合预期的 1:1 分离比例")
    
    return del_count, wt_count

# 运行脚本
if __name__ == "__main__":
    if os.path.exists(TETRAD_PATH) and os.path.exists(HYG_PATH):
        print("="*60)
        print("四分体解剖 HYG 筛选分析 (带图像对齐)")
        print("="*60)
        del_count, wt_count = analyze_replica_plating(
            TETRAD_PATH, HYG_PATH,
            use_adaptive_threshold=True
        )
    else:
        print("请检查文件路径是否正确。")
        print(f"Tetrad: {TETRAD_PATH}")
        print(f"HYG: {HYG_PATH}")