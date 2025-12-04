import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from skimage import io, filters, measure, transform, morphology
import os

# ================= 配置区域 =================
# 请替换为你本地图片的实际路径
TETRAD_PATH = "/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/17th_round/3d/302_meu23_3d_#1_202511.cropped.png"  # 基准图 (黑底)
HYG_PATH = "/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion/17th_round/replica/302_meu23_HYG_#1_202511.cropped.png"    # 筛选图 (绿底/胶底)
MATCH_DIST_THRESHOLD = 40  # 匹配半径(像素)，如果对其不准可适当调大
# ===========================================

def analyze_replica_plating(tetrad_path, hyg_path, threshold_dist=40):
    # 1. 读取图片
    try:
        img_tetrad = io.imread(tetrad_path)
        img_hyg = io.imread(hyg_path)
    except Exception as e:
        print(f"Error: 无法读取图片. {e}")
        return

    # 2. 尺寸同步：将 HYG 图缩放到 Tetrad 图的大小
    target_shape = img_tetrad.shape[:2]
    # preserve_range=True 保持数值在 0-255 之间
    img_hyg = transform.resize(img_hyg, target_shape, preserve_range=True).astype(np.uint8)

    # 3. 核心检测函数
    def detect_colonies(img, is_hyg_plate=False):
        if is_hyg_plate:
            # === HYG 平板特殊处理 ===
            # 提取红色通道 (R=0)，因为绿色背景在 R 通道下是黑色的，对比度最高
            if img.ndim == 3:
                gray = img[:, :, 0] 
            else:
                gray = img
            
            # 高斯模糊：模糊半径大一点 (sigma=3)，把模糊的菌斑融合成一个整块
            blurred = filters.gaussian(gray, sigma=3, preserve_range=True)
            
            # Otsu 阈值分割
            try:
                thresh_val = filters.threshold_otsu(blurred)
                mask = blurred > thresh_val
            except:
                mask = blurred > 100 # 如果自动阈值失败，使用固定值
            
            # 形态学操作：去除细小的噪点
            mask = morphology.binary_opening(mask, morphology.disk(3))
            min_area = 100 # HYG 菌落通常比较弥散，阈值设大点
            
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
            
        # 连通区域标记
        label_img = measure.label(mask)
        props = measure.regionprops(label_img)
        
        centroids = []
        for prop in props:
            if prop.area > min_area:
                # prop.centroid 返回 (row, col)，我们需要 (x, y) 作图
                y, x = prop.centroid
                centroids.append((x, y))
                
        return centroids, mask

    # 4. 执行检测
    print("正在检测 Tetrad 平板...")
    tetrad_centroids, _ = detect_colonies(img_tetrad, is_hyg_plate=False)
    print(f"检测到 {len(tetrad_centroids)} 个孢子。")
    
    print("正在检测 HYG 平板...")
    hyg_centroids, hyg_mask = detect_colonies(img_hyg, is_hyg_plate=True)
    print(f"检测到 {len(hyg_centroids)} 个抗性克隆。")

    # 5. 匹配逻辑 (Spatial Matching)
    if not hyg_centroids:
        hyg_centroids = [(-1000, -1000)] # 防止为空报错

    # 计算距离矩阵
    dists = distance.cdist(tetrad_centroids, hyg_centroids)
    
    wt_count = 0
    del_count = 0
    
    # 6. 可视化绘图
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：HYG 检测情况 (Debugging)
    ax[0].imshow(img_hyg)
    ax[0].imshow(hyg_mask, alpha=0.3, cmap='spring') # 粉色覆盖层表示识别到的区域
    ax[0].set_title(f"HYG Selection Plate\n(Detected: {len(hyg_centroids)})")
    ax[0].axis('off')
    
    # 右图：最终结果
    ax[1].imshow(img_tetrad)
    ax[1].set_title("Genotyping Result")
    ax[1].axis('off')
    
    for i, (cX, cY) in enumerate(tetrad_centroids):
        # 寻找最近的 HYG 菌落
        min_dist = np.min(dists[i])
        
        if min_dist < threshold_dist:
            # 距离足够近 -> 判定为生长 -> Deletion
            color = 'lime' # 亮绿色
            label = "DEL"
            del_count += 1
        else:
            # 附近没有 HYG 菌落 -> 判定为未生长 -> WT
            color = 'red'
            label = "WT"
            wt_count += 1
            
        # 画圈标记
        circ = plt.Circle((cX, cY), radius=25, color=color, fill=False, linewidth=2)
        ax[1].text(cX, cY-30, label, color=color, fontsize=8, ha='center')
        ax[1].add_patch(circ)
        
    ax[1].set_title(f"Result: Green=Deletion (R), Red=WT (S)\nCount: Del={del_count}, WT={wt_count}")
    
    plt.tight_layout()
    plt.savefig("replica_plating_analysis_result.png", dpi=300)
    plt.show()
    plt.close()
    
    return del_count, wt_count

# 运行脚本
if __name__ == "__main__":
    if os.path.exists(TETRAD_PATH) and os.path.exists(HYG_PATH):
        analyze_replica_plating(TETRAD_PATH, HYG_PATH)
    else:
        print("请检查文件路径是否正确。")