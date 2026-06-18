#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术论文图片取证独立工具 / Academic Image Forensics Standalone Tool
直接分析图片文件夹，不进行子图拆解。 / Directly analyze image folder without subfigure extraction.
功能：全局重复检测、局部重复检测、复制-移动检测、ELA+噪声分析、ELA热图生成 / Functions: global duplicate detection, local duplicate detection, copy-move detection, ELA+noise analysis, ELA heatmap generation
"""

import os
import sys
import json
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob

from PIL import Image, ImageDraw, ImageFont
import imagehash
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# ---------- 全局常量（可调参数） / Global Constants (Tunable Parameters) ----------
DEFAULT_HASH_SIZE = 8  # 默认pHash尺寸 / Default pHash size
DEFAULT_ELA_DIFF_THRESHOLD = 30  # ELA差异阈值 / ELA difference threshold
DEFAULT_NOISE_WIN_SIZE = 15  # 噪声分析窗口大小 / Noise analysis window size
DEFAULT_RANSAC_THRESH = 5.0  # RANSAC重投影误差阈值 / RANSAC reprojection error threshold
DEFAULT_GMS_RADIUS = 30  # GMS支持度邻域半径 / GMS support neighborhood radius
DEFAULT_GMS_ANGLE_THRESH = 15  # GMS角度阈值（度） / GMS angle threshold (degrees)
DEFAULT_GMS_SCALE_THRESH = 0.5  # GMS尺度比容忍度 / GMS scale ratio tolerance
DEFAULT_MIN_SUPPORT = 3  # 最小支持度（GMS） / Minimum support (GMS)
DEFAULT_CM_MIN_MATCH = 15  # 复制-移动检测最少匹配数 / Minimum matches for copy-move
DEFAULT_CM_EPS = 2  # 复制-移动DBSCAN聚类半径（像素） / DBSCAN eps for copy-move (pixels)
DEFAULT_CM_MIN_SAMPLES = 5  # 复制-移动DBSCAN最小样本数 / DBSCAN min_samples for copy-move

LAPLACIAN_VAR_THRESH = 50  # 拉普拉斯方差阈值，低于此值视为平坦区域 / Laplacian variance threshold for flat regions

# ---------- 辅助函数 / Helper Functions ----------
def numpy_to_python(obj):
    """
    将numpy类型转换为Python原生类型，便于JSON序列化 / Convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def compute_phash(img_path, hash_size=DEFAULT_HASH_SIZE):
    """
    计算图像感知哈希 / Compute perceptual hash of an image
    """
    try:
        return imagehash.phash(Image.open(img_path), hash_size=hash_size)
    except Exception:
        return None

def safe_imread(img_path, flags=cv2.IMREAD_COLOR):
    """
    安全读取图像，失败返回None并打印警告 / Safely read image, return None on failure with warning
    """
    img = cv2.imread(img_path, flags)
    if img is None:
        print(f"警告: 无法读取图像 {img_path} / Warning: Cannot read image {img_path}", file=sys.stderr)
    return img

def generate_ela_heatmap(img_path, output_path, quality=98):
    """
    生成彩色ELA热图（JET色图），异常区域呈红/黄色 / Generate color ELA heatmap (JET colormap), anomalies in red/yellow
    """
    try:
        img = safe_imread(img_path)
        if img is None:
            return False
        # 用JPEG质量压缩再解码，计算差异 / Compress with JPEG quality then decode, compute difference
        _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        ela_img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(img, ela_img)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # 归一化到0-255并应用JET色图 / Normalize to 0-255 and apply JET colormap
        diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
        cv2.imwrite(output_path, heatmap)
        return True
    except Exception as e:
        print(f"生成ELA热图失败 {img_path}: {e} / ELA heatmap generation failed {img_path}: {e}", file=sys.stderr)
        return False

# ---------- 全局重复检测 / Global Duplicate Detection ----------
def find_global_duplicates_images(img_paths, threshold=2):
    """
    在所有图片中寻找pHash距离小于阈值的重复图片对 / Find duplicate image pairs with pHash distance below threshold
    """
    hash_dict = {}
    for p in img_paths:
        h = compute_phash(p)
        if h is not None:
            hash_dict[p] = h
    duplicates = []
    keys = list(hash_dict.keys())
    # 两两比较 / Pairwise comparison
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            dist = hash_dict[keys[i]] - hash_dict[keys[j]]
            if dist < threshold:
                duplicates.append((os.path.basename(keys[i]), os.path.basename(keys[j]), dist))
    return duplicates

# ---------- 局部重复检测 / Local Duplicate Detection ----------
def find_local_duplicates_strict(img_path, block_size=64, hash_size=DEFAULT_HASH_SIZE,
                                 eps=0.2, min_samples=3, dynamic_area_max=0.15,
                                 edge_ratio_thresh=0.7, std_thresh=15):
    """
    在单张图像内检测局部重复区域（基于滑动窗口pHash + DBSCAN聚类） / Detect local duplicate regions within a single image (sliding window pHash + DBSCAN)
    参数 / Parameters:
        block_size: 滑动窗口大小 / sliding window size
        eps: DBSCAN聚类半径（汉明距离归一化） / DBSCAN eps (normalized Hamming distance)
        min_samples: DBSCAN最小样本数 / DBSCAN min_samples
        dynamic_area_max: 动态面积比上限 / dynamic max area ratio
        edge_ratio_thresh: 边缘区域比例阈值 / edge region ratio threshold
        std_thresh: 块内标准差阈值，低于此值跳过（平坦块） / block std threshold, skip flat blocks
    """
    try:
        img = safe_imread(img_path)
        if img is None:
            return []
        h, w = img.shape[:2]
        coords = []
        hashes = []
        # 滑动窗口遍历 / Sliding window traversal
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = img[y:y+block_size, x:x+block_size]
                gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
                # 跳过低纹理块（平坦区域） / Skip low-texture blocks (flat areas)
                if np.std(gray_block) < std_thresh:
                    continue
                block_pil = Image.fromarray(cv2.cvtColor(block, cv2.COLOR_BGR2RGB))
                ph = imagehash.phash(block_pil, hash_size=hash_size)
                coords.append((x, y))
                hashes.append(ph)
        if len(hashes) < 2:
            return []
        # 将哈希转为特征向量进行DBSCAN聚类 / Convert hashes to feature vectors for DBSCAN
        hash_vectors = np.array([ph.hash.flatten().astype(np.float64) for ph in hashes])
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='hamming').fit(hash_vectors)
        labels = clustering.labels_
        # 动态面积比：根据图像大小自适应 / Dynamic area ratio adaptive to image size
        dynamic_max_area_ratio = max(0.03, min(dynamic_area_max, 20000.0 / (w * h)))
        suspicious_groups = []
        for label in set(labels):
            if label == -1:
                continue
            idxs = [i for i, l in enumerate(labels) if l == label]
            if len(idxs) < min_samples:
                continue
            group_coords = [coords[i] for i in idxs]
            xs, ys = zip(*group_coords)
            # 空间紧凑性过滤：避免散布在整个图像 / Spatial compactness filter
            if np.std(xs) > block_size * 3 or np.std(ys) > block_size * 3:
                continue
            # 面积比限制 / Area ratio limit
            area_ratio = (max(xs) - min(xs)) * (max(ys) - min(ys)) / (w * h)
            if area_ratio > dynamic_max_area_ratio:
                continue
            # 边缘过滤：避免边缘重复（如边框） / Edge filter: avoid border repeats
            edge_margin = min(block_size, 0.05 * min(w, h))
            edge_count = sum(1 for x, y in group_coords if
                             x < edge_margin or x > w - edge_margin or
                             y < edge_margin or y > h - edge_margin)
            if edge_count / len(group_coords) > edge_ratio_thresh:
                continue
            suspicious_groups.append(group_coords)
        return suspicious_groups
    except Exception as e:
        print(f"局部检测失败 {img_path}: {e}\n{traceback.format_exc()} / Local detection failed {img_path}: {e}", file=sys.stderr)
        return []

# ---------- 复制-移动检测 / Copy-Move Detection ----------
def detect_copy_move_gms(img_path,
                         min_match=DEFAULT_CM_MIN_MATCH,
                         eps=DEFAULT_CM_EPS,
                         min_samples=DEFAULT_CM_MIN_SAMPLES,
                         gms_radius=DEFAULT_GMS_RADIUS,
                         gms_angle_thresh=DEFAULT_GMS_ANGLE_THRESH,
                         gms_scale_thresh=DEFAULT_GMS_SCALE_THRESH,
                         min_support=DEFAULT_MIN_SUPPORT,
                         ransac_thresh=DEFAULT_RANSAC_THRESH,
                         min_displacement=10,
                         max_area_ratio=0.3):
    """
    基于SIFT特征 + GMS（运动支持度） + RANSAC验证的复制-移动检测 / Copy-move detection using SIFT + GMS (Grid-based Motion Statistics) + RANSAC verification
    返回可疑位移向量列表 / Returns list of suspicious displacement vectors (dx, dy)
    """
    try:
        img = safe_imread(img_path)
        if img is None:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 提取SIFT特征 / Extract SIFT features
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        if des is None or len(kp) < 10:
            return []
        # 对自身进行匹配（寻找自相似性） / Self-matching to find self-similarities
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des, des)
        matches = [m for m in matches if m.queryIdx != m.trainIdx]  # 排除自身匹配 / Exclude self-match
        if len(matches) < min_match:
            return []
        pts_query = np.array([kp[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts_train = np.array([kp[m.trainIdx].pt for m in matches], dtype=np.float32)
        displacements = pts_train - pts_query
        # 过滤位移过小的匹配 / Filter matches with too small displacement
        dists = np.linalg.norm(displacements, axis=1)
        valid = dists >= min_displacement
        matches = [m for m, v in zip(matches, valid) if v]
        pts_query = pts_query[valid]
        pts_train = pts_train[valid]
        displacements = displacements[valid]
        if len(matches) < min_match:
            return []

        # ---- GMS 支持度计算 / GMS Support Computation ----
        support = np.zeros(len(matches), dtype=int)
        for i in range(len(matches)):
            qi = pts_query[i]
            di = displacements[i]
            # 寻找空间邻域 / Find spatial neighbors
            dists_to_qi = np.linalg.norm(pts_query - qi, axis=1)
            neighbors = np.where(dists_to_qi < gms_radius)[0]
            for j in neighbors:
                if i == j: continue
                dj = displacements[j]
                # 角度一致性 / Angle consistency
                cos_angle = np.dot(di, dj) / (np.linalg.norm(di) * np.linalg.norm(dj) + 1e-8)
                angle_diff = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                if angle_diff < gms_angle_thresh:
                    # 尺度一致性 / Scale consistency
                    scale_ratio = np.linalg.norm(dj) / (np.linalg.norm(di) + 1e-8)
                    if 1 - gms_scale_thresh < scale_ratio < 1 + gms_scale_thresh:
                        support[i] += 1
        good_idx = support >= min_support
        if np.sum(good_idx) < min_match:
            return []
        displacements_good = displacements[good_idx]
        X = displacements_good
        # 对位移向量聚类，找主要复制-移动方向 / Cluster displacement vectors to find main copy-move directions
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = clustering.labels_
        suspicious_vectors = []
        good_matches_indices = np.where(good_idx)[0]
        img_area = gray.shape[0] * gray.shape[1]
        for label in set(labels):
            if label == -1:
                continue
            idxs_in_good = [i for i, l in enumerate(labels) if l == label]
            if len(idxs_in_good) < min_samples:
                continue
            orig_indices = good_matches_indices[idxs_in_good]
            pts_q = pts_query[orig_indices]
            pts_t = pts_train[orig_indices]
            # 计算凸包面积，排除面积过大的区域（可能是整体场景） / Compute convex hull area, exclude too large regions
            hull_q = cv2.convexHull(pts_q.astype(np.float32))
            hull_t = cv2.convexHull(pts_t.astype(np.float32))
            area_q = cv2.contourArea(hull_q)
            area_t = cv2.contourArea(hull_t)
            if area_q > max_area_ratio * img_area or area_t > max_area_ratio * img_area:
                continue
            # RANSAC单应性验证，确保几何一致性 / RANSAC homography verification for geometric consistency
            if len(pts_q) >= 4:
                H, mask = cv2.findHomography(pts_q, pts_t, cv2.RANSAC, ransac_thresh)
                if mask is not None:
                    inlier_ratio = np.sum(mask) / len(mask)
                    if inlier_ratio < 0.5:  # 内点比例太低则视为不通过 / Too low inlier ratio -> reject
                        continue
            # 取该类位移向量的平均值作为代表 / Take average displacement vector as representative
            mean_vec = np.mean(X[idxs_in_good], axis=0)
            suspicious_vectors.append(tuple(mean_vec))
        return suspicious_vectors
    except Exception as e:
        print(f"复制-移动检测失败 {img_path}: {e}\n{traceback.format_exc()} / Copy-move detection failed {img_path}: {e}", file=sys.stderr)
        return []

# ---------- ELA + 噪声分析 / ELA + Noise Analysis ----------
def ela_and_noise_analysis(img_path, quality_list=[75, 90, 98], diff_threshold=30,
                           noise_std_threshold=50, ela_bg_correction=True,
                           noise_win_size=DEFAULT_NOISE_WIN_SIZE):
    """
    对图像进行ELA（Error Level Analysis）和噪声方差分析 / Perform ELA and noise variance analysis
    返回字典：{ "ela": {quality: abnormal_ratio%}, "noise_inconsistency": std_value }
    """
    results = {"ela": {}, "noise_inconsistency": 0.0}
    try:
        img = safe_imread(img_path)
        if img is None:
            return results
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 若图像过于平坦（如柱状图），直接返回0，避免误报 / If image is too flat (e.g., chart), return zero to avoid false positives
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < LAPLACIAN_VAR_THRESH:
            return {"ela": {str(q): 0.0 for q in quality_list}, "noise_inconsistency": 0.0}

        # ELA：对每个质量级别进行压缩再解码比较 / ELA: for each quality level, compress and decode then compare
        for q in quality_list:
            _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            ela_img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            diff = cv2.absdiff(img, ela_img)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff_equalized = cv2.equalizeHist(diff_gray)  # 直方图均衡增强差异 / Histogram equalization to enhance difference
            if ela_bg_correction:
                # 基于背景噪声的动态阈值 / Dynamic threshold based on background noise
                global_median = np.median(diff_equalized)
                global_std = np.std(diff_equalized)
                dynamic_thresh = global_median + 3 * global_std
                _, high_diff = cv2.threshold(diff_equalized, dynamic_thresh, 255, cv2.THRESH_BINARY)
            else:
                _, high_diff = cv2.threshold(diff_equalized, diff_threshold, 255, cv2.THRESH_BINARY)
            abnormal_ratio = np.sum(high_diff > 0) / diff_equalized.size * 100
            results["ela"][q] = round(abnormal_ratio, 2) if abnormal_ratio > 1.0 else 0.0

        # 噪声方差分析：基于拉普拉斯算子局部方差 / Noise variance analysis: local variance of Laplacian
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        win_size = noise_win_size
        # 用盒式滤波快速计算局部均值与平方均值 / Use box filter for fast local mean and mean of squares
        mean_lap = cv2.boxFilter(lap, -1, (win_size, win_size), normalize=True)
        mean_lap2 = cv2.boxFilter(lap**2, -1, (win_size, win_size), normalize=True)
        var_map = mean_lap2 - mean_lap**2
        var_std = np.std(var_map)
        global_noise_level = np.median(var_map)
        adaptive_thresh = max(noise_std_threshold, global_noise_level * 3)
        results["noise_inconsistency"] = round(var_std, 2) if var_std > adaptive_thresh else 0.0
        return results
    except Exception as e:
        print(f"ELA/噪声分析失败 {img_path}: {e}\n{traceback.format_exc()} / ELA/noise analysis failed {img_path}: {e}", file=sys.stderr)
        return results

# ---------- 可视化标注 / Visualization Annotations ----------
def create_comparison(orig_path, annotated_path, output_path, labels=('原图','标注')):
    """
    生成原图与标注图的并排对比图 / Create side-by-side comparison of original and annotated image
    """
    try:
        orig = Image.open(orig_path)
        ann = Image.open(annotated_path)
        w = max(orig.width, ann.width)
        h = max(orig.height, ann.height)
        canvas = Image.new('RGB', (w*2 + 30, h + 60), color='white')
        canvas.paste(orig, (10, 40))
        canvas.paste(ann, (w + 20, 40))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        draw.text((10, 10), labels[0], fill='black', font=font)
        draw.text((w + 20, 10), labels[1], fill='black', font=font)
        canvas.save(output_path)
    except Exception as e:
        print(f"生成对比图失败: {e}\n{traceback.format_exc()} / Comparison generation failed: {e}", file=sys.stderr)

def draw_annotations_on_original(orig_path, annotations, output_path):
    """
    在原图上绘制标注（矩形和箭头） / Draw annotations (rectangles and arrows) on original image
    标注格式 / Annotation format: ('rect', (x,y,w,h), color) 或 / or ('arrow', ((sx,sy),(ex,ey)), color)
    """
    img = safe_imread(orig_path)
    if img is None:
        return
    h_img, w_img = img.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    arrow_len = max(10, int(min(w_img, h_img) * 0.02))  # 箭头长度自适应 / Adaptive arrow length
    for ann in annotations:
        if ann[0] == 'rect':
            x, y, w, h = ann[1]
            color = ann[2] if len(ann) > 2 else 'red'
            draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
        elif ann[0] == 'arrow':
            start, end = ann[1]
            color = ann[2] if len(ann) > 2 else 'blue'
            draw.line([start, end], fill=color, width=3)
            # 绘制箭头三角形 / Draw arrowhead triangles
            angle = np.arctan2(end[1]-start[1], end[0]-start[0])
            for sign in [-1, 1]:
                tip_x = end[0] + arrow_len * np.cos(angle + sign * np.pi/6)
                tip_y = end[1] + arrow_len * np.sin(angle + sign * np.pi/6)
                draw.line([end, (tip_x, tip_y)], fill=color, width=3)
    img_pil.save(output_path)

# ---------- 单图处理（用于多进程） / Single Image Processing (for multiprocessing) ----------
def process_single_image(img_path, block_size, ela_qualities, noise_std_threshold,
                         local_eps, local_min_samples, cm_params):
    """
    对单张图片执行所有检测（局部重复、ELA/噪声、复制-移动），返回结果字典 / Run all detections (local, ELA/noise, copy-move) on a single image, return result dict
    """
    local = find_local_duplicates_strict(
        img_path, block_size=block_size,
        eps=local_eps, min_samples=local_min_samples
    )
    en = ela_and_noise_analysis(
        img_path, quality_list=ela_qualities,
        noise_std_threshold=noise_std_threshold
    )
    cm = detect_copy_move_gms(
        img_path,
        min_match=cm_params.get('min_match', DEFAULT_CM_MIN_MATCH),
        eps=cm_params.get('eps', DEFAULT_CM_EPS),
        min_samples=cm_params.get('min_samples', DEFAULT_CM_MIN_SAMPLES),
        gms_radius=cm_params.get('gms_radius', DEFAULT_GMS_RADIUS),
        gms_angle_thresh=cm_params.get('gms_angle_thresh', DEFAULT_GMS_ANGLE_THRESH),
        gms_scale_thresh=cm_params.get('gms_scale_thresh', DEFAULT_GMS_SCALE_THRESH),
        min_support=cm_params.get('min_support', DEFAULT_MIN_SUPPORT),
        ransac_thresh=cm_params.get('ransac_thresh', DEFAULT_RANSAC_THRESH)
    )
    return {'path': img_path, 'local': local, 'ela_noise': en, 'copy_move': cm}

# ---------- 主流程 / Main Workflow ----------
def scan_images(image_dir, output_dir, threshold=2, block_size=64,
                ela_qualities=[75,90,98], noise_std_threshold=50,
                workers=1, local_eps=0.2, local_min_samples=3,
                cm_params=None, ela_bg_correction=True):
    """
    扫描图片文件夹，执行所有分析并生成报告和可视化结果 / Scan image folder, perform all analyses and generate report and visualizations
    """
    if cm_params is None:
        cm_params = {}
    ann_dir = os.path.join(output_dir, "annotated")
    comp_dir = os.path.join(output_dir, "comparisons")
    ela_heatmap_dir = os.path.join(output_dir, "ela_heatmaps")
    for d in [ann_dir, comp_dir, ela_heatmap_dir]:
        os.makedirs(d, exist_ok=True)

    # 收集所有图片文件 / Collect all image files
    image_files = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    if not image_files:
        print("错误：指定目录中没有找到图片文件。 / Error: No image files found in the specified directory.")
        return

    print(f"找到 {len(image_files)} 个图片文件。 / Found {len(image_files)} image files.")

    # 全局重复检测 / Global duplicate detection
    print("执行全局重复检测... / Performing global duplicate detection...")
    global_dup = find_global_duplicates_images(image_files, threshold=threshold)

    local_results = {}
    ela_noise_results = {}
    copy_move_results = {}
    orig_annotations = defaultdict(list)  # 原图路径 -> 标注列表 / original path -> annotation list

    # 并行或串行处理 / Parallel or serial processing
    if workers > 1 and len(image_files) > 1:
        print(f"使用 {workers} 个工作进程并行分析... / Using {workers} worker processes for parallel analysis...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_single_image, p, block_size, ela_qualities,
                                noise_std_threshold, local_eps, local_min_samples, cm_params): p
                for p in image_files
            }
            completed = 0
            for future in as_completed(futures):
                img_path = futures[future]
                completed += 1
                print(f"  进度: {completed}/{len(image_files)} / Progress: {completed}/{len(image_files)}", end='\r')
                try:
                    res = future.result()
                except Exception as e:
                    print(f"\n处理 {img_path} 出错: {e} / Error processing {img_path}: {e}")
                    continue
                base = os.path.basename(res['path'])
                if res['local']:
                    local_results[base] = res['local']
                if res['ela_noise']['ela'] or res['ela_noise']['noise_inconsistency'] > 0:
                    ela_noise_results[base] = res['ela_noise']
                    # 生成ELA热图 / Generate ELA heatmap
                    ela_base = base.replace('.png', '_ela.png')
                    ela_path = os.path.join(ela_heatmap_dir, ela_base)
                    generate_ela_heatmap(res['path'], ela_path)
                if res['copy_move']:
                    copy_move_results[base] = res['copy_move']
                # 收集标注 / Collect annotations
                if res['local']:
                    for group in res['local']:
                        xs, ys = zip(*group)
                        rect = (min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
                        orig_annotations[res['path']].append(('rect', rect, 'red'))
                if res['copy_move']:
                    img = safe_imread(res['path'])
                    if img is not None:
                        h, w = img.shape[:2]
                        center = (w/2, h/2)
                        for dx, dy in res['copy_move']:
                            end = (center[0] + dx, center[1] + dy)
                            orig_annotations[res['path']].append(
                                ('arrow', (center, end), 'blue'))
        print()
    else:
        for idx, img_path in enumerate(image_files, 1):
            base = os.path.basename(img_path)
            print(f"  分析 {idx}/{len(image_files)}: {base} / Analyzing {idx}/{len(image_files)}: {base}")
            res = process_single_image(img_path, block_size, ela_qualities,
                                       noise_std_threshold, local_eps, local_min_samples, cm_params)
            if res['local']:
                local_results[base] = res['local']
            if res['ela_noise']['ela'] or res['ela_noise']['noise_inconsistency'] > 0:
                ela_noise_results[base] = res['ela_noise']
                ela_base = base.replace('.png', '_ela.png')
                ela_path = os.path.join(ela_heatmap_dir, ela_base)
                generate_ela_heatmap(res['path'], ela_path)
            if res['copy_move']:
                copy_move_results[base] = res['copy_move']
            if res['local']:
                for group in res['local']:
                    xs, ys = zip(*group)
                    rect = (min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
                    orig_annotations[img_path].append(('rect', rect, 'red'))
            if res['copy_move']:
                img = safe_imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    center = (w/2, h/2)
                    for dx, dy in res['copy_move']:
                        end = (center[0] + dx, center[1] + dy)
                        orig_annotations[img_path].append(
                            ('arrow', (center, end), 'blue'))

    # 生成标注图与对比图 / Generate annotated images and comparisons
    print("生成标注与对比图像... / Generating annotations and comparison images...")
    for img_path, ann_list in orig_annotations.items():
        out_name = "annotated_" + os.path.basename(img_path)
        out_path = os.path.join(ann_dir, out_name)
        draw_annotations_on_original(img_path, ann_list, out_path)
        comp_path = os.path.join(comp_dir, "comparison_" + os.path.basename(img_path))
        create_comparison(img_path, out_path, comp_path)

    # 生成JSON报告 / Generate JSON report
    report = {
        "total_images": len(image_files),
        "global_duplicates": global_dup,
        "local_duplicates": local_results,
        "ela_noise": ela_noise_results,
        "copy_move_vectors": copy_move_results
    }
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=numpy_to_python)
    print(f"扫描完成！报告已保存至 {report_path} / Scan complete! Report saved to {report_path}")
    print(f"标注图像: {ann_dir}\n对比图像: {comp_dir}\nELA热图: {ela_heatmap_dir} / Annotations: {ann_dir}\nComparisons: {comp_dir}\nELA heatmaps: {ela_heatmap_dir}")

# ---------- 命令行入口 / Command Line Entry ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="学术论文图片取证独立工具 - 直接分析图片文件夹 / Academic Image Forensics Standalone Tool - Directly analyze image folder")
    parser.add_argument("--image_dir", required=True, help="包含图片文件的文件夹路径 / Path to folder containing images")
    parser.add_argument("--output", required=True, help="输出目录 / Output directory")
    parser.add_argument("--threshold", type=int, default=2, help="全局pHash距离阈值 / Global pHash distance threshold")
    parser.add_argument("--block", type=int, default=64, help="局部检测块大小 / Local detection block size")
    parser.add_argument("--ela_qualities", nargs=3, type=int, default=[75,90,98], help="ELA质量级别 / ELA quality levels")
    parser.add_argument("--noise_threshold", type=float, default=50, help="噪声方差不一致性基础阈值 / Noise variance inconsistency base threshold")
    parser.add_argument("--workers", type=int, default=1, help="并行处理图片的进程数 / Number of parallel worker processes")
    parser.add_argument("--local_eps", type=float, default=0.2, help="局部DBSCAN的eps / Local DBSCAN eps")
    parser.add_argument("--local_min_samples", type=int, default=3, help="局部DBSCAN的最小样本数 / Local DBSCAN min_samples")
    parser.add_argument("--cm_min_match", type=int, default=DEFAULT_CM_MIN_MATCH)
    parser.add_argument("--cm_eps", type=float, default=DEFAULT_CM_EPS)
    parser.add_argument("--cm_min_samples", type=int, default=DEFAULT_CM_MIN_SAMPLES)
    parser.add_argument("--cm_gms_radius", type=int, default=DEFAULT_GMS_RADIUS)
    parser.add_argument("--cm_ransac_thresh", type=float, default=DEFAULT_RANSAC_THRESH)
    args = parser.parse_args()

    cm_params = {
        'min_match': args.cm_min_match,
        'eps': args.cm_eps,
        'min_samples': args.cm_min_samples,
        'gms_radius': args.cm_gms_radius,
        'ransac_thresh': args.cm_ransac_thresh,
    }

    scan_images(
        args.image_dir, args.output,
        threshold=args.threshold,
        block_size=args.block,
        ela_qualities=args.ela_qualities,
        noise_std_threshold=args.noise_threshold,
        workers=args.workers,
        local_eps=args.local_eps,
        local_min_samples=args.local_min_samples,
        cm_params=cm_params,
        ela_bg_correction=True
    )
