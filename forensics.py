#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术论文图像取证工具 / Academic Paper Image Forensics Tool
- 子图拆解：基于OTSU+轮廓检测的视觉分割 / Subfigure extraction: visual segmentation based on OTSU + contour detection
- 局部重复检测：动态面积比、紧凑度、边缘过滤 / Local duplicate detection: dynamic area ratio, compactness, edge filtering
- 复制‑移动检测：SIFT + 运动支持度 GMS + RANSAC 验证 / Copy-move detection: SIFT + GMS (Grid-based Motion Statistics) + RANSAC verification
- ELA 分析：背景噪声基线校正，内存操作 / ELA analysis: background noise baseline correction, in-memory processing
- 噪声分析：积分图加速 + 动态阈值 / Noise analysis: integral image acceleration + dynamic threshold
- ELA热图：生成彩色JET热图，直观显示疑似篡改区域 / ELA heatmap: generate color JET heatmap to visually display suspected tampered areas
- 精确标注：原图坐标映射，生成对比图，自适应箭头 / Precise annotation: original image coordinate mapping, comparison generation, adaptive arrows
- 批量加速：多进程处理子图分析（参数开放） / Batch acceleration: multiprocessing for subfigure analysis (tunable parameters)
"""

import os
import sys
import json
import base64
import traceback
from io import BytesIO
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz
import lxml.etree as ET
from PIL import Image, ImageDraw, ImageFont
import imagehash
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# ---------- 全局常量（可调参数） / Global Constants (Tunable Parameters) ----------
DEFAULT_HASH_SIZE = 8
DEFAULT_ELA_DIFF_THRESHOLD = 30
DEFAULT_NOISE_WIN_SIZE = 15
DEFAULT_RANSAC_THRESH = 5.0
DEFAULT_GMS_RADIUS = 30
DEFAULT_GMS_ANGLE_THRESH = 15
DEFAULT_GMS_SCALE_THRESH = 0.5
DEFAULT_MIN_SUPPORT = 3
DEFAULT_CM_MIN_MATCH = 15
DEFAULT_CM_EPS = 2
DEFAULT_CM_MIN_SAMPLES = 5

EDGE_MARGIN = 20               # 距离边缘多少像素以内视为边缘 / Pixels within this distance from border considered as edge
EDGE_AREA_RATIO_MAX = 0.03     # 边缘区域面积占比超过此值才保留（小于则跳过） / Keep only if edge area ratio exceeds this (skip if smaller)

LAPLACIAN_VAR_THRESH = 50      # 拉普拉斯方差低于此值视为平坦图像 / Laplacian variance below this considered as flat image

# ---------- 辅助函数 / Helper Functions ----------
def numpy_to_python(obj):
    """将 numpy 类型转换为 Python 原生类型，用于 JSON 序列化 / Convert numpy types to Python native types for JSON serialization"""
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
    """计算图像感知哈希 / Compute perceptual hash of an image"""
    try:
        return imagehash.phash(Image.open(img_path), hash_size=hash_size)
    except Exception:
        return None

def safe_imread(img_path, flags=cv2.IMREAD_COLOR):
    """安全读取图像，失败返回 None 并打印错误 / Safely read image, return None and print error on failure"""
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
        _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        ela_img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(img, ela_img)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
        cv2.imwrite(output_path, heatmap)
        return True
    except Exception as e:
        print(f"生成ELA热图失败 {img_path}: {e} / ELA heatmap generation failed {img_path}: {e}", file=sys.stderr)
        return False

# ---------- 1. 图像提取 / Image Extraction ----------
def extract_images_from_pdf(pdf_path, output_dir):
    """从PDF提取所有嵌入图像，返回图像数量 / Extract all embedded images from PDF, return count"""
    doc = fitz.open(pdf_path)
    img_count = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.colorspace is not None and pix.colorspace.n != 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_path = os.path.join(output_dir, f"page{page_num+1}_img{img_index+1}.png")
            pix.save(img_path)
            img_count += 1
    doc.close()
    return img_count

def extract_images_from_xml(xml_path, output_dir):
    """从XML（如JATS）提取嵌入的base64图像 / Extract embedded base64 images from XML (e.g., JATS)"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'xlink': 'http://www.w3.org/1999/xlink'}
    img_count = 0
    # 查找 graphic 元素 / Find graphic elements
    for graphic in root.xpath('//graphic', namespaces=ns):
        href = graphic.get('{http://www.w3.org/1999/xlink}href')
        if href and href.startswith('data:image/'):
            header, data = href.split(',', 1)
            img_data = base64.b64decode(data)
            img = Image.open(BytesIO(img_data))
            img_path = os.path.join(output_dir, f"xml_img_{img_count+1}.png")
            img.save(img_path)
            img_count += 1
    # 查找 media 元素 / Find media elements
    for media in root.xpath('//media', namespaces=ns):
        href = media.get('{http://www.w3.org/1999/xlink}href')
        if href and href.startswith('data:image/'):
            header, data = href.split(',', 1)
            img_data = base64.b64decode(data)
            img = Image.open(BytesIO(img_data))
            img_path = os.path.join(output_dir, f"xml_img_{img_count+1}.png")
            img.save(img_path)
            img_count += 1
    return img_count

# ---------- 2. 子图拆解（纯视觉分割） / Subfigure Extraction (Pure Visual Segmentation) ----------
def extract_subfigures(img_path, output_dir, min_area_ratio=0.02, border_margin=10,
                       otsu_block_size=5, morph_iter=2):
    """
    使用OTSU二值化 + 轮廓检测拆解组合大图 / Use OTSU binarization + contour detection to split composite images
    返回列表，每个元素为 (子图路径, (x,y,w,h)) 在原图中的坐标 / Returns list of (subfigure_path, (x,y,w,h)) coordinates in original
    """
    img = safe_imread(img_path)
    if img is None:
        return []
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # OTSU二值化，前景为白色（子图区域） / OTSU threshold, foreground white (subfigure regions)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((otsu_block_size, otsu_block_size), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)  # 闭合操作连接邻近区域 / Close operation to connect nearby regions
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(5000, int(w * h * min_area_ratio))  # 最小面积阈值 / Minimum area threshold
    subfigures = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < min_area:
            continue

        # 如果轮廓紧贴图像边缘（距边缘 < EDGE_MARGIN）且面积占比 < EDGE_AREA_RATIO_MAX，
        # 视为页眉页脚的Logo/页码等装饰元素，直接跳过 / Skip if contour is near edge and area ratio is small (like headers/footers)
        on_edge = (x < EDGE_MARGIN or y < EDGE_MARGIN or
                   x + cw > w - EDGE_MARGIN or y + ch > h - EDGE_MARGIN)
        area_ratio = area / (w * h)
        if on_edge and area_ratio < EDGE_AREA_RATIO_MAX:
            continue

        # 扩展边界，避免切割过紧 / Expand border to avoid cutting too tight
        x = max(0, x - border_margin)
        y = max(0, y - border_margin)
        cw = min(w - x, cw + 2 * border_margin)
        ch = min(h - y, ch + 2 * border_margin)
        if cw <= 0 or ch <= 0:
            continue
        sub_img = img[y:y+ch, x:x+cw]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_name = f"{base_name}_sub{x}_{y}_{cw}_{ch}.png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, sub_img)
        subfigures.append((out_path, (x, y, cw, ch)))
    # 若未拆解出任何子图，则将整图作为单个子图 / If no subfigure extracted, treat the whole image as one subfigure
    if not subfigures:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}_full.png")
        cv2.imwrite(out_path, img)
        subfigures.append((out_path, (0, 0, w, h)))
    return subfigures

# ---------- 3. 全局重复检测 / Global Duplicate Detection ----------
def find_global_duplicates_subfigs(subfig_paths, threshold=2):
    """在所有子图中寻找pHash距离小于阈值的重复对 / Find duplicate subfigure pairs with pHash distance below threshold"""
    hash_dict = {}
    for p in subfig_paths:
        h = compute_phash(p)
        if h is not None:
            hash_dict[p] = h
    duplicates = []
    keys = list(hash_dict.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            dist = hash_dict[keys[i]] - hash_dict[keys[j]]
            if dist < threshold:
                duplicates.append((os.path.basename(keys[i]), os.path.basename(keys[j]), dist))
    return duplicates

# ---------- 4. 局部重复检测 / Local Duplicate Detection ----------
def find_local_duplicates_strict(img_path, block_size=64, hash_size=DEFAULT_HASH_SIZE,
                                 eps=0.2, min_samples=3, dynamic_area_max=0.15,
                                 edge_ratio_thresh=0.7, std_thresh=15):
    """
    在单张子图内检测局部重复区域 / Detect local duplicate regions within a single subfigure
    参数同 scan_images.py 中的对应函数 / Parameters same as corresponding function in scan_images.py
    """
    try:
        img = safe_imread(img_path)
        if img is None:
            return []
        h, w = img.shape[:2]
        coords = []
        hashes = []
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = img[y:y+block_size, x:x+block_size]
                gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
                if np.std(gray_block) < std_thresh:
                    continue
                block_pil = Image.fromarray(cv2.cvtColor(block, cv2.COLOR_BGR2RGB))
                ph = imagehash.phash(block_pil, hash_size=hash_size)
                coords.append((x, y))
                hashes.append(ph)
        if len(hashes) < 2:
            return []
        hash_vectors = np.array([ph.hash.flatten().astype(np.float64) for ph in hashes])
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='hamming').fit(hash_vectors)
        labels = clustering.labels_
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
            if np.std(xs) > block_size * 3 or np.std(ys) > block_size * 3:
                continue
            area_ratio = (max(xs) - min(xs)) * (max(ys) - min(ys)) / (w * h)
            if area_ratio > dynamic_max_area_ratio:
                continue
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

# ---------- 5. 复制-移动检测 / Copy-Move Detection ----------
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
    与 scan_images.py 中相同 / Same as in scan_images.py
    """
    try:
        img = safe_imread(img_path)
        if img is None:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        if des is None or len(kp) < 10:
            return []
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des, des)
        matches = [m for m in matches if m.queryIdx != m.trainIdx]
        if len(matches) < min_match:
            return []
        pts_query = np.array([kp[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts_train = np.array([kp[m.trainIdx].pt for m in matches], dtype=np.float32)
        displacements = pts_train - pts_query
        dists = np.linalg.norm(displacements, axis=1)
        valid = dists >= min_displacement
        matches = [m for m, v in zip(matches, valid) if v]
        pts_query = pts_query[valid]
        pts_train = pts_train[valid]
        displacements = displacements[valid]
        if len(matches) < min_match:
            return []
        # GMS 支持度 / GMS support
        support = np.zeros(len(matches), dtype=int)
        for i in range(len(matches)):
            qi = pts_query[i]
            di = displacements[i]
            dists_to_qi = np.linalg.norm(pts_query - qi, axis=1)
            neighbors = np.where(dists_to_qi < gms_radius)[0]
            for j in neighbors:
                if i == j: continue
                dj = displacements[j]
                cos_angle = np.dot(di, dj) / (np.linalg.norm(di) * np.linalg.norm(dj) + 1e-8)
                angle_diff = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                if angle_diff < gms_angle_thresh:
                    scale_ratio = np.linalg.norm(dj) / (np.linalg.norm(di) + 1e-8)
                    if 1 - gms_scale_thresh < scale_ratio < 1 + gms_scale_thresh:
                        support[i] += 1
        good_idx = support >= min_support
        if np.sum(good_idx) < min_match:
            return []
        displacements_good = displacements[good_idx]
        X = displacements_good
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
            hull_q = cv2.convexHull(pts_q.astype(np.float32))
            hull_t = cv2.convexHull(pts_t.astype(np.float32))
            area_q = cv2.contourArea(hull_q)
            area_t = cv2.contourArea(hull_t)
            if area_q > max_area_ratio * img_area or area_t > max_area_ratio * img_area:
                continue
            # RANSAC 单应性验证 / RANSAC homography verification
            if len(pts_q) >= 4:
                H, mask = cv2.findHomography(pts_q, pts_t, cv2.RANSAC, ransac_thresh)
                if mask is not None:
                    inlier_ratio = np.sum(mask) / len(mask)
                    if inlier_ratio < 0.5:
                        continue
            mean_vec = np.mean(X[idxs_in_good], axis=0)
            suspicious_vectors.append(tuple(mean_vec))
        return suspicious_vectors
    except Exception as e:
        print(f"复制-移动检测失败 {img_path}: {e}\n{traceback.format_exc()} / Copy-move detection failed {img_path}: {e}", file=sys.stderr)
        return []

# ---------- 6. ELA + 噪声分析 / ELA + Noise Analysis ----------
def ela_and_noise_analysis(img_path, quality_list=[75, 90, 98], diff_threshold=30,
                           noise_std_threshold=50, ela_bg_correction=True,
                           noise_win_size=DEFAULT_NOISE_WIN_SIZE):
    """
    与 scan_images.py 中相同，但增加了对平坦图像的跳过 / Same as in scan_images.py, but with flat image skipping
    """
    results = {"ela": {}, "noise_inconsistency": 0.0}
    try:
        img = safe_imread(img_path)
        if img is None:
            return results

        # 柱状图/图表检测（低纹理平坦图像） / chart/flat image detection
        # 如果拉普拉斯方差低于阈值，说明是平坦区域（如柱状图、表格），直接返回0避免误报 / If Laplacian variance low, treat as flat (e.g., charts) and return zero
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < LAPLACIAN_VAR_THRESH:
            return {"ela": {str(q): 0.0 for q in quality_list}, "noise_inconsistency": 0.0}

        # ELA 使用内存编解码 / ELA using in-memory codec
        for q in quality_list:
            _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            ela_img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            diff = cv2.absdiff(img, ela_img)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff_equalized = cv2.equalizeHist(diff_gray)
            if ela_bg_correction:
                global_median = np.median(diff_equalized)
                global_std = np.std(diff_equalized)
                dynamic_thresh = global_median + 3 * global_std
                _, high_diff = cv2.threshold(diff_equalized, dynamic_thresh, 255, cv2.THRESH_BINARY)
            else:
                _, high_diff = cv2.threshold(diff_equalized, diff_threshold, 255, cv2.THRESH_BINARY)
            abnormal_ratio = np.sum(high_diff > 0) / diff_equalized.size * 100
            results["ela"][q] = round(abnormal_ratio, 2) if abnormal_ratio > 1.0 else 0.0

        # 噪声方差 / Noise variance
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        win_size = noise_win_size
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

# ---------- 7. 可视化标注 / Visualization Annotations ----------
def create_comparison(orig_path, annotated_path, output_path, labels=('原图','标注')):
    """生成对比图 / Generate comparison image"""
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
    """在原图上绘制标注 / Draw annotations on original image"""
    img = safe_imread(orig_path)
    if img is None:
        return
    h_img, w_img = img.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    arrow_len = max(10, int(min(w_img, h_img) * 0.02))
    for ann in annotations:
        if ann[0] == 'rect':
            x, y, w, h = ann[1]
            color = ann[2] if len(ann) > 2 else 'red'
            draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
        elif ann[0] == 'arrow':
            start, end = ann[1]
            color = ann[2] if len(ann) > 2 else 'blue'
            draw.line([start, end], fill=color, width=3)
            angle = np.arctan2(end[1]-start[1], end[0]-start[0])
            for sign in [-1, 1]:
                tip_x = end[0] + arrow_len * np.cos(angle + sign * np.pi/6)
                tip_y = end[1] + arrow_len * np.sin(angle + sign * np.pi/6)
                draw.line([end, (tip_x, tip_y)], fill=color, width=3)
    img_pil.save(output_path)

# ---------- 8. 单子图处理（用于多进程） / Single Subfigure Processing (for multiprocessing) ----------
def process_single_subfig(sub_path, block_size, ela_qualities, noise_std_threshold,
                          local_eps, local_min_samples, cm_params):
    """对单个子图执行所有检测 / Run all detections on a single subfigure"""
    local = find_local_duplicates_strict(
        sub_path, block_size=block_size,
        eps=local_eps, min_samples=local_min_samples
    )
    en = ela_and_noise_analysis(
        sub_path, quality_list=ela_qualities,
        noise_std_threshold=noise_std_threshold
    )
    cm = detect_copy_move_gms(
        sub_path,
        min_match=cm_params.get('min_match', DEFAULT_CM_MIN_MATCH),
        eps=cm_params.get('eps', DEFAULT_CM_EPS),
        min_samples=cm_params.get('min_samples', DEFAULT_CM_MIN_SAMPLES),
        gms_radius=cm_params.get('gms_radius', DEFAULT_GMS_RADIUS),
        gms_angle_thresh=cm_params.get('gms_angle_thresh', DEFAULT_GMS_ANGLE_THRESH),
        gms_scale_thresh=cm_params.get('gms_scale_thresh', DEFAULT_GMS_SCALE_THRESH),
        min_support=cm_params.get('min_support', DEFAULT_MIN_SUPPORT),
        ransac_thresh=cm_params.get('ransac_thresh', DEFAULT_RANSAC_THRESH)
    )
    return {
        'path': sub_path,
        'local': local,
        'ela_noise': en,
        'copy_move': cm
    }

# ---------- 9. 主流程 / Main Workflow ----------
def scan_paper(input_path, output_dir, threshold=2, block_size=64,
               ela_qualities=[75,90,98], min_area_ratio=0.02, border_margin=10,
               noise_std_threshold=50, workers=1,
               local_eps=0.2, local_min_samples=3,
               cm_params=None, ela_bg_correction=True):
    """
    扫描论文（PDF或XML），提取图像、拆分子图、执行所有检测并生成报告 / Scan paper (PDF or XML), extract images, extract subfigures, run all detections and generate report
    """
    if cm_params is None:
        cm_params = {}
    img_dir = os.path.join(output_dir, "images")
    sub_dir = os.path.join(output_dir, "subfigures")
    ann_dir = os.path.join(output_dir, "annotated")
    comp_dir = os.path.join(output_dir, "comparisons")
    ela_heatmap_dir = os.path.join(output_dir, "ela_heatmaps")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(ela_heatmap_dir, exist_ok=True)

    print("正在提取原始图像... / Extracting original images...")
    if input_path.lower().endswith('.pdf'):
        count = extract_images_from_pdf(input_path, img_dir)
    elif input_path.lower().endswith('.xml'):
        count = extract_images_from_xml(input_path, img_dir)
    else:
        raise ValueError("仅支持 .pdf 或 .xml / Only .pdf or .xml supported")
    print(f"提取到 {count} 张大图。 / Extracted {count} large images.")

    if count == 0:
        print("未提取到任何图像，退出。 / No images extracted, exiting.")
        return

    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    all_subfig_paths = []
    subfig_bbox_map = {}  # 子图路径 -> (原图路径, (x,y,w,h)) / subfigure path -> (original path, bbox)

    print("正在拆分子图... / Extracting subfigures...")
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(img_dir, img_file)
        subfigs = extract_subfigures(img_path, sub_dir,
                                     min_area_ratio=min_area_ratio,
                                     border_margin=border_margin)
        for sub_path, bbox in subfigs:
            all_subfig_paths.append(sub_path)
            subfig_bbox_map[sub_path] = (img_path, bbox)
        print(f"  图像 {idx}/{len(image_files)} 拆解出 {len(subfigs)} 个子图 / Image {idx}/{len(image_files)} extracted {len(subfigs)} subfigures")
    print(f"拆解得到 {len(all_subfig_paths)} 个子图。 / Extracted {len(all_subfig_paths)} subfigures.")

    # 全局重复 / Global duplicates
    print(f"执行子图全局重复检测（阈值={threshold}） / Performing global duplicate detection on subfigures (threshold={threshold})...")
    global_dup = find_global_duplicates_subfigs(all_subfig_paths, threshold=threshold)

    # 子图级分析 / Subfigure-level analysis
    local_results = {}
    ela_noise_results = {}
    copy_move_results = {}
    orig_annotations = defaultdict(list)

    def is_ela_abnormal(ela_noise_dict):
        """判断ELA/噪声是否异常 / Check if ELA/noise is abnormal"""
        if ela_noise_dict.get('noise_inconsistency', 0) > 0:
            return True
        for q, val in ela_noise_dict.get('ela', {}).items():
            if val > 1.0:   # 异常面积比 > 1% / abnormal area ratio > 1%
                return True
        return False

    if workers > 1:
        print(f"使用 {workers} 个工作进程并行分析子图... / Using {workers} worker processes for parallel subfigure analysis...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_single_subfig, p, block_size, ela_qualities,
                                noise_std_threshold, local_eps, local_min_samples, cm_params): p
                for p in all_subfig_paths
            }
            completed = 0
            for future in as_completed(futures):
                sub_path = futures[future]
                completed += 1
                print(f"  进度: {completed}/{len(all_subfig_paths)} / Progress: {completed}/{len(all_subfig_paths)}", end='\r')
                try:
                    res = future.result()
                except Exception as e:
                    print(f"\n处理子图 {sub_path} 出错: {e}\n{traceback.format_exc()} / Error processing subfigure {sub_path}: {e}", file=sys.stderr)
                    continue
                base = os.path.basename(res['path'])
                if res['local']:
                    local_results[base] = res['local']
                if res['ela_noise']['ela'] or res['ela_noise']['noise_inconsistency'] > 0:
                    ela_noise_results[base] = res['ela_noise']
                    # 生成ELA热图 / Generate ELA heatmap
                    if is_ela_abnormal(res['ela_noise']):
                        heatmap_name = os.path.splitext(base)[0] + "_ela.png"
                        heatmap_path = os.path.join(ela_heatmap_dir, heatmap_name)
                        generate_ela_heatmap(res['path'], heatmap_path)
                if res['copy_move']:
                    copy_move_results[base] = res['copy_move']
                # 将子图坐标映射到原图，收集标注 / Map subfigure coordinates to original image and collect annotations
                orig_path, (ox, oy, ow, oh) = subfig_bbox_map[res['path']]
                if res['local']:
                    for group in res['local']:
                        mapped = [(x+ox, y+oy) for (x, y) in group]
                        xs, ys = zip(*mapped)
                        rect = (min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
                        orig_annotations[orig_path].append(('rect', rect, 'red'))
                if res['copy_move']:
                    center_sub = (ow/2, oh/2)
                    center_orig = (ox + center_sub[0], oy + center_sub[1])
                    for dx, dy in res['copy_move']:
                        end_x = center_orig[0] + dx
                        end_y = center_orig[1] + dy
                        orig_annotations[orig_path].append(
                            ('arrow', ((center_orig[0], center_orig[1]), (end_x, end_y)), 'blue'))
            print()
    else:
        for idx, sub_path in enumerate(all_subfig_paths, 1):
            base = os.path.basename(sub_path)
            print(f"  分析 {idx}/{len(all_subfig_paths)}: {base} / Analyzing {idx}/{len(all_subfig_paths)}: {base}")
            res = process_single_subfig(sub_path, block_size, ela_qualities,
                                        noise_std_threshold, local_eps, local_min_samples, cm_params)
            if res['local']:
                local_results[base] = res['local']
            if res['ela_noise']['ela'] or res['ela_noise']['noise_inconsistency'] > 0:
                ela_noise_results[base] = res['ela_noise']
                if is_ela_abnormal(res['ela_noise']):
                    heatmap_name = os.path.splitext(base)[0] + "_ela.png"
                    heatmap_path = os.path.join(ela_heatmap_dir, heatmap_name)
                    generate_ela_heatmap(res['path'], heatmap_path)
            if res['copy_move']:
                copy_move_results[base] = res['copy_move']
            orig_path, (ox, oy, ow, oh) = subfig_bbox_map[sub_path]
            if res['local']:
                for group in res['local']:
                    mapped = [(x+ox, y+oy) for (x, y) in group]
                    xs, ys = zip(*mapped)
                    rect = (min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
                    orig_annotations[orig_path].append(('rect', rect, 'red'))
            if res['copy_move']:
                center_sub = (ow/2, oh/2)
                center_orig = (ox + center_sub[0], oy + center_sub[1])
                for dx, dy in res['copy_move']:
                    end_x = center_orig[0] + dx
                    end_y = center_orig[1] + dy
                    orig_annotations[orig_path].append(
                        ('arrow', ((center_orig[0], center_orig[1]), (end_x, end_y)), 'blue'))

    # 生成标注图与对比图 / Generate annotated images and comparisons
    print("生成标注与对比图像... / Generating annotations and comparison images...")
    for orig_path, ann_list in orig_annotations.items():
        out_name = "annotated_" + os.path.basename(orig_path)
        out_path = os.path.join(ann_dir, out_name)
        draw_annotations_on_original(orig_path, ann_list, out_path)
        comp_path = os.path.join(comp_dir, "comparison_" + os.path.basename(orig_path))
        create_comparison(orig_path, out_path, comp_path)

    # 报告 / Report
    report = {
        "total_subfigures": len(all_subfig_paths),
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

# ---------- 10. 命令行入口 / Command Line Entry ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="学术论文图像取证工具 / Academic Paper Image Forensics Tool")
    parser.add_argument("--input", required=True, help="PDF或XML文件路径 / Path to PDF or XML file")
    parser.add_argument("--output", required=True, help="输出目录 / Output directory")
    parser.add_argument("--threshold", type=int, default=2, help="全局pHash距离阈值 / Global pHash distance threshold")
    parser.add_argument("--block", type=int, default=64, help="局部检测块大小 / Local detection block size")
    parser.add_argument("--ela_qualities", nargs=3, type=int, default=[75,90,98], help="ELA质量级别 / ELA quality levels")
    parser.add_argument("--min_area_ratio", type=float, default=0.02, help="子图最小面积占原图比例 / Minimum subfigure area ratio to original image")
    parser.add_argument("--border_margin", type=int, default=10, help="子图边框扩展像素 / Subfigure border expansion margin (pixels)")
    parser.add_argument("--noise_threshold", type=float, default=50, help="噪声方差不一致性基础阈值 / Noise variance inconsistency base threshold")
    parser.add_argument("--workers", type=int, default=1, help="并行处理子图的进程数 / Number of parallel worker processes for subfigures")
    parser.add_argument("--local_eps", type=float, default=0.2, help="局部DBSCAN的eps（归一化汉明距离） / Local DBSCAN eps (normalized Hamming distance)")
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
    scan_paper(
        args.input, args.output,
        threshold=args.threshold,
        block_size=args.block,
        ela_qualities=args.ela_qualities,
        min_area_ratio=args.min_area_ratio,
        border_margin=args.border_margin,
        noise_std_threshold=args.noise_threshold,
        workers=args.workers,
        local_eps=args.local_eps,
        local_min_samples=args.local_min_samples,
        cm_params=cm_params,
        ela_bg_correction=True
    )
