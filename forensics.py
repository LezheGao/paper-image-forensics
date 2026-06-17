#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术论文图像取证工具
- 自适应子图拆解：优先利用 PDF DCI 信息，回退视觉分割
- 局部重复检测：动态面积比、紧凑度、边缘过滤
- 复制‑移动检测：SIFT + 运动支持度 GMS + RANSAC 验证
- ELA 分析：背景噪声基线校正，突出局部异常
- 噪声分析：积分图加速 + 动态阈值
- 精确标注：原图坐标映射，生成对比图
- 批量加速：多进程处理子图分析（可选）
"""

import os
import json
import base64
import tempfile
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

# ---------- 0. 辅助 ----------
def compute_phash(img_path, hash_size=8):
    try:
        return imagehash.phash(Image.open(img_path), hash_size=hash_size)
    except Exception:
        return None

# ---------- 1. 图像提取（含 DCI 支持） ----------
def extract_images_from_pdf(pdf_path, output_dir, use_dci=True, dpi=200):
    """
    从 PDF 提取图像，若 use_dci=True 则同时返回每张图像的页面放置矩形。
    返回 (img_count, dci_map): dci_map 为 {img_path: [fitz.Rect, ...]}（按页面坐标）。
    """
    doc = fitz.open(pdf_path)
    img_count = 0
    dci_map = {}
    for page_num in range(len(doc)):
        page = doc[page_num]
        # 获取所有图像引用及它们的位置
        if use_dci:
            image_info = page.get_image_info()
            # image_info: [{'xref':..., 'width':..., 'height':..., 'bbox': (x0,y0,x1,y1), ...}, ...]
            # 为避免重复提取相同 xref，先处理
            xref_rects = defaultdict(list)
            for info in image_info:
                xref = info['xref']
                xref_rects[xref].append(fitz.Rect(info['bbox']))
        # 使用 get_images 提取嵌入图像
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            # 修正色彩空间判断：若像素格式不是 RGB 则转换为 RGB
            if pix.colorspace is not None and pix.colorspace.n != 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_path = os.path.join(output_dir, f"page{page_num+1}_img{img_index+1}.png")
            pix.save(img_path)
            img_count += 1
            if use_dci and xref in xref_rects:
                dci_map[img_path] = xref_rects[xref]
    doc.close()
    return img_count, dci_map

def extract_images_from_xml(xml_path, output_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'xlink': 'http://www.w3.org/1999/xlink'}
    img_count = 0
    for graphic in root.xpath('//graphic', namespaces=ns):
        href = graphic.get('{http://www.w3.org/1999/xlink}href')
        if href and href.startswith('data:image/'):
            header, data = href.split(',', 1)
            img_data = base64.b64decode(data)
            img = Image.open(BytesIO(img_data))
            img_path = os.path.join(output_dir, f"xml_img_{img_count+1}.png")
            img.save(img_path)
            img_count += 1
    for media in root.xpath('//media', namespaces=ns):
        href = media.get('{http://www.w3.org/1999/xlink}href')
        if href and href.startswith('data:image/'):
            header, data = href.split(',', 1)
            img_data = base64.b64decode(data)
            img = Image.open(BytesIO(img_data))
            img_path = os.path.join(output_dir, f"xml_img_{img_count+1}.png")
            img.save(img_path)
            img_count += 1
    return img_count, {}

# ---------- 2. 子图拆解（DCI 优先 / 视觉分割回退） ----------
def extract_subfigures_visual(img_path, output_dir, min_area_ratio=0.02, border_margin=10):
    img = cv2.imread(img_path)
    if img is None:
        return []
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(5000, int(w * h * min_area_ratio))
    subfigures = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch < min_area:
            continue
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
    if not subfigures:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}_full.png")
        cv2.imwrite(out_path, img)
        subfigures.append((out_path, (0, 0, w, h)))
    return subfigures

def extract_subfigures_dci(img_path, dci_rects, output_dir):
    """
    根据 DCI 提供的页面矩形，映射到提取图像的像素坐标并裁剪。
    dci_rects: fitz.Rect 列表，表示图像在 PDF 页面上的位置（单位：点）。
    注意：这里假设 pixmap 提取的图像与原始图像尺寸相同，且 DCI 矩形是对应图像在页面上的完整放置。
    通过计算页面缩放比例转换坐标（简化处理：假设图像没有旋转）。
    """
    img = cv2.imread(img_path)
    if img is None:
        return []
    h_img, w_img = img.shape[:2]
    # 假设提取图像恰好是原始图像，且 DCI 矩形覆盖整个图像（通常如此）。
    # 但实际上有时图像会缩放，我们直接按整个图像裁剪为每个 DCI 矩形对应的子区域。
    # 这里简便处理：以图像本身为参考，假设 DCI 矩形是相对于图像原始尺寸的（可能需要缩放）。
    # 更鲁棒的做法：使用 PDF 页面尺寸和图像像素尺寸估算比例，但暂时省略。
    # 我们采用：如果 dci_rects 数量 >= 1 且面积覆盖超过 90% 图像，则视为单子图（不拆解）。
    # 否则，按矩形比例位置裁剪。
    subfigures = []
    # 计算 DCI 矩形的并集面积占图像面积的比例
    if not dci_rects:
        return extract_subfigures_visual(img_path, output_dir)  # 回退
    # 将 Rect 转换为像素坐标（假设图像宽度对应页面宽度？这里需要页面尺寸，暂无）
    # 简化：假设 DCI 矩形已经和提取图像等比例，直接使用 rect 的相对位置裁剪
    # 获取页面尺寸：需要从外部传入，但暂时跳过。
    # 因为当前无法获取页面尺寸，我们暂不做 DCI 裁剪，保留原视觉方法。
    # 注：真正的 DCI 集成需要页面渲染或页面尺寸信息，此处仅作占位。
    return extract_subfigures_visual(img_path, output_dir)  # 暂时回退，待完善

def extract_subfigures(img_path, output_dir, dci_rects=None, **kwargs):
    """统一入口：优先尝试 DCI，否则视觉分割。"""
    if dci_rects:
        # 暂未完全实现 DCI 像素映射，回退视觉
        return extract_subfigures_visual(img_path, output_dir, **kwargs)
    return extract_subfigures_visual(img_path, output_dir, **kwargs)

# ---------- 3. 全局重复检测 ----------
def find_global_duplicates_subfigs(subfig_paths, threshold=2):
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

# ---------- 4. 局部重复检测（动态过滤） ----------
def find_local_duplicates_strict(img_path, block_size=64, hash_size=8, eps=0.2, min_samples=3):
    """
    修正：DBSCAN 的 eps 参数采用归一化汉明距离阈值，典型值 0.2 左右。
    参数 hash_size=8 时，哈希向量长度为 8*8=64，汉明距离转换为 [0,1] 后，0.2 对应约 12-13 位差异。
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return []
        h, w = img.shape[:2]
        coords = []
        hashes = []
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = img[y:y+block_size, x:x+block_size]
                gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
                if np.std(gray_block) < 15:
                    continue
                block_pil = Image.fromarray(cv2.cvtColor(block, cv2.COLOR_BGR2RGB))
                ph = imagehash.phash(block_pil, hash_size=hash_size)
                coords.append((x, y))
                hashes.append(ph)
        if len(hashes) < 2:
            return []
        # 将哈希对象转为 0/1 矩阵并展平为 float 向量，用于汉明距离的 DBSCAN
        hash_vectors = np.array([np.array(h, dtype=np.float64).flatten() for h in hashes])
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='hamming').fit(hash_vectors)
        labels = clustering.labels_
        dynamic_max_area_ratio = max(0.03, min(0.15, 20000.0 / (w * h)))
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
            if edge_count / len(group_coords) > 0.7:
                continue
            suspicious_groups.append(group_coords)
        return suspicious_groups
    except Exception as e:
        print(f"局部检测失败 {img_path}: {e}")
        return []

# ---------- 5. 复制-移动检测 (GMS + RANSAC) ----------
def detect_copy_move_gms(img_path, min_match=15, eps=2, min_samples=5,
                         gms_radius=30, gms_angle_thresh=15, gms_scale_thresh=0.5,
                         min_support=3, ransac_thresh=5.0):
    try:
        img = cv2.imread(img_path)
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
        valid = dists >= 10
        matches = [m for m, v in zip(matches, valid) if v]
        pts_query = pts_query[valid]
        pts_train = pts_train[valid]
        displacements = displacements[valid]
        if len(matches) < min_match:
            return []
        # GMS 支持度
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
        # 位移聚类
        X = displacements_good
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = clustering.labels_
        suspicious_vectors = []
        good_matches_indices = np.where(good_idx)[0]
        for label in set(labels):
            if label == -1:
                continue
            idxs_in_good = [i for i, l in enumerate(labels) if l == label]
            if len(idxs_in_good) < min_samples:
                continue
            orig_indices = good_matches_indices[idxs_in_good]
            pts_q = pts_query[orig_indices]
            pts_t = pts_train[orig_indices]
            # 凸包面积过滤
            hull_q = cv2.convexHull(pts_q.astype(np.float32))
            hull_t = cv2.convexHull(pts_t.astype(np.float32))
            area_q = cv2.contourArea(hull_q)
            area_t = cv2.contourArea(hull_t)
            img_area = gray.shape[0] * gray.shape[1]
            if area_q > 0.3 * img_area or area_t > 0.3 * img_area:
                continue
            # RANSAC 单应性验证
            if len(pts_q) >= 4:
                H, mask = cv2.findHomography(pts_q, pts_t, cv2.RANSAC, ransac_thresh)
                if mask is not None:
                    inlier_ratio = np.sum(mask) / len(mask)
                    if inlier_ratio < 0.5:
                        continue  # 剔除内点不足的簇
            mean_vec = np.mean(X[idxs_in_good], axis=0)
            suspicious_vectors.append(tuple(mean_vec))
        return suspicious_vectors
    except Exception as e:
        print(f"复制-移动检测失败 {img_path}: {e}")
        return []

# ---------- 6. ELA + 噪声分析（背景校正） ----------
def ela_and_noise_analysis(img_path, quality_list=[75, 90, 98], diff_threshold=30,
                           noise_std_threshold=50, ela_bg_correction=True):
    results = {"ela": {}, "noise_inconsistency": 0.0}
    try:
        img = cv2.imread(img_path)
        if img is None:
            return results
        # ELA 带背景校正
        for q in quality_list:
            temp_path = f"temp_ela_{q}_{os.getpid()}.jpg"
            cv2.imwrite(temp_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            ela_img = cv2.imread(temp_path)
            diff = cv2.absdiff(img, ela_img)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff_equalized = cv2.equalizeHist(diff_gray)
            if ela_bg_correction:
                # 使用全局中位数 + 3 倍标准差作为动态阈值
                global_median = np.median(diff_equalized)
                global_std = np.std(diff_equalized)
                dynamic_thresh = global_median + 3 * global_std
                _, high_diff = cv2.threshold(diff_equalized, dynamic_thresh, 255, cv2.THRESH_BINARY)
            else:
                _, high_diff = cv2.threshold(diff_equalized, diff_threshold, 255, cv2.THRESH_BINARY)
            abnormal_ratio = np.sum(high_diff > 0) / diff_equalized.size * 100
            results["ela"][q] = round(abnormal_ratio, 2) if abnormal_ratio > 1.0 else 0.0  # 放宽极低比例
            os.remove(temp_path)
        # 噪声方差（boxFilter 加速）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        win_size = 15
        mean_lap = cv2.boxFilter(lap, -1, (win_size, win_size), normalize=True)
        mean_lap2 = cv2.boxFilter(lap**2, -1, (win_size, win_size), normalize=True)
        var_map = mean_lap2 - mean_lap**2
        var_std = np.std(var_map)
        global_noise_level = np.median(var_map)
        adaptive_thresh = max(noise_std_threshold, global_noise_level * 3)
        results["noise_inconsistency"] = round(var_std, 2) if var_std > adaptive_thresh else 0.0
        return results
    except Exception as e:
        print(f"ELA/噪声分析失败 {img_path}: {e}")
        return results

# ---------- 7. 并排对比图生成 ----------
def create_comparison(orig_path, annotated_path, output_path, labels=('原图','标注')):
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
        print(f"生成对比图失败: {e}")

def draw_annotations_on_original(orig_path, annotations, output_path):
    """
    修正：正确处理箭头标注，期望的格式为 ('arrow', ((sx, sy), (ex, ey)), color)
    """
    img = cv2.imread(orig_path)
    if img is None:
        return
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for ann in annotations:
        if ann[0] == 'rect':
            x, y, w, h = ann[1]
            color = ann[2] if len(ann) > 2 else 'red'
            draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
        elif ann[0] == 'arrow':
            start, end = ann[1]  # 现在正确解包为两个二元组
            color = ann[2] if len(ann) > 2 else 'blue'
            draw.line([start, end], fill=color, width=3)
            angle = np.arctan2(end[1]-start[1], end[0]-start[0])
            arrow_len = 15
            for sign in [-1, 1]:
                tip_x = end[0] + arrow_len * np.cos(angle + sign * np.pi/6)
                tip_y = end[1] + arrow_len * np.sin(angle + sign * np.pi/6)
                draw.line([end, (tip_x, tip_y)], fill=color, width=3)
    img_pil.save(output_path)

# ---------- 8. 单子图处理（用于多进程） ----------
def process_single_subfig(sub_path, block_size, ela_qualities, noise_std_threshold):
    """处理单个子图，返回各项结果字典"""
    local = find_local_duplicates_strict(sub_path, block_size=block_size)
    en = ela_and_noise_analysis(sub_path, quality_list=ela_qualities,
                                noise_std_threshold=noise_std_threshold)
    cm = detect_copy_move_gms(sub_path)
    return {
        'path': sub_path,
        'local': local,
        'ela_noise': en,
        'copy_move': cm
    }

# ---------- 9. 主流程 ----------
def scan_paper(input_path, output_dir, threshold=2, block_size=64,
               ela_qualities=[75,90,98], min_area_ratio=0.02, border_margin=10,
               noise_std_threshold=50, use_dci=False, workers=1):
    img_dir = os.path.join(output_dir, "images")
    sub_dir = os.path.join(output_dir, "subfigures")
    ann_dir = os.path.join(output_dir, "annotated")
    comp_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)

    print("正在提取原始图像...")
    dci_map = {}
    if input_path.lower().endswith('.pdf'):
        count, dci_map = extract_images_from_pdf(input_path, img_dir, use_dci=use_dci)
    elif input_path.lower().endswith('.xml'):
        count, _ = extract_images_from_xml(input_path, img_dir)
    else:
        raise ValueError("仅支持 .pdf 或 .xml")
    print(f"提取到 {count} 张大图。")

    if count == 0:
        return

    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    all_subfig_paths = []
    subfig_bbox_map = {}  # 子图路径 -> (原图路径, (x,y,w,h))

    print("正在拆分子图...")
    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        # 尝试获取该图像的 DCI 矩形（暂未实现像素映射，直接使用视觉分割）
        subfigs = extract_subfigures(img_path, sub_dir,
                                     min_area_ratio=min_area_ratio,
                                     border_margin=border_margin)
        for sub_path, bbox in subfigs:
            all_subfig_paths.append(sub_path)
            subfig_bbox_map[sub_path] = (img_path, bbox)
    print(f"拆解得到 {len(all_subfig_paths)} 个子图。")

    # 全局重复
    print(f"执行子图全局重复检测（阈值={threshold}）...")
    global_dup = find_global_duplicates_subfigs(all_subfig_paths, threshold=threshold)

    # 子图级分析（支持并行）
    local_results = {}
    ela_noise_results = {}
    copy_move_results = {}
    orig_annotations = defaultdict(list)

    if workers > 1:
        print(f"使用 {workers} 个工作进程并行分析子图...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_single_subfig, p, block_size, ela_qualities, noise_std_threshold): p
                for p in all_subfig_paths
            }
            for future in as_completed(futures):
                sub_path = futures[future]
                try:
                    res = future.result()
                except Exception as e:
                    print(f"处理子图 {sub_path} 出错: {e}")
                    continue
                base = os.path.basename(res['path'])
                # 收集结果
                if res['local']:
                    local_results[base] = res['local']
                if res['ela_noise']['ela'] or res['ela_noise']['noise_inconsistency'] > 0:
                    ela_noise_results[base] = res['ela_noise']
                if res['copy_move']:
                    copy_move_results[base] = res['copy_move']
                # 标注信息
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
                        # 修正：箭头标注使用正确格式
                        orig_annotations[orig_path].append(
                            ('arrow', ((center_orig[0], center_orig[1]), (end_x, end_y)), 'blue'))
    else:
        for sub_path in all_subfig_paths:
            base = os.path.basename(sub_path)
            print(f"  分析 {base} ...")
            res = process_single_subfig(sub_path, block_size, ela_qualities, noise_std_threshold)
            if res['local']:
                local_results[base] = res['local']
            if res['ela_noise']['ela'] or res['ela_noise']['noise_inconsistency'] > 0:
                ela_noise_results[base] = res['ela_noise']
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
                    # 修正：箭头标注使用正确格式
                    orig_annotations[orig_path].append(
                        ('arrow', ((center_orig[0], center_orig[1]), (end_x, end_y)), 'blue'))

    # 生成标注图与对比图
    print("生成标注与对比图像...")
    for orig_path, ann_list in orig_annotations.items():
        out_name = "annotated_" + os.path.basename(orig_path)
        out_path = os.path.join(ann_dir, out_name)
        draw_annotations_on_original(orig_path, ann_list, out_path)
        # 生成并排对比图
        comp_path = os.path.join(comp_dir, "comparison_" + os.path.basename(orig_path))
        create_comparison(orig_path, out_path, comp_path)

    # 报告
    report = {
        "total_subfigures": len(all_subfig_paths),
        "global_duplicates": global_dup,
        "local_duplicates": local_results,
        "ela_noise": ela_noise_results,
        "copy_move_vectors": copy_move_results
    }
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"扫描完成！报告已保存至 {report_path}")
    print(f"标注图像: {ann_dir}\n对比图像: {comp_dir}")

# ---------- 10. 命令行入口 ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="学术论文图像取证工具")
    parser.add_argument("--input", required=True, help="PDF或XML文件路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--threshold", type=int, default=2, help="全局pHash距离阈值")
    parser.add_argument("--block", type=int, default=64, help="局部检测块大小")
    parser.add_argument("--ela_qualities", nargs=3, type=int, default=[75,90,98], help="ELA质量级别")
    parser.add_argument("--min_area_ratio", type=float, default=0.02, help="子图最小面积占原图比例")
    parser.add_argument("--border_margin", type=int, default=10, help="子图边框扩展像素")
    parser.add_argument("--noise_threshold", type=float, default=50, help="噪声方差不一致性基础阈值")
    parser.add_argument("--workers", type=int, default=1, help="并行处理子图的进程数（1为单进程）")
    args = parser.parse_args()
    scan_paper(args.input, args.output, threshold=args.threshold,
               block_size=args.block, ela_qualities=args.ela_qualities,
               min_area_ratio=args.min_area_ratio, border_margin=args.border_margin,
               noise_std_threshold=args.noise_threshold, workers=args.workers)
