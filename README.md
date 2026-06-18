# 学术论文图像取证工具

## 概述

本工具集包含两个独立的 Python 脚本，专门用于学术论文图像的自动化取证分析，帮助检测图像篡改、重复使用、复制‑移动等学术不端行为。

* **`scan\_images.py`**：直接对图片文件夹进行批处理分析，适用于已提取好的图片集（如从 PDF 手动导出或已有子图文件夹）。
* **`forensics.py`**：完整的端到端流程，支持从 **PDF** 或 **XML（如 JATS）** 文件中自动提取嵌入图像，并基于视觉分割拆解组合大图中的子图，再进行多维度取证检测。

两者共享核心检测算法，包括：

* **全局重复检测**：基于感知哈希（pHash）查找不同图像间的相似重复。
* **局部重复检测**：在单张图像内，通过滑动窗口 + DBSCAN 聚类发现可疑的重复区域（如拼接、复制粘贴）。
* **复制‑移动检测**：利用 SIFT 特征匹配、GMS（网格运动统计）支持度与 RANSAC 单应性验证，检测图像内的复制‑移动篡改。
* **ELA（误差水平分析）+ 噪声分析**：评估 JPEG 压缩伪影异常与局部噪声方差一致性，辅助识别后期编辑区域，并自动生成彩色 ELA 热图（JET 色图）直观显示异常。
* **可视化标注**：在原图上绘制红色矩形框（局部重复）和蓝色箭头（复制‑移动位移方向），并生成原图与标注图的并排对比图。

所有检测结果汇总为 JSON 格式报告，便于后续审查与统计。

## 安装与依赖

### 环境要求

* Python 3.7 或以上
* 建议使用虚拟环境（如 conda 或 venv）

### 安装依赖

```bash
pip install pillow imagehash opencv-python numpy scikit-learn PyMuPDF lxml
```

**各库用途**：

* `pillow`：图像读写与基础处理
* `imagehash`：感知哈希计算
* `opencv-python`：图像处理、特征提取（SIFT）、DBSCAN 聚类
* `numpy`：数值计算
* `scikit-learn`：DBSCAN 聚类实现
* `PyMuPDF (fitz)`：PDF 图像提取（仅 `forensics.py` 需要）
* `lxml`：XML 解析（仅 `forensics.py` 需要）

## 使用方法

### 1\. 直接分析图片文件夹（`scan\_images.py`）

适用于已有图片文件（如手动从论文中截取或导出的子图）。

**基本用法**：

```bash
python scan\_images.py --image\_dir /path/to/images --output /path/to/output
```

**完整参数**：

|参数|类型|默认值|说明|
|-|-|-|-|
|`--image\_dir`|字符串|必填|包含图片文件的文件夹路径|
|`--output`|字符串|必填|输出目录（将自动创建子目录）|
|`--threshold`|整数|2|全局 pHash 距离阈值（越小越严格）|
|`--block`|整数|64|局部检测的滑动窗口大小（像素）|
|`--ela\_qualities`|三个整数|75 90 98|ELA 分析的 JPEG 质量级别|
|`--noise\_threshold`|浮点数|50|噪声方差不一致性基础阈值|
|`--workers`|整数|1|并行处理的进程数（>1 加速）|
|`--local\_eps`|浮点数|0.2|局部 DBSCAN 的聚类半径（归一化汉明距离）|
|`--local\_min\_samples`|整数|3|局部 DBSCAN 的最小样本数|
|`--cm\_min\_match`|整数|15|复制‑移动检测所需最少匹配特征点数|
|`--cm\_eps`|浮点数|2.0|复制‑移动位移向量聚类的 DBSCAN 半径（像素）|
|`--cm\_min\_samples`|整数|5|复制‑移动聚类的最小样本数|
|`--cm\_gms\_radius`|整数|30|GMS 支持度计算的空间邻域半径（像素）|
|`--cm\_ransac\_thresh`|浮点数|5.0|RANSAC 单应性验证的重投影误差阈值|

**示例**：

```bash
python scan\_images.py --image\_dir ./figures --output ./results --workers 4 --threshold 3 --block 48
```

### 2\. 端到端论文分析（`forensics.py`）

直接从 PDF 或 XML 文件中提取图像，自动拆解组合图（如多面板子图），再进行所有检测。

**基本用法**：

```bash
python forensics.py --input paper.pdf --output ./paper\_forensics
# 或
python forensics.py --input article.xml --output ./xml\_forensics
```

**完整参数**（除继承 `scan\_images.py` 的参数外，额外包含）：

|参数|类型|默认值|说明|
|-|-|-|-|
|`--input`|字符串|必填|PDF 或 XML 文件路径|
|`--output`|字符串|必填|输出目录|
|`--min\_area\_ratio`|浮点数|0.02|子图最小面积占原图比例（过滤噪点）|
|`--border\_margin`|整数|10|子图裁剪时边缘扩展像素（避免切到内容）|
|（其他参数同 `scan\_images.py`）||||

**示例**：

```bash
python forensics.py --input ./sample.pdf --output ./sample\_out --workers 4 --block 64 --min\_area\_ratio 0.015
```

## 输出结构

无论使用哪个脚本，输出目录均包含以下内容：

```
output/
├── report.json                 # 所有检测结果的 JSON 报告
├── annotated/                  # 带有标注（矩形+箭头）的原始大图
├── comparisons/                # 原图与标注图的并排对比图
├── ela\_heatmaps/               # 异常子图的彩色 ELA 热图（仅当 ELA 异常时生成）
```

针对于forensics.py ，输出目录还会包含

```
    ├── images/                 # 从 PDF/XML 提取的原始大图
    └── subfigures/             # 拆解后的子图文件
```

**报告内容**（`report.json`）：

* `total\_images` 或 `total\_subfigures`：分析对象总数
* `global\_duplicates`：全局重复图片对列表，包含文件名和 pHash 距离
* `local\_duplicates`：各图片内检测到的局部重复区域坐标组（以子图文件名索引）
* `ela\_noise`：各子图的 ELA 异常面积比（按质量级别）和噪声不一致性数值
* `copy\_move\_vectors`：各子图中检测到的复制‑移动位移向量（dx, dy）

## 注意事项

1. **图像格式**：支持常见格式（PNG, JPG, JPEG, BMP, TIFF）。
2. **并行加速**：`--workers` 可设置为 CPU 核心数以加快批量处理，但需确保内存充足。
3. **参数调优**：若出现大量误报，可适当提高 `--threshold`（全局）或降低 `--local\_eps`、增加 `--local\_min\_samples`；若漏检则反向调整。
4. **ELA 热图**：'forensics.py'仅当 ELA 异常面积比 > 1% 或噪声不一致性 > 0 时生成，避免大量无意义文件；而'scan\_images.py'为所有成功读取的图片生成彩色 ELA 热图（无论是否异常），可用于人工筛查。
5. **XML 支持**：目前支持符合 JATS 标准的 XML，其中图像以 `<graphic>` 或 `<media>` 元素内嵌 base64 数据。
6. **依赖版本**：建议使用 OpenCV 4.x 以上，以支持 SIFT（非专利限制版本需额外配置）。

\---

# Paper Forensics Toolkit: Automated Image Analysis for Scientific Integrity

## Overview

This toolkit comprises two independent Python scripts designed for automated forensic analysis of academic paper images, assisting in detecting image manipulation, duplication, and copy‑move tampering that may indicate scientific misconduct.

* **`scan\_images.py`** : A standalone batch processor for a folder of images, suitable when images are already extracted (e.g., manually exported from PDF or a pre‑existing subfigure collection).
* **`forensics.py`** : A complete end‑to‑end pipeline that automatically extracts embedded images from **PDF** or **XML (e.g., JATS)** files, visually segments composite figures into subfigures, and then performs multi‑dimensional forensic analyses on each subfigure.

Both scripts share the core detection algorithms, including:

* **Global duplicate detection** : Finds similar/duplicate images across the dataset using perceptual hashing (pHash).
* **Local duplicate detection** : Within a single image, uses sliding‑window pHash + DBSCAN clustering to locate suspicious repeated regions (e.g., splicing, cloning).
* **Copy‑move detection** : Employs SIFT feature matching, GMS (Grid‑based Motion Statistics) support, and RANSAC homography verification to detect copied‑and‑pasted regions within an image.
* **ELA (Error Level Analysis) + noise analysis** : Evaluates JPEG compression artifacts and local noise variance inconsistencies to flag potential post‑processing areas. Automatically generates color ELA heatmaps (JET colormap) for visual inspection.
* **Visual annotations** : Draws red bounding boxes (local duplicates) and blue arrows (copy‑move displacement vectors) on the original images, and produces side‑by‑side comparison images.

All results are consolidated into a JSON report for further review and statistical analysis.

## Installation \& Dependencies

### Requirements

* Python 3.7 or above
* A virtual environment (conda/venv) is recommended

### Install Dependencies

```bash
pip install pillow imagehash opencv-python numpy scikit-learn PyMuPDF lxml
```

**Package purposes**:

* `pillow` : Image I/O and basic processing
* `imagehash` : Perceptual hash computation
* `opencv-python` : Image processing, feature extraction (SIFT), DBSCAN clustering
* `numpy` : Numerical computations
* `scikit-learn` : DBSCAN clustering implementation
* `PyMuPDF (fitz)` : PDF image extraction (only needed for `forensics.py`)
* `lxml` : XML parsing (only needed for `forensics.py`)

## Usage

### 1\. Direct Image Folder Analysis (`scan\_images.py`)

Use this when you already have image files (e.g., manually extracted or a folder of subfigures).

**Basic command**:

```bash
python scan\_images.py --image\_dir /path/to/images --output /path/to/output
```

**Full parameters**:

|Parameter|Type|Default|Description|
|-|-|-|-|
|`--image\_dir`|str|required|Path to the folder containing images|
|`--output`|str|required|Output directory (subdirectories will be created)|
|`--threshold`|int|2|Global pHash distance threshold (lower = stricter)|
|`--block`|int|64|Sliding window size for local detection (pixels)|
|`--ela\_qualities`|three ints|75 90 98|JPEG quality levels for ELA analysis|
|`--noise\_threshold`|float|50|Base threshold for noise variance inconsistency|
|`--workers`|int|1|Number of parallel processes (>1 for speed)|
|`--local\_eps`|float|0.2|DBSCAN eps for local detection (normalized Hamming)|
|`--local\_min\_samples`|int|3|DBSCAN min\_samples for local detection|
|`--cm\_min\_match`|int|15|Minimum feature matches required for copy‑move|
|`--cm\_eps`|float|2.0|DBSCAN eps for copy‑move displacement clustering (pixels)|
|`--cm\_min\_samples`|int|5|DBSCAN min\_samples for copy‑move clustering|
|`--cm\_gms\_radius`|int|30|Spatial neighborhood radius for GMS support (pixels)|
|`--cm\_ransac\_thresh`|float|5.0|RANSAC reprojection error threshold for homography|

**Example**:

```bash
python scan\_images.py --image\_dir ./figures --output ./results --workers 4 --threshold 3 --block 48
```

### 2\. End‑to‑End Paper Analysis (`forensics.py`)

Extracts images directly from a PDF or XML file, automatically splits composite figures into subfigures, and performs all detections.

**Basic command**:

```bash
python forensics.py --input paper.pdf --output ./paper\_forensics
# or
python forensics.py --input article.xml --output ./xml\_forensics
```

**Additional parameters** (beyond those inherited from `scan\_images.py`):

|Parameter|Type|Default|Description|
|-|-|-|-|
|`--input`|str|required|Path to the PDF or XML file|
|`--output`|str|required|Output directory|
|`--min\_area\_ratio`|float|0.02|Minimum subfigure area relative to original (filters noise)|
|`--border\_margin`|int|10|Extra margin (pixels) around subfigure crops|
|(Other parameters same as `scan\_images.py`)||||

**Example**:

```bash
python forensics.py --input ./sample.pdf --output ./sample\_out --workers 4 --block 64 --min\_area\_ratio 0.015
```

## Output Structure

Regardless of the script used, the output directory contains the following:

```
output/
├── report.json                 # JSON report with all detection results
├── annotated/                  # Original large images with annotations (rectangles + arrows)
├── comparisons/                # Side‑by‑side comparisons of original vs annotated
├── ela\_heatmaps/               # Color ELA heatmaps for subfigures with detected anomalies
```

specific to forensics.py, the output directory contains the following:

```
    ├── images/                 # Extracted original large images from PDF/XML
    └── subfigures/             # Segmented subfigure files
```

**Report contents** (`report.json`):

* `total\_images` or `total\_subfigures` : Total number of analyzed objects
* `global\_duplicates` : List of duplicate image pairs, with filenames and pHash distances
* `local\_duplicates` : Per‑image local duplicate region coordinates (indexed by subfigure filename)
* `ela\_noise` : ELA abnormal area ratios (per quality level) and noise inconsistency values for each subfigure
* `copy\_move\_vectors` : Detected copy‑move displacement vectors (dx, dy) per subfigure

## Important Notes

1. **Image formats** : Supports common formats (PNG, JPG, JPEG, BMP, TIFF).
2. **Parallel processing** : Set `--workers` to the number of CPU cores to speed up batch processing, but ensure sufficient memory.
3. **Parameter tuning** : If many false positives occur, increase `--threshold` (global) or decrease `--local\_eps`, increase `--local\_min\_samples`; adjust oppositely if missed detections are observed.
4. **ELA heatmaps** : Generated only when ELA abnormal area ratio > 1% or noise inconsistency > 0 to avoid creating numerous unnecessary files if you use 'forensics.py', while 'scan\_images.py' will generate ELA heatmaps for every successfully read image.
5. **XML support** : Currently supports JATS‑compliant XML where images are embedded as base64 data within `<graphic>` or `<media>` elements.
6. **Dependency versions** : OpenCV 4.x or above is recommended; ensure SIFT is available (may require additional configuration in non‑free versions).

