

# 学术论文图像取证工具

针对学术论文中可能出现的图像重复使用、拼接、复制‑移动等造假行为，本工具在 **“宁放过、不误报”** 的原则下，提供了一套自动化多维度检测流程。核心策略包括：**自适应子图拆解**、**动态空间连续性校验**、**GMS+RANSAC 增强匹配**、**ELA 背景噪声校正**以及**快速噪声方差分析**。

## 主要特点

* **智能子图拆解**  
采用 OTSU 自适应阈值与轮廓检测，自动将组合大图拆分为独立子图。同时预留 DCI 接口（需 PyMuPDF 页面信息），可进一步利用 PDF 原始定位提升精度。
* **误报多重抑制**  
局部重复检测通过动态面积比例、紧凑度、边缘比例等多重条件过滤纹理误报；ELA 分析引入全局背景基线校正，仅突出显著偏离背景的异常区域；复制‑移动检测加入 RANSAC 单应性验证，剔除不可靠的位移簇。
* **GMS 增强与几何验证**  
复制‑移动检测在 SIFT 基础上，利用邻域运动支持度（GMS 简化版）过滤误匹配，再通过 DBSCAN 聚类位移并做 RANSAC 一致性检查，确保检测到的复制区域具有真实的几何关系。
* **高效噪声分析**  
使用积分图/盒式滤波器快速计算局部噪声方差，并依据图像自身噪声水平动态设定异常阈值，可发现无压缩痕迹的拼接操作。
* **精确标注与对比图**  
所有检测结果均将坐标修正回原图坐标系，在原图上用红框和蓝箭头标示异常区域，并自动生成原图‑标注图的并排对比图，方便审查与展示。
* **可选的并行加速**  
支持通过 `--workers` 参数指定子图分析进程数，充分利用多核 CPU 提升批量处理效率。

## 环境依赖

```bash
pip install PyMuPDF lxml pillow imagehash opencv-python scikit-learn
```

## 使用方式

### 基础命令

```bash
python forensics.py --input paper.pdf --output results/
```

### 完整参数

|参数|默认|说明|
|-|-|-|
|--input|必填|PDF 或 XML 文件路径|
|--output|必填|输出目录|
|--threshold|2|全局子图相似度阈值（pHashing 距离，越小越严格）|
|--block|64|局部重复检测的滑动窗口大小（像素）|
|--ela\_qualities|75 90 98|ELA 使用的三个 JPEG 质量因子|
|--min\_area\_ratio|0.02|子图最小面积占原图比例（过滤噪声轮廓）|
|--border\_margin|10|子图轮廓向外扩展的像素数|
|--noise\_threshold|50|噪声方差不一致性基础阈值|
|--workers|1|并行处理子图的进程数（建议设为 CPU 核心数）|

## 输出结构

results/
├── images/            # 从 PDF/XML 提取的原始大图
├── subfigures/        # 拆解出的子图（含全图回退）
├── annotated/         # 在原图上标注的检测结果
├── comparisons/       # 原图与标注图的并排对比图
└── report.json        # 完整 JSON 检测报告

### 报告字段示例

```json
{
  "total\_subfigures": 42,
  "global\_duplicates": \[\["sub1.png", "sub2.png", 1]],
  "local\_duplicates": {
    "sub3.png": \[\[\[x1,y1], \[x2,y2], ...], ...]
  },
  "ela\_noise": {
    "sub4.png": {
      "ela": {"75": 0.0, "90": 0.0, "98": 1.2},
      "noise\_inconsistency": 62.4
    }
  },
  "copy\_move\_vectors": {
    "sub5.png": \[\[-120.5, 80.3]]
  }
}
```

* global\_duplicates：子图间重复（文件名、距离）。
* local\_duplicates：子图内疑似拼接的块坐标组（坐标已映射回原图）。
* ela\_noise：ELA 异常百分比（仅记录显著偏离背景的局部区域）和噪声方差评分（非零表示空间噪声不一致）。
* copy\_move\_vectors：经过 GMS+RANSAC 验证的复制‑移动位移向量。

## 设计哲学与调参建议

本工具所有默认阈值均偏向 **极低假阳性**。实际使用时可依照场景微调：

* 若希望提升召回率（发现更多可疑线索），可适当增大 `--threshold`、减小 `--noise\_threshold` 或降低 `--min\_area\_ratio`。
* 对于排版极度复杂的论文，调整 `--border\_margin` 和 `--block` 可能改善子图拆解与局部检测效果。
* 并行加速：`--workers` 设为 CPU 物理核心数可获得最佳性能，但请注意内存占用。

检测结果仅供技术参考，最终判定需结合人工审查。

## 许可

MIT License
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

