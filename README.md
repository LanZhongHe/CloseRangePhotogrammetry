# 近景摄影测量处理系统

从控制点自动检测到相机标定与三维坐标量测的桌面应用。

## 概述

本系统实现了完整的近景摄影测量数据处理流程：

1. **控制点检测** -- 高分辨率影像中环形标志点的自动检测与亚像素定位
2. **像点匹配** -- 检测像点与已知控制场坐标的交互式关联
3. **直接线性变换（DLT）** -- 含可选畸变改正的相机参数估计
4. **空间后方交会** -- 基于共线条件方程的 Levenberg-Marquardt 精密标定
5. **前方交会** -- 立体像对的三维坐标解算

## 功能特性

### 检测流水线

- **自适应预处理** -- CLAHE 直方图均衡化、双边滤波、自适应阈值，参数自动适配目标尺寸
- **多级轮廓筛选** -- 面积、圆度、纵横比、凸度约束，环形结构验证
- **两级亚像素定位**
  - 第一级：内外环轮廓椭圆拟合（~0.1 px 精度）
  - 第二级：灰度加权质心精化（~0.02 px 精度）

### 摄影测量解算

- **像点匹配** -- 交互式 GUI 对话框，含局部预览、重复检测、检查点标记
- **DLT 解算** -- 11 参数 DLT，支持三种畸变模式（无畸变 / K1 / K1+K2+P1+P2），自动提取内外方位元素
- **空间后方交会** -- Levenberg-Marquardt 迭代求解 6 个外方位元素 + 可选内参（f, x0, y0）+ 畸变（K1, K2, K3, P1, P2），协方差矩阵精度评定
- **前方交会** -- 两种方法：
  - DLT 法：由两张影像的 DLT 参数最小二乘交会
  - DLT + 畸变改正 + 迭代精化：利用 DLT 畸变系数去畸变后，在 DLT 框架内迭代精化物方坐标

### GUI 应用

- **三面板布局** -- 文件浏览器、影像查看器、参数与点位信息面板
- **大幅面影像支持** -- 鼠标滚轮平滑缩放、中键平移
- **可视化叠加** -- 检测目标渲染为绿色圆环 + 十字丝 + 编号标注
- **半自动编辑** -- 修正编号、删除误检、手动点击添加
- **键盘微调** -- 方向键移动 1 px，Shift + 方向键 0.1 px 精细调整
- **批处理** -- 后台线程批量检测文件夹内所有影像，带进度条
- **摄影测量面板** -- DLT、后方交会、前方交会专用面板，含参数配置、结果展示、JSON 导出

## 环境要求

- Python 3.10+
- OpenCV (`opencv-python >= 4.8`)
- NumPy (`numpy >= 1.24`)
- PyQt5 (`PyQt5 >= 5.15`)

## 安装

```bash
git clone https://github.com/LanZHongHe/CloseRangePhotogrammetry.git
cd CloseRangePhotogrammetry

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 使用方法

```bash
python main.py
```

### 检测流程

1. **打开文件夹** -- 点击"Open Folder..."选择包含影像的目录（.jpg, .png, .tif, .bmp）
2. **加载影像** -- 左侧面板点击文件名显示影像
3. **调整参数** -- 右侧面板调节目标尺寸、圆度下限、面积容差
4. **检测** -- 点击"Detect Current"单张检测或"Detect All"批处理
5. **检查修正** -- 点击标记查看详情，修正编号、删除误检或手动添加遗漏目标
6. **导出** -- 点击"Export JSON..."保存检测结果

### 摄影测量流程

1. **加载控制场** -- 导入已知三维坐标的控制点文本文件
2. **像点匹配** -- 为每张影像通过匹配对话框关联检测点与控制场坐标
3. **DLT 解算** -- 计算 11 个 DLT 参数（可选畸变改正）
4. **空间后方交会** -- 迭代求解相机内参、外参和畸变系数
5. **前方交会** -- 利用两张影像参数计算物点三维坐标
6. **可视化** -- 生成相机位置、光线交会、残差分析的三维可视化

### 可视化脚本

```bash
# DLT 与后方交会结果对比
python compare_results.py

# 残差分析（多页 PDF）
python visualize_residuals.py

# 镜头畸变可视化
python visualize_distortion.py

# 后方交会三维可视化
python visualize_resection.py

# 前方交会三维可视化
python visualize_forward_intersection.py
```

## 输出格式

### 检测结果

```json
{
  "image": "photo/DSC_0035.JPG",
  "image_size": [8256, 5504],
  "detection_time": "2026-04-14T16:14:33",
  "targets": [
    {
      "id": "181",
      "pixel_x": 8062.7158,
      "pixel_y": 4129.8107,
      "confidence": 0.6986,
      "source": "auto",
      "subpixel_method": "centroid",
      "ellipse": {
        "semi_major": 74.34,
        "semi_minor": 68.87,
        "angle_deg": 118.85
      },
      "eccentricity": 0.3766
    }
  ]
}
```

### 匹配结果

```json
{
  "image_path": "photo/DSC_0035.JPG",
  "matched_points": [
    {
      "detected_id": "001",
      "control_id": "356",
      "pixel_x": 5210.1,
      "pixel_y": 425.0,
      "image_x_mm": 4.705,
      "image_y_mm": 10.119,
      "obj_x": 3693.4,
      "obj_y": 5936.2,
      "obj_z": 792.2,
      "is_manual": false,
      "is_check": false
    }
  ]
}
```

### DLT 结果

```json
{
  "image": "DSC_0035",
  "results": {
    "dlt_with_k1k2p1p2": {
      "L_params": [-0.0714, "..."],
      "sigma0_mm": 0.0123,
      "num_iterations": 4,
      "intrinsics": { "f": 51.966, "x0": 0.0255, "y0": -0.2770 },
      "exterior": { "Xs": 4019.83, "Ys": 1369.81, "Zs": -69.27, "..." : "..." },
      "distortion": { "K1": 1.92e-5, "K2": 0, "P1": 0, "P2": 0 }
    }
  }
}
```

### 空间后方交会结果

```json
{
  "converged": true,
  "num_iterations": 14,
  "sigma0_mm": 0.002056,
  "sigma0_px": 0.47,
  "intrinsics": { "f": 51.982, "x0": 0.014, "y0": -0.252 },
  "exterior": { "Xs": 4019.72, "Ys": 1369.88, "Zs": -69.45, "..." : "..." },
  "distortion": { "K1": -2.71e-6, "K2": 3.87e-9, "P1": 6.29e-7, "P2": 4.11e-7 },
  "param_std": { "Xs": 0.245, "Ys": 0.978, "Zs": 0.198, "f": 0.018 }
}
```

### 前方交会结果

```json
{
  "sigma0_mm": 0.002992,
  "points": [
    {
      "point_id": "001",
      "X": 2725.85, "Y": 3966.14, "Z": -77.52,
      "intersection_angle_deg": 51.02,
      "residuals": { "vx1": 0.001, "vy1": -0.002, "vx2": 0.003, "vy2": -0.001 }
    }
  ]
}
```

## 项目结构

```
CloseRangePhotogrammetry/
├── main.py                          # 应用入口
├── requirements.txt                 # Python 依赖
├── gui/
│   ├── main_window.py               # PyQt5 主窗口（ImageViewer, DetectionWorker）
│   ├── matching_dialog.py           # 像点匹配对话框
│   ├── resection_panel.py           # 后方交会面板
│   ├── dlt_panel.py                 # DLT 解算面板
│   ├── forward_intersection_panel.py # 前方交会面板
│   ├── image_pick_dialog.py         # 影像刺点对话框
│   └── detection_params_dialog.py   # 检测参数对话框
├── src/
│   ├── preprocessing.py             # CLAHE、双边滤波、自适应阈值
│   ├── detection.py                 # 轮廓粗检测
│   ├── subpixel.py                  # 椭圆拟合 + 质心精化
│   ├── id_recognition.py            # 顺序编号
│   ├── data_model.py                # TargetPoint, EllipseInfo, DetectionResult
│   ├── io_utils.py                  # JSON 序列化
│   ├── camera_model.py              # 相机内参、畸变、旋转矩阵
│   ├── matching.py                  # MatchedPoint 数据结构与 I/O
│   ├── dlt.py                       # DLT 解算（11 参数及含畸变）
│   ├── resection.py                 # 空间后方交会（LM 算法）
│   └── forward_intersection.py      # 前方交会（DLT 法与后方交会法）
├── photo/                           # 示例影像
├── output/                          # 解算结果（JSON）与可视化
├── docs/                            # 控制场坐标与实验指导书
├── visualize_resection.py           # 后方交会三维可视化
├── visualize_forward_intersection.py # 前方交会三维可视化
├── visualize_residuals.py           # 残差分析图
├── visualize_distortion.py          # 镜头畸变可视化
├── compare_results.py               # DLT 与后方交会对比
└── whuthesis/                       # 实习报告（whu-thesis 模板）
```

## 算法说明

### DLT（直接线性变换）

DLT 通过 11 个参数建立三维物方坐标与二维像方坐标之间的线性映射。每个控制点提供 2 个方程，最少需要 6 个点。支持三种模式：

- **无畸变**：基本 11 参数 DLT
- **仅 K1**：径向畸变改正，迭代更新畸变中心
- **K1+K2+P1+P2**：完整 Brown 畸变模型（径向 + 偏心）

### 空间后方交会

基于共线条件方程，求解影像的 6 个外方位元素（Xs, Ys, Zs, ω, φ, κ），可选求解内方位元素（f, x0, y0）和畸变系数（K1--K3, P1, P2）。采用 Levenberg-Marquardt 算法保证稳健收敛：

```
(B^T B + λ · diag(B^T B)) · δ = B^T L
```

其中 B 为雅可比矩阵，L 为残差向量，λ 为自适应阻尼因子。

### 前方交会

实现了两种方法：

1. **DLT 法**：对于每个同名点，两张影像的 DLT 方程共提供 4 个方程，求解 3 个未知数（X, Y, Z），最小二乘求解。

2. **DLT + 畸变改正 + 迭代精化**：
   - 利用 DLT 畸变系数对像点坐标去畸变：`x_undist = x - δx(x, y)`
   - 用去畸变坐标通过 DLT 计算初始物方坐标
   - 迭代精化：DLT 正算投影 → 计算残差 → 雅可比矩阵改正物方坐标 → 直至收敛

## 精度汇总

| 方法 | σ₀ (mm) | σ₀ (px) |
|------|---------|---------|
| DLT（无畸变） | 0.005 -- 0.013 | 1.2 -- 3.0 |
| DLT + K1 | 0.005 -- 0.012 | 1.2 -- 2.8 |
| DLT + K1K2P1P2 | 0.004 -- 0.012 | 1.0 -- 2.8 |
| 空间后方交会 | 0.001 -- 0.002 | 0.33 -- 0.47 |
| 前方交会 | ~0.003 | ~0.7 |

## 致谢

本项目为武汉大学遥感信息工程学院近景摄影测量课程实习成果。
