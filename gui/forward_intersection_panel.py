"""GUI panel for forward intersection from two-image DLT parameters."""

import json
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QTextEdit,
    QFileDialog, QHeaderView, QComboBox, QDoubleSpinBox,
    QSpinBox, QMessageBox, QSplitter,
)
from PyQt5.QtCore import Qt

from src.forward_intersection import (
    TiePoint, forward_intersection_dlt, forward_intersection_dlt_distorted,
    forward_intersection_resection,
    compute_point_distance, ForwardIntersectionResult,
)
from src.camera_model import CameraIntrinsics, ExteriorOrientation, DistortionCoefficients
from gui.image_pick_dialog import ImagePickDialog


class ForwardIntersectionPanel(QWidget):
    """Panel for computing 3D coordinates via forward intersection."""

    def __init__(self, image_paths: list[str] | None = None,
                 intrinsics: CameraIntrinsics | None = None,
                 parent=None):
        super().__init__(parent)
        self._L1: np.ndarray | None = None
        self._L2: np.ndarray | None = None
        self._dist1: DistortionCoefficients | None = None
        self._dist2: DistortionCoefficients | None = None
        self._intr1: CameraIntrinsics | None = None
        self._intr2: CameraIntrinsics | None = None
        self._ext1: ExteriorOrientation | None = None
        self._ext2: ExteriorOrientation | None = None
        self._intrinsics: CameraIntrinsics = intrinsics or CameraIntrinsics()
        self._result: ForwardIntersectionResult | None = None
        self._image_paths: list[str] = image_paths or []
        self._image1_path: str = ""
        self._image2_path: str = ""
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # DLT parameters group
        dlt_group = QGroupBox("DLT 参数")
        dlt_layout = QVBoxLayout(dlt_group)

        btn_row = QHBoxLayout()
        self._load_L1_btn = QPushButton("导入影像1 L参数 (JSON)")
        self._load_L1_btn.clicked.connect(lambda: self._load_l_params(1))
        btn_row.addWidget(self._load_L1_btn)
        self._load_L2_btn = QPushButton("导入影像2 L参数 (JSON)")
        self._load_L2_btn.clicked.connect(lambda: self._load_l_params(2))
        btn_row.addWidget(self._load_L2_btn)
        dlt_layout.addLayout(btn_row)

        self._l_status = QLabel("未加载 L 参数")
        dlt_layout.addWidget(self._l_status)
        layout.addWidget(dlt_group)

        # Image selection group
        img_group = QGroupBox("影像文件")
        img_layout = QVBoxLayout(img_group)

        img_row1 = QHBoxLayout()
        img_row1.addWidget(QLabel("影像1:"))
        self._img1_label = QLabel("未选择")
        img_row1.addWidget(self._img1_label, 1)
        img1_btn = QPushButton("选择...")
        img1_btn.clicked.connect(lambda: self._select_image(1))
        img_row1.addWidget(img1_btn)
        img_layout.addLayout(img_row1)

        img_row2 = QHBoxLayout()
        img_row2.addWidget(QLabel("影像2:"))
        self._img2_label = QLabel("未选择")
        img_row2.addWidget(self._img2_label, 1)
        img2_btn = QPushButton("选择...")
        img2_btn.clicked.connect(lambda: self._select_image(2))
        img_row2.addWidget(img2_btn)
        img_layout.addLayout(img_row2)

        # Camera intrinsics for pixel-to-mm conversion
        intr_row = QHBoxLayout()
        intr_row.addWidget(QLabel("传感器宽(mm):"))
        self._sw_spin = QDoubleSpinBox()
        self._sw_spin.setRange(0.1, 200.0)
        self._sw_spin.setValue(self._intrinsics.sensor_width)
        self._sw_spin.setDecimals(2)
        intr_row.addWidget(self._sw_spin)
        intr_row.addWidget(QLabel("影像宽(px):"))
        self._iw_spin = QSpinBox()
        self._iw_spin.setRange(1, 50000)
        self._iw_spin.setValue(self._intrinsics.img_width)
        intr_row.addWidget(self._iw_spin)
        intr_row.addWidget(QLabel("影像高(px):"))
        self._ih_spin = QSpinBox()
        self._ih_spin.setRange(1, 50000)
        self._ih_spin.setValue(self._intrinsics.img_height)
        intr_row.addWidget(self._ih_spin)
        img_layout.addLayout(intr_row)

        layout.addWidget(img_group)

        # Tie points group
        tp_group = QGroupBox("同名像点")
        tp_layout = QVBoxLayout(tp_group)

        self._tie_table = QTableWidget(0, 5)
        self._tie_table.setHorizontalHeaderLabels([
            "点号", "影像1 x (mm)", "影像1 y (mm)", "影像2 x (mm)", "影像2 y (mm)"
        ])
        self._tie_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tp_layout.addWidget(self._tie_table)

        btn_row2 = QHBoxLayout()
        pick_btn = QPushButton("刺点添加")
        pick_btn.clicked.connect(self._pick_tie_point)
        btn_row2.addWidget(pick_btn)
        add_btn = QPushButton("手动添加行")
        add_btn.clicked.connect(self._add_row)
        btn_row2.addWidget(add_btn)
        del_btn = QPushButton("删除行")
        del_btn.clicked.connect(self._del_row)
        btn_row2.addWidget(del_btn)
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(lambda: self._tie_table.setRowCount(0))
        btn_row2.addWidget(clear_btn)
        tp_layout.addLayout(btn_row2)
        layout.addWidget(tp_group)

        # Info and run
        self._info_label = QLabel()
        layout.addWidget(self._info_label)

        self._run_btn = QPushButton("运行前方交会")
        self._run_btn.clicked.connect(self._run)
        layout.addWidget(self._run_btn)

        # Results area (splitter: text + distance selector)
        result_splitter = QSplitter(Qt.Vertical)

        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        result_splitter.addWidget(self._result_text)

        # Distance computation
        dist_widget = QWidget()
        dist_layout = QVBoxLayout(dist_widget)
        dist_layout.setContentsMargins(0, 0, 0, 0)

        dist_label = QLabel("点间距离计算:")
        dist_layout.addWidget(dist_label)

        dist_row = QHBoxLayout()
        dist_row.addWidget(QLabel("点1:"))
        self._dist_p1 = QComboBox()
        dist_row.addWidget(self._dist_p1)
        dist_row.addWidget(QLabel("点2:"))
        self._dist_p2 = QComboBox()
        dist_row.addWidget(self._dist_p2)
        self._dist_btn = QPushButton("计算距离")
        self._dist_btn.clicked.connect(self._calc_distance)
        dist_row.addWidget(self._dist_btn)
        dist_layout.addLayout(dist_row)

        self._dist_result = QLabel("")
        dist_layout.addWidget(self._dist_result)

        # Export button
        export_btn = QPushButton("导出结果")
        export_btn.clicked.connect(self._export)
        dist_layout.addWidget(export_btn)

        result_splitter.addWidget(dist_widget)
        layout.addWidget(result_splitter)

    def _load_l_params(self, image_num: int):
        """Load DLT L parameters, distortion, and intrinsics from a JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, f"导入影像{image_num} DLT 参数", "",
            "JSON文件 (*.json);;所有文件 (*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Support both flat and nested JSON formats
            if "L_params" in data:
                # Flat format (single result)
                L = np.array(data["L_params"])
                dist_data = data.get("distortion", {})
                intr_data = data.get("intrinsics", {})
            elif "results" in data:
                # Nested format from compare/analysis scripts
                # Try to find the best available result (prefer with distortion)
                results = data["results"]
                if "dlt_with_k1k2p1p2" in results:
                    key = "dlt_with_k1k2p1p2"
                elif "dlt_with_k1" in results:
                    key = "dlt_with_k1"
                else:
                    key = "dlt_no_distortion"
                L = np.array(results[key]["L_params"])
                dist_data = results[key].get("distortion", {})
                intr_data = results[key].get("intrinsics", {})
            else:
                raise ValueError("JSON 格式不正确，未找到 L_params 或 results 字段")

            if len(L) != 11:
                raise ValueError(f"L 参数应有 11 个，实际有 {len(L)}")

            # Extract distortion coefficients
            dist = DistortionCoefficients(
                K1=dist_data.get("K1", 0.0),
                K2=dist_data.get("K2", 0.0),
                K3=dist_data.get("K3", 0.0),
                P1=dist_data.get("P1", 0.0),
                P2=dist_data.get("P2", 0.0),
            )

            # Extract intrinsics (x0, y0 for undistortion)
            intr = CameraIntrinsics(
                f=intr_data.get("f", self._intrinsics.f),
                x0=intr_data.get("x0", 0.0),
                y0=intr_data.get("y0", 0.0),
                sensor_width=self._intrinsics.sensor_width,
                sensor_height=self._intrinsics.sensor_height,
                img_width=self._intrinsics.img_width,
                img_height=self._intrinsics.img_height,
            )

            if image_num == 1:
                self._L1 = L
                self._dist1 = dist
                self._intr1 = intr
            else:
                self._L2 = L
                self._dist2 = dist
                self._intr2 = intr

            # Build status message
            has_dist = any([dist.K1, dist.K2, dist.K3, dist.P1, dist.P2])
            dist_str = "含畸变" if has_dist else "无畸变"
            self._l_status.setText(
                f"影像1: {'已加载' if self._L1 is not None else '未加载'}  "
                f"影像2: {'已加载' if self._L2 is not None else '未加载'}  "
                f"[{dist_str}]"
            )
        except Exception as e:
            QMessageBox.warning(self, "加载失败", str(e))

    def _select_image(self, image_num: int):
        """Select an image file for picking tie points."""
        start_dir = ""
        if self._image_paths:
            import os
            start_dir = os.path.dirname(self._image_paths[0])
        path, _ = QFileDialog.getOpenFileName(
            self, f"选择影像{image_num}", start_dir,
            "影像文件 (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;所有文件 (*)"
        )
        if not path:
            return
        if image_num == 1:
            self._image1_path = path
            self._img1_label.setText(path.split("/")[-1].split("\\")[-1])
        else:
            self._image2_path = path
            self._img2_label.setText(path.split("/")[-1].split("\\")[-1])

    def _pick_tie_point(self):
        """Pick a tie point on both images by clicking."""
        if not self._image1_path or not self._image2_path:
            QMessageBox.warning(self, "缺少影像", "请先选择两张影像文件。")
            return

        # Pick on image 1
        dlg1 = ImagePickDialog(self._image1_path, "影像1", self)
        if dlg1.exec_() != ImagePickDialog.Accepted:
            return
        p1 = dlg1.get_pixel_coords()
        if p1 is None:
            QMessageBox.warning(self, "未选点", "请在影像1上点击选择一个点。")
            return

        # Pick on image 2
        dlg2 = ImagePickDialog(self._image2_path, "影像2", self)
        if dlg2.exec_() != ImagePickDialog.Accepted:
            return
        p2 = dlg2.get_pixel_coords()
        if p2 is None:
            QMessageBox.warning(self, "未选点", "请在影像2上点击选择一个点。")
            return

        # Convert to mm
        x1_mm, y1_mm = self._pixel_to_mm(p1[0], p1[1])
        x2_mm, y2_mm = self._pixel_to_mm(p2[0], p2[1])

        # Add row
        row = self._tie_table.rowCount()
        self._tie_table.insertRow(row)
        self._tie_table.setItem(row, 0, QTableWidgetItem(f"{row+1:03d}"))
        self._tie_table.setItem(row, 1, QTableWidgetItem(f"{x1_mm:.6f}"))
        self._tie_table.setItem(row, 2, QTableWidgetItem(f"{y1_mm:.6f}"))
        self._tie_table.setItem(row, 3, QTableWidgetItem(f"{x2_mm:.6f}"))
        self._tie_table.setItem(row, 4, QTableWidgetItem(f"{y2_mm:.6f}"))

    def _pixel_to_mm(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """Convert pixel coordinates to image coordinates in mm."""
        img_w = self._iw_spin.value()
        img_h = self._ih_spin.value()
        sensor_w = self._sw_spin.value()
        pixel_size = sensor_w / img_w  # mm per pixel
        x_mm = (pixel_x - img_w / 2.0) * pixel_size
        y_mm = -(pixel_y - img_h / 2.0) * pixel_size
        return x_mm, y_mm

    def _add_row(self):
        row = self._tie_table.rowCount()
        self._tie_table.insertRow(row)
        self._tie_table.setItem(row, 0, QTableWidgetItem(f"{row+1:03d}"))
        for col in range(1, 5):
            self._tie_table.setItem(row, col, QTableWidgetItem("0.0"))

    def _del_row(self):
        row = self._tie_table.currentRow()
        if row >= 0:
            self._tie_table.removeRow(row)

    def _get_tie_points(self) -> list[TiePoint]:
        """Read tie points from the table."""
        points = []
        for row in range(self._tie_table.rowCount()):
            try:
                pid = self._tie_table.item(row, 0).text()
                x1 = float(self._tie_table.item(row, 1).text())
                y1 = float(self._tie_table.item(row, 2).text())
                x2 = float(self._tie_table.item(row, 3).text())
                y2 = float(self._tie_table.item(row, 4).text())
                points.append(TiePoint(pid, x1, y1, x2, y2))
            except (ValueError, AttributeError):
                continue
        return points

    def _run(self):
        if self._L1 is None or self._L2 is None:
            QMessageBox.warning(self, "缺少参数", "请先导入两张影像的 DLT L 参数。")
            return

        tie_points = self._get_tie_points()
        if not tie_points:
            QMessageBox.warning(self, "无数据", "请添加同名像点。")
            return

        try:
            # Use distortion-corrected forward intersection if distortion data available
            if (self._dist1 is not None and self._dist2 is not None
                    and self._intr1 is not None and self._intr2 is not None):
                has_dist = any([
                    self._dist1.K1, self._dist1.K2, self._dist1.K3,
                    self._dist1.P1, self._dist1.P2,
                    self._dist2.K1, self._dist2.K2, self._dist2.K3,
                    self._dist2.P1, self._dist2.P2,
                ])
                if has_dist:
                    self._result = forward_intersection_dlt_distorted(
                        tie_points, self._L1, self._L2,
                        self._dist1, self._dist2,
                        self._intr1, self._intr2,
                    )
                else:
                    self._result = forward_intersection_dlt(tie_points, self._L1, self._L2)
            else:
                self._result = forward_intersection_dlt(tie_points, self._L1, self._L2)
            self._display_result()
        except Exception as e:
            self._result_text.setPlainText(f"解算失败: {e}")

    def _display_result(self):
        r = self._result
        if r is None:
            return

        # Determine method used
        has_dist = (self._dist1 is not None and self._dist2 is not None
                    and any([self._dist1.K1, self._dist1.K2, self._dist1.K3,
                             self._dist1.P1, self._dist1.P2,
                             self._dist2.K1, self._dist2.K2, self._dist2.K3,
                             self._dist2.P1, self._dist2.P2]))
        method = "DLT+畸变改正+迭代" if has_dist else "DLT"
        lines = [f"=== 前方交会结果 ({method}) ===", ""]
        lines.append(f"单位权中误差 σ₀: {r.sigma0:.6f} mm")
        lines.append("")

        lines.append("--- 物方坐标 ---")
        lines.append(f"{'点号':>6s}  {'X':>12s}  {'Y':>12s}  {'Z':>12s}  {'交会角':>8s}")
        for i in range(len(r.point_ids)):
            x, y, z = r.coordinates[i]
            angle = r.intersection_angles[i]
            lines.append(
                f"{r.point_ids[i]:>6s}  {x:12.4f}  {y:12.4f}  {z:12.4f}  {angle:8.2f}°"
            )
        lines.append("")

        lines.append("--- 残差 (mm) ---")
        lines.append(f"{'点号':>6s}  {'vx1':>10s}  {'vy1':>10s}  {'vx2':>10s}  {'vy2':>10s}")
        for i, (vx1, vy1, vx2, vy2) in enumerate(r.residuals):
            lines.append(
                f"{r.point_ids[i]:>6s}  {vx1:10.6f}  {vy1:10.6f}  {vx2:10.6f}  {vy2:10.6f}"
            )

        self._result_text.setPlainText("\n".join(lines))

        # Update distance combo boxes
        self._dist_p1.clear()
        self._dist_p2.clear()
        for pid in r.point_ids:
            self._dist_p1.addItem(pid)
            self._dist_p2.addItem(pid)
        if len(r.point_ids) >= 2:
            self._dist_p2.setCurrentIndex(1)

    def _calc_distance(self):
        if self._result is None:
            return
        idx1 = self._dist_p1.currentIndex()
        idx2 = self._dist_p2.currentIndex()
        if idx1 < 0 or idx2 < 0 or idx1 == idx2:
            self._dist_result.setText("请选择两个不同的点")
            return
        d = compute_point_distance(self._result.coordinates, idx1, idx2)
        p1 = self._result.point_ids[idx1]
        p2 = self._result.point_ids[idx2]
        self._dist_result.setText(f"{p1} → {p2}: {d:.4f} mm")

    def _export(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出前方交会结果", "forward_intersection.json",
            "JSON文件 (*.json)"
        )
        if not path:
            return

        r = self._result
        data = {
            "sigma0_mm": r.sigma0,
            "points": [],
        }
        for i in range(len(r.point_ids)):
            vx1, vy1, vx2, vy2 = r.residuals[i]
            data["points"].append({
                "point_id": r.point_ids[i],
                "X": float(r.coordinates[i, 0]),
                "Y": float(r.coordinates[i, 1]),
                "Z": float(r.coordinates[i, 2]),
                "intersection_angle_deg": r.intersection_angles[i],
                "residuals": {"vx1": vx1, "vy1": vy1, "vx2": vx2, "vy2": vy2},
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
