"""GUI panel for configuring and running space resection."""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QDoubleSpinBox, QCheckBox, QTableWidget,
    QTableWidgetItem, QTextEdit, QSpinBox, QFileDialog,
    QHeaderView, QComboBox,
)
from PyQt5.QtCore import Qt

from src.camera_model import (
    CameraIntrinsics, ExteriorOrientation, DistortionCoefficients,
    SolveConfig,
)
from src.matching import MatchedPoint
from src.resection import space_resection, ResectionResult


class ResectionPanel(QWidget):
    """Panel for configuring and running single-image space resection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._matched_points: list[MatchedPoint] = []
        self._result: ResectionResult | None = None
        self._init_ui()

    def set_matched_points(self, points: list[MatchedPoint]):
        """Set the matched points to use for resection."""
        self._matched_points = points
        self._update_info()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Camera intrinsics group
        intr_group = QGroupBox("相机内参数")
        intr_layout = QVBoxLayout(intr_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("主距 f (mm):"))
        self._f_spin = QDoubleSpinBox()
        self._f_spin.setRange(1.0, 1000.0)
        self._f_spin.setValue(50.0)
        self._f_spin.setDecimals(3)
        row1.addWidget(self._f_spin)
        self._solve_f = QCheckBox("求解")
        self._solve_f.toggled.connect(self._update_info)
        row1.addWidget(self._solve_f)
        intr_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("传感器宽 (mm):"))
        self._sw_spin = QDoubleSpinBox()
        self._sw_spin.setRange(1.0, 100.0)
        self._sw_spin.setValue(36.0)
        self._sw_spin.setDecimals(2)
        row2.addWidget(self._sw_spin)
        row2.addWidget(QLabel("传感器高 (mm):"))
        self._sh_spin = QDoubleSpinBox()
        self._sh_spin.setRange(1.0, 100.0)
        self._sh_spin.setValue(24.0)
        self._sh_spin.setDecimals(2)
        row2.addWidget(self._sh_spin)
        intr_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("影像宽 (px):"))
        self._iw_spin = QSpinBox()
        self._iw_spin.setRange(100, 50000)
        self._iw_spin.setValue(8256)
        row3.addWidget(self._iw_spin)
        row3.addWidget(QLabel("影像高 (px):"))
        self._ih_spin = QSpinBox()
        self._ih_spin.setRange(100, 50000)
        self._ih_spin.setValue(5504)
        row3.addWidget(self._ih_spin)
        intr_layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("x0 (mm):"))
        self._x0_spin = QDoubleSpinBox()
        self._x0_spin.setRange(-100.0, 100.0)
        self._x0_spin.setValue(0.0)
        self._x0_spin.setDecimals(4)
        row4.addWidget(self._x0_spin)
        self._solve_x0 = QCheckBox("求解")
        self._solve_x0.toggled.connect(self._update_info)
        row4.addWidget(self._solve_x0)
        row4.addWidget(QLabel("y0 (mm):"))
        self._y0_spin = QDoubleSpinBox()
        self._y0_spin.setRange(-100.0, 100.0)
        self._y0_spin.setValue(0.0)
        self._y0_spin.setDecimals(4)
        row4.addWidget(self._y0_spin)
        self._solve_y0 = QCheckBox("求解")
        self._solve_y0.toggled.connect(self._update_info)
        row4.addWidget(self._solve_y0)
        intr_layout.addLayout(row4)

        layout.addWidget(intr_group)

        # Distortion options group
        dist_group = QGroupBox("畸变参数")
        dist_layout = QVBoxLayout(dist_group)

        preset_row = QHBoxLayout()
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(["仅K1", "K1+K2", "径向+偏心", "径向+偏心+薄棱镜", "全部"])
        self._preset_combo.currentTextChanged.connect(self._apply_preset)
        preset_row.addWidget(QLabel("预设:"))
        preset_row.addWidget(self._preset_combo)
        preset_row.addStretch()
        dist_layout.addLayout(preset_row)

        check_row = QHBoxLayout()
        self._solve_k1 = QCheckBox("K1"); self._solve_k1.setChecked(True)
        self._solve_k2 = QCheckBox("K2")
        self._solve_k3 = QCheckBox("K3")
        self._solve_p1 = QCheckBox("P1")
        self._solve_p2 = QCheckBox("P2")
        self._solve_a1 = QCheckBox("A1")
        self._solve_a2 = QCheckBox("A2")
        self._solve_b1 = QCheckBox("B1")
        self._solve_b2 = QCheckBox("B2")
        for cb in [self._solve_k1, self._solve_k2, self._solve_k3,
                   self._solve_p1, self._solve_p2,
                   self._solve_a1, self._solve_a2,
                   self._solve_b1, self._solve_b2]:
            check_row.addWidget(cb)
            cb.toggled.connect(self._update_info)
        dist_layout.addLayout(check_row)

        layout.addWidget(dist_group)

        # Initial values group
        init_group = QGroupBox("外方位元素初始值")
        init_layout = QVBoxLayout(init_group)

        ext_row1 = QHBoxLayout()
        self._xs_spin = self._make_spin(0.0, ext_row1, "Xs:")
        self._ys_spin = self._make_spin(0.0, ext_row1, "Ys:")
        self._zs_spin = self._make_spin(5000.0, ext_row1, "Zs:")
        init_layout.addLayout(ext_row1)

        ext_row2 = QHBoxLayout()
        self._omega_spin = self._make_spin(0.0, ext_row2, "ω:")
        self._phi_spin = self._make_spin(0.0, ext_row2, "φ:")
        self._kappa_spin = self._make_spin(0.0, ext_row2, "κ:")
        init_layout.addLayout(ext_row2)

        auto_btn = QPushButton("自动估计")
        auto_btn.clicked.connect(self._auto_estimate)
        init_layout.addWidget(auto_btn)

        layout.addWidget(init_group)

        # Info label
        self._info_label = QLabel()
        layout.addWidget(self._info_label)

        # Run button
        run_row = QHBoxLayout()
        self._run_btn = QPushButton("运行后方交会")
        self._run_btn.clicked.connect(self._run)
        run_row.addWidget(self._run_btn)
        self._export_btn = QPushButton("导出结果")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export)
        run_row.addWidget(self._export_btn)
        layout.addLayout(run_row)

        # Results area
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        layout.addWidget(self._result_text)

        # Residuals table
        self._residual_table = QTableWidget(0, 3)
        self._residual_table.setHorizontalHeaderLabels(["点号", "vx (mm)", "vy (mm)"])
        self._residual_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(QLabel("残差表:"))
        layout.addWidget(self._residual_table)

    def _make_spin(self, default, layout, label):
        layout.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(-1e6, 1e6)
        spin.setValue(default)
        spin.setDecimals(4)
        layout.addWidget(spin)
        return spin

    def _apply_preset(self, text):
        presets = {
            "仅K1": (True, False, False, False, False, False, False, False, False),
            "K1+K2": (True, True, False, False, False, False, False, False, False),
            "径向+偏心": (True, True, False, True, True, False, False, False, False),
            "径向+偏心+薄棱镜": (True, True, False, True, True, True, True, True, True),
            "全部": (True, True, True, True, True, True, True, True, True),
        }
        vals = presets.get(text)
        if vals:
            for cb, v in zip(
                [self._solve_k1, self._solve_k2, self._solve_k3,
                 self._solve_p1, self._solve_p2,
                 self._solve_a1, self._solve_a2,
                 self._solve_b1, self._solve_b2],
                vals
            ):
                cb.setChecked(v)

    def _auto_estimate(self):
        if not self._matched_points:
            return
        from src.resection import _estimate_initial_rotation
        X = [p.obj_x for p in self._matched_points]
        Y = [p.obj_y for p in self._matched_points]
        Z = [p.obj_z for p in self._matched_points]
        xs = np.mean(X)
        ys = np.mean(Y)
        zs = np.mean(Z) + max(np.ptp(Z) * 1.5, 3000)
        self._xs_spin.setValue(xs)
        self._ys_spin.setValue(ys)
        self._zs_spin.setValue(zs)
        omega, phi, kappa = _estimate_initial_rotation(xs, ys, zs, self._matched_points)
        self._omega_spin.setValue(omega)
        self._phi_spin.setValue(phi)
        self._kappa_spin.setValue(kappa)

    def _get_solve_config(self) -> SolveConfig:
        return SolveConfig(
            solve_f=self._solve_f.isChecked(),
            solve_x0=self._solve_x0.isChecked(),
            solve_y0=self._solve_y0.isChecked(),
            solve_k1=self._solve_k1.isChecked(),
            solve_k2=self._solve_k2.isChecked(),
            solve_k3=self._solve_k3.isChecked(),
            solve_p1=self._solve_p1.isChecked(),
            solve_p2=self._solve_p2.isChecked(),
            solve_a1=self._solve_a1.isChecked(),
            solve_a2=self._solve_a2.isChecked(),
            solve_b1=self._solve_b1.isChecked(),
            solve_b2=self._solve_b2.isChecked(),
        )

    def _get_intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics(
            f=self._f_spin.value(),
            x0=self._x0_spin.value(),
            y0=self._y0_spin.value(),
            sensor_width=self._sw_spin.value(),
            sensor_height=self._sh_spin.value(),
            img_width=self._iw_spin.value(),
            img_height=self._ih_spin.value(),
        )

    def _update_info(self):
        config = self._get_solve_config()
        n = len(self._matched_points)
        self._info_label.setText(
            f"已匹配 {n} 点 / 最少需要 {config.min_points} / "
            f"未知数 {config.num_unknowns} / 冗余度 {n * 2 - config.num_unknowns}"
        )

    def _run(self):
        if not self._matched_points:
            self._result_text.setPlainText("错误：未有匹配点，请先完成像点匹配。")
            return

        config = self._get_solve_config()
        intrinsics = self._get_intrinsics()
        initial_ext = ExteriorOrientation(
            Xs=self._xs_spin.value(),
            Ys=self._ys_spin.value(),
            Zs=self._zs_spin.value(),
            omega=self._omega_spin.value(),
            phi=self._phi_spin.value(),
            kappa=self._kappa_spin.value(),
        )

        try:
            self._result = space_resection(
                self._matched_points, intrinsics, config,
                initial_exterior=initial_ext,
            )
            self._display_result()
        except Exception as e:
            self._result_text.setPlainText(f"解算失败: {e}")

    def _display_result(self):
        r = self._result
        if r is None:
            return

        lines = ["=== 空间后方交会结果 ===", ""]
        lines.append(f"迭代次数: {r.num_iterations}")
        lines.append(f"收敛: {'是' if r.converged else '否'}")
        lines.append(f"单位权中误差 σ₀: {r.sigma0:.6f} mm  ({r.sigma0 / r.intrinsics.pixel_size:.2f} 像素)")
        lines.append("")

        lines.append("--- 外方位元素 ---")
        ext = r.exterior
        lines.append(f"  Xs = {ext.Xs:.4f} mm")
        lines.append(f"  Ys = {ext.Ys:.4f} mm")
        lines.append(f"  Zs = {ext.Zs:.4f} mm")
        lines.append(f"  ω  = {ext.omega:.6f} rad  ({np.degrees(ext.omega):.4f}°)")
        lines.append(f"  φ  = {ext.phi:.6f} rad  ({np.degrees(ext.phi):.4f}°)")
        lines.append(f"  κ  = {ext.kappa:.6f} rad  ({np.degrees(ext.kappa):.4f}°)")
        lines.append("")

        lines.append("--- 内参数 ---")
        lines.append(f"  f  = {r.intrinsics.f:.4f} mm")
        lines.append(f"  x0 = {r.intrinsics.x0:.4f} mm")
        lines.append(f"  y0 = {r.intrinsics.y0:.4f} mm")
        lines.append("")

        lines.append("--- 畸变系数 ---")
        d = r.distortion
        for name, val in [("K1", d.K1), ("K2", d.K2), ("K3", d.K3),
                          ("P1", d.P1), ("P2", d.P2),
                          ("A1", d.A1), ("A2", d.A2), ("B1", d.B1), ("B2", d.B2)]:
            if val != 0.0:
                lines.append(f"  {name} = {val:.10e}")
        lines.append("")

        if r.param_std:
            lines.append("--- 参数中误差 ---")
            for name, std in r.param_std.items():
                lines.append(f"  σ({name}) = {std:.6f}")
            lines.append("")

        self._result_text.setPlainText("\n".join(lines))

        # Fill residuals table
        self._residual_table.setRowCount(len(r.residuals))
        for i, (vx, vy) in enumerate(r.residuals):
            cid = self._matched_points[i].control_id if i < len(self._matched_points) else ""
            self._residual_table.setItem(i, 0, QTableWidgetItem(cid))
            self._residual_table.setItem(i, 1, QTableWidgetItem(f"{vx:.6f}"))
            self._residual_table.setItem(i, 2, QTableWidgetItem(f"{vy:.6f}"))

        self._export_btn.setEnabled(True)

    def _export(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出后方交会结果", "resection_result.json", "JSON Files (*.json)"
        )
        if not path:
            return

        import json
        r = self._result
        data = {
            "converged": r.converged,
            "num_iterations": r.num_iterations,
            "sigma0_mm": r.sigma0,
            "sigma0_px": r.sigma0 / r.intrinsics.pixel_size,
            "exterior_orientation": {
                "Xs": r.exterior.Xs, "Ys": r.exterior.Ys, "Zs": r.exterior.Zs,
                "omega_rad": r.exterior.omega, "phi_rad": r.exterior.phi, "kappa_rad": r.exterior.kappa,
                "omega_deg": np.degrees(r.exterior.omega),
                "phi_deg": np.degrees(r.exterior.phi),
                "kappa_deg": np.degrees(r.exterior.kappa),
            },
            "intrinsics": {
                "f": r.intrinsics.f, "x0": r.intrinsics.x0, "y0": r.intrinsics.y0,
                "pixel_size": r.intrinsics.pixel_size,
            },
            "distortion": {
                "K1": r.distortion.K1, "K2": r.distortion.K2, "K3": r.distortion.K3,
                "P1": r.distortion.P1, "P2": r.distortion.P2,
                "A1": r.distortion.A1, "A2": r.distortion.A2,
                "B1": r.distortion.B1, "B2": r.distortion.B2,
            },
            "param_std": r.param_std,
            "residuals": [
                {"control_id": self._matched_points[i].control_id, "vx": vx, "vy": vy}
                for i, (vx, vy) in enumerate(r.residuals)
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
