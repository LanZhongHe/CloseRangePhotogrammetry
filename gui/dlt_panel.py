"""GUI panel for configuring and running DLT."""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QCheckBox, QTableWidget, QTableWidgetItem,
    QTextEdit, QFileDialog, QHeaderView,
)
from PyQt5.QtCore import Qt

from src.matching import MatchedPoint
from src.camera_model import SolveConfig, pixel_to_image_coords, CameraIntrinsics
from src.dlt import dlt_solve, dlt_with_distortion, DLTResult


class DLTPanel(QWidget):
    """Panel for configuring and running Direct Linear Transform."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._matched_points: list[MatchedPoint] = []
        self._result: DLTResult | None = None
        self._init_ui()

    def set_matched_points(self, points: list[MatchedPoint]):
        """Set the matched points to use for DLT."""
        self._matched_points = points
        self._update_info()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Info
        self._info_label = QLabel()
        layout.addWidget(self._info_label)

        # Distortion options
        dist_group = QGroupBox("畸变校正")
        dist_layout = QVBoxLayout(dist_group)

        self._use_distortion = QCheckBox("启用畸变校正的迭代 DLT")
        self._use_distortion.toggled.connect(self._update_info)
        dist_layout.addWidget(self._use_distortion)

        check_row = QHBoxLayout()
        self._solve_k1 = QCheckBox("K1"); self._solve_k1.setChecked(True)
        self._solve_k2 = QCheckBox("K2")
        self._solve_p1 = QCheckBox("P1")
        self._solve_p2 = QCheckBox("P2")
        for cb in [self._solve_k1, self._solve_k2, self._solve_p1, self._solve_p2]:
            check_row.addWidget(cb)
        dist_layout.addLayout(check_row)

        layout.addWidget(dist_group)

        # Run button
        run_row = QHBoxLayout()
        self._run_btn = QPushButton("运行 DLT")
        self._run_btn.clicked.connect(self._run)
        run_row.addWidget(self._run_btn)
        self._export_btn = QPushButton("导出结果")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export)
        run_row.addWidget(self._export_btn)
        layout.addLayout(run_row)

        # Results
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        layout.addWidget(self._result_text)

        # Residuals table
        self._residual_table = QTableWidget(0, 3)
        self._residual_table.setHorizontalHeaderLabels(["点号", "vx (mm)", "vy (mm)"])
        self._residual_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(QLabel("残差表:"))
        layout.addWidget(self._residual_table)

    def _get_solve_config(self) -> SolveConfig:
        return SolveConfig(
            solve_k1=self._solve_k1.isChecked(),
            solve_k2=self._solve_k2.isChecked(),
            solve_p1=self._solve_p1.isChecked(),
            solve_p2=self._solve_p2.isChecked(),
        )

    def _update_info(self):
        n_ctrl = sum(1 for p in self._matched_points if not p.is_check)
        n_check = sum(1 for p in self._matched_points if p.is_check)
        min_pts = 6
        self._info_label.setText(
            f"控制点 {n_ctrl} / 检查点 {n_check} / 最少需要 {min_pts} / "
            f"未知数 11 / 冗余度 {n_ctrl * 2 - 11}"
        )

    def _run(self):
        if not self._matched_points:
            self._result_text.setPlainText("错误：未有匹配点，请先完成像点匹配。")
            return

        try:
            if self._use_distortion.isChecked():
                config = self._get_solve_config()
                self._result = dlt_with_distortion(self._matched_points, config)
            else:
                self._result = dlt_solve(self._matched_points)
            self._display_result()
        except Exception as e:
            self._result_text.setPlainText(f"解算失败: {e}")

    def _display_result(self):
        r = self._result
        if r is None:
            return

        lines = ["=== DLT 解算结果 ===", ""]
        lines.append(f"迭代次数: {r.num_iterations}")
        lines.append(f"单位权中误差 σ₀: {r.sigma0:.6f} mm")
        lines.append("")

        lines.append("--- 11 个 L 参数 ---")
        for i in range(11):
            std_str = ""
            if r.param_std and f"L{i+1}" in r.param_std:
                std_str = f"  ± {r.param_std[f'L{i+1}']:.6f}"
            lines.append(f"  L{i+1:2d} = {r.L_params[i]:14.8f}{std_str}")
        lines.append("")

        lines.append("--- 反求内参数 ---")
        lines.append(f"  f  = {r.intrinsics.f:.4f} mm")
        lines.append(f"  x0 = {r.intrinsics.x0:.4f} mm")
        lines.append(f"  y0 = {r.intrinsics.y0:.4f} mm")
        lines.append("")

        lines.append("--- 反求外方位元素 ---")
        ext = r.exterior
        lines.append(f"  Xs = {ext.Xs:.4f} mm")
        lines.append(f"  Ys = {ext.Ys:.4f} mm")
        lines.append(f"  Zs = {ext.Zs:.4f} mm")
        lines.append(f"  ω  = {ext.omega:.6f} rad  ({np.degrees(ext.omega):.4f}°)")
        lines.append(f"  φ  = {ext.phi:.6f} rad  ({np.degrees(ext.phi):.4f}°)")
        lines.append(f"  κ  = {ext.kappa:.6f} rad  ({np.degrees(ext.kappa):.4f}°)")
        lines.append("")

        d = r.distortion
        if any([d.K1, d.K2, d.P1, d.P2]):
            lines.append("--- 畸变系数 ---")
            for name, val in [("K1", d.K1), ("K2", d.K2), ("P1", d.P1), ("P2", d.P2)]:
                if val != 0:
                    lines.append(f"  {name} = {val:.10e}")
            lines.append("")

        # Check point accuracy
        if r.check_point_ids:
            lines.append("--- 检查点精度 ---")
            lines.append(f"  检查点数量: {len(r.check_point_ids)}")
            lines.append(f"  检查点 σ₀: {r.check_point_sigma0:.6f} mm")
            lines.append(f"{'点号':>6s}  {'vx (mm)':>12s}  {'vy (mm)':>12s}")
            for cid, (vx, vy) in zip(r.check_point_ids, r.check_point_residuals):
                lines.append(f"{cid:>6s}  {vx:12.6f}  {vy:12.6f}")
            lines.append("")

        self._result_text.setPlainText("\n".join(lines))

        # Fill residuals table (control points only)
        control_pts = [p for p in self._matched_points if not p.is_check]
        self._residual_table.setRowCount(len(r.residuals))
        for i, (vx, vy) in enumerate(r.residuals):
            cid = control_pts[i].control_id if i < len(control_pts) else ""
            self._residual_table.setItem(i, 0, QTableWidgetItem(cid))
            self._residual_table.setItem(i, 1, QTableWidgetItem(f"{vx:.6f}"))
            self._residual_table.setItem(i, 2, QTableWidgetItem(f"{vy:.6f}"))

        self._export_btn.setEnabled(True)

    def _export(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 DLT 结果", "dlt_result.json", "JSON Files (*.json)"
        )
        if not path:
            return

        import json
        r = self._result
        control_pts = [p for p in self._matched_points if not p.is_check]
        data = {
            "L_params": r.L_params.tolist(),
            "sigma0_mm": r.sigma0,
            "num_iterations": r.num_iterations,
            "intrinsics": {
                "f": r.intrinsics.f, "x0": r.intrinsics.x0, "y0": r.intrinsics.y0,
            },
            "exterior_orientation": {
                "Xs": r.exterior.Xs, "Ys": r.exterior.Ys, "Zs": r.exterior.Zs,
                "omega_rad": r.exterior.omega, "phi_rad": r.exterior.phi, "kappa_rad": r.exterior.kappa,
                "omega_deg": np.degrees(r.exterior.omega),
                "phi_deg": np.degrees(r.exterior.phi),
                "kappa_deg": np.degrees(r.exterior.kappa),
            },
            "distortion": {
                "K1": r.distortion.K1, "K2": r.distortion.K2,
                "P1": r.distortion.P1, "P2": r.distortion.P2,
            },
            "param_std": r.param_std,
            "residuals": [
                {"control_id": control_pts[i].control_id, "vx": vx, "vy": vy}
                for i, (vx, vy) in enumerate(r.residuals)
            ],
            "check_points": {
                "sigma0_mm": r.check_point_sigma0,
                "residuals": [
                    {"control_id": cid, "vx": vx, "vy": vy}
                    for cid, (vx, vy) in zip(r.check_point_ids, r.check_point_residuals)
                ],
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
