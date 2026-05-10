"""Dialog for matching detected image points to control field coordinates."""

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QComboBox, QLabel, QPushButton, QHeaderView, QSplitter, QWidget,
    QMessageBox, QLineEdit,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal

from src.matching import MatchedPoint
from src.camera_model import CameraIntrinsics, pixel_to_image_coords, SolveConfig
from src.data_model import TargetPoint


class MatchingDialog(QDialog):
    """Interactive point matching dialog.

    Left: table of detected points with combo-box for control field ID selection.
    Right: local crop preview of selected detected point.

    Signals:
        match_changed(int, str) — emitted when a match changes: (row_index, control_id)
    """

    match_changed = pyqtSignal(int, str)

    def __init__(
        self,
        detected_points: list[TargetPoint],
        control_field: dict[str, tuple[float, float, float]],
        image_path: str,
        image: np.ndarray | None,
        intrinsics: CameraIntrinsics,
        solve_config: SolveConfig,
        previous_matches: dict[int, str] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("像点匹配")
        self.setMinimumSize(900, 600)

        self._detected = detected_points
        self._control_field = control_field
        self._control_ids = sorted(control_field.keys(), key=lambda x: int(x))
        self._image = image
        self._intrinsics = intrinsics
        self._solve_config = solve_config
        self._previous_matches = previous_matches or {}

        self._combos: list[QComboBox] = []
        self._init_ui()
        self._restore_previous()

    def _init_ui(self):
        layout = QHBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)

        # Left: table
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)

        self._table = QTableWidget(len(self._detected), 6)
        self._table.setHorizontalHeaderLabels([
            "检测ID", "像素X", "像素Y", "控制场点号", "物方X", "物方Y"
        ])
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.currentCellChanged.connect(self._on_row_changed)

        for i, pt in enumerate(self._detected):
            # Detection ID (read-only)
            id_item = QTableWidgetItem(pt.id)
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(i, 0, id_item)

            # Pixel coords (read-only)
            px_item = QTableWidgetItem(f"{pt.pixel_x:.2f}")
            px_item.setFlags(px_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(i, 1, px_item)
            py_item = QTableWidgetItem(f"{pt.pixel_y:.2f}")
            py_item.setFlags(py_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(i, 2, py_item)

            # Control field combo (editable so user can type point ID)
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItem("")  # empty = unmatched
            for cid in self._control_ids:
                combo.addItem(cid)
            combo.currentTextChanged.connect(lambda text, row=i: self._on_combo_changed(row, text))
            self._table.setCellWidget(i, 3, combo)
            self._combos.append(combo)

            # Object coords (read-only, filled when matched)
            objx_item = QTableWidgetItem("")
            objx_item.setFlags(objx_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(i, 4, objx_item)
            objy_item = QTableWidgetItem("")
            objy_item.setFlags(objy_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(i, 5, objy_item)

        table_layout.addWidget(self._table)

        # Bottom buttons
        btn_layout = QHBoxLayout()
        self._clear_btn = QPushButton("清除所有匹配")
        self._clear_btn.clicked.connect(self._clear_all)
        btn_layout.addWidget(self._clear_btn)

        self._status_label = QLabel()
        btn_layout.addWidget(self._status_label)
        btn_layout.addStretch()

        self._ok_btn = QPushButton("确定")
        self._ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self._ok_btn)
        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self._cancel_btn)

        table_layout.addLayout(btn_layout)
        splitter.addWidget(table_widget)

        # Right: preview
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.addWidget(QLabel("选中像点局部预览:"))
        self._preview_label = QLabel()
        self._preview_label.setFixedSize(400, 400)
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setStyleSheet("border: 1px solid gray;")
        preview_layout.addWidget(self._preview_label)
        preview_layout.addStretch()

        self._info_label = QLabel()
        preview_layout.addWidget(self._info_label)

        splitter.addWidget(preview_widget)
        splitter.setSizes([500, 420])

        layout.addWidget(splitter)
        self._update_status()

    def _restore_previous(self):
        """Restore previous matching selections."""
        for row_idx, cid in self._previous_matches.items():
            if 0 <= row_idx < len(self._combos):
                combo = self._combos[row_idx]
                idx = combo.findText(cid)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                else:
                    combo.setEditText(cid)
        self._highlight_duplicates()

    def _on_combo_changed(self, row: int, text: str):
        """Update object coordinates when combo selection changes."""
        if text and text in self._control_field:
            x, y, z = self._control_field[text]
            self._table.item(row, 4).setText(f"{x:.2f}")
            self._table.item(row, 5).setText(f"{y:.2f}")
        else:
            self._table.item(row, 4).setText("")
            self._table.item(row, 5).setText("")

        self._highlight_duplicates()
        self._update_status()
        self.match_changed.emit(row, text)

    def _on_row_changed(self, row: int, col: int, prev_row: int, prev_col: int):
        """Update preview when row selection changes."""
        if row < 0 or row >= len(self._detected):
            return
        self._update_preview(row)

    def _update_preview(self, row: int):
        """Show a local crop of the image around the selected detected point."""
        if self._image is None:
            self._info_label.setText("无影像数据")
            return

        pt = self._detected[row]
        h, w = self._image.shape[:2]
        cx, cy = int(pt.pixel_x), int(pt.pixel_y)

        # Crop 400x400 region
        half = 200
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)

        crop = self._image[y1:y2, x1:x2]
        if crop.size == 0:
            return

        # Draw crosshair at point position
        local_x = cx - x1
        local_y = cy - y1
        if len(crop.shape) == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        cv2.drawMarker(crop, (local_x, local_y), (0, 255, 0),
                       cv2.MARKER_CROSS, 30, 2)

        # Convert to QPixmap
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                       rgb.shape[1] * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._preview_label.setPixmap(pixmap)

        info = f"检测点 {pt.id}  像素({pt.pixel_x:.1f}, {pt.pixel_y:.1f})"
        self._info_label.setText(info)

    def _highlight_duplicates(self):
        """Highlight rows with duplicate control field IDs."""
        selected = {}
        for i, combo in enumerate(self._combos):
            cid = combo.currentText()
            if cid:
                if cid in selected:
                    # Mark both rows as duplicate
                    for r in selected[cid]:
                        for c in range(6):
                            item = self._table.item(r, c)
                            if item:
                                item.setBackground(Qt.red)
                    for c in range(6):
                        item = self._table.item(i, c)
                        if item:
                            item.setBackground(Qt.red)
                    selected[cid].append(i)
                else:
                    selected[cid] = [i]

        # Reset non-duplicate rows
        for i, combo in enumerate(self._combos):
            cid = combo.currentText()
            if not cid or cid not in selected or len(selected[cid]) == 1:
                for c in range(6):
                    item = self._table.item(i, c)
                    if item:
                        if not cid:
                            item.setBackground(Qt.yellow)
                        else:
                            item.setBackground(Qt.white)

    def _clear_all(self):
        """Clear all combo selections."""
        for combo in self._combos:
            combo.setCurrentIndex(0)
        self._highlight_duplicates()

    def _update_status(self):
        """Update status label with match count."""
        matched = sum(1 for c in self._combos if c.currentText())
        self._status_label.setText(f"已匹配 {matched}")
        self._ok_btn.setEnabled(matched >= 2)

    def get_matched_points(self) -> list[MatchedPoint]:
        """Return the list of successfully matched point pairs."""
        result = []
        for i, combo in enumerate(self._combos):
            cid = combo.currentText().strip()
            if not cid or cid not in self._control_field:
                continue
            pt = self._detected[i]
            ox, oy, oz = self._control_field[cid]
            x_mm, y_mm = pixel_to_image_coords(pt.pixel_x, pt.pixel_y, self._intrinsics)
            result.append(MatchedPoint(
                detected_id=pt.id,
                control_id=cid,
                pixel_x=pt.pixel_x, pixel_y=pt.pixel_y,
                image_x_mm=x_mm, image_y_mm=y_mm,
                obj_x=ox, obj_y=oy, obj_z=oz,
                is_manual=pt.source == "manual",
            ))
        return result

    def get_matches_dict(self) -> dict[int, str]:
        """Return current matching state as {row_index: control_id} for persistence."""
        result = {}
        for i, combo in enumerate(self._combos):
            cid = combo.currentText().strip()
            if cid:
                result[i] = cid
        return result
