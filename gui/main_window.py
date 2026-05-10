"""Main GUI window for control point detection system.

PyQt5-based interface with:
  - File list panel (left)
  - Large image viewer with zoom/pan + click-to-select (center)
  - Detection parameter panel + point info panel (right)
  - Batch processing with progress bar
  - Detection overlay (green markers + ID labels)
  - Click to select/delete/add targets
"""

import os
import time
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsTextItem,
    QGroupBox, QFormLayout, QDoubleSpinBox,
    QSpinBox, QPushButton, QLabel, QFileDialog, QStatusBar,
    QAction, QProgressBar, QMessageBox, QSplitter,
    QDialog, QLineEdit, QDialogButtonBox, QInputDialog,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPen, QColor, QFont, QPainter

from src.preprocessing import preprocess, PreprocessingParams
from src.detection import detect_candidates, DetectionParams
from src.subpixel import localize_target, centroid_refine
from src.id_recognition import assign_sequential_ids
from src.data_model import DetectionResult, TargetPoint
from src.io_utils import save_results
from src.coord_io import load_control_field
from src.matching import MatchedPoint
from src.camera_model import CameraIntrinsics, SolveConfig
from gui.matching_dialog import MatchingDialog
from gui.resection_panel import ResectionPanel
from gui.dlt_panel import DLTPanel


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Detection worker thread
# ---------------------------------------------------------------------------

class DetectionWorker(QThread):
    """Runs detection on images in a background thread."""
    progress = pyqtSignal(int, int)
    image_done = pyqtSignal(int, object)
    finished_all = pyqtSignal()

    def __init__(self, image_paths: List[str], params: DetectionParams, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.params = params
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        prep_params = PreprocessingParams(target_size_px=self.params.target_size_px)

        for i, path in enumerate(self.image_paths):
            if self._cancelled:
                break
            result = self._detect_one(path, prep_params)
            self.image_done.emit(i, result)
            self.progress.emit(i + 1, len(self.image_paths))

        self.finished_all.emit()

    def _detect_one(self, path, prep_params) -> DetectionResult:
        image = cv2.imread(path)
        if image is None:
            return DetectionResult(
                image_path=path, image_width=0, image_height=0, targets=[]
            )

        h, w = image.shape[:2]
        gray, binary_inv = preprocess(image, prep_params)
        candidates = detect_candidates(binary_inv, self.params)

        targets = []
        for cand in candidates:
            tp = localize_target(gray, cand)
            targets.append(tp)

        assign_sequential_ids(targets)

        return DetectionResult(
            image_path=path,
            image_width=w,
            image_height=h,
            targets=targets,
        )


# ---------------------------------------------------------------------------
# Image viewer with zoom/pan + click-to-select
# ---------------------------------------------------------------------------

class ImageViewer(QGraphicsView):
    """QGraphicsView with mouse-wheel zoom, middle-button pan,
    and left-click to select nearest marker."""

    left_clicked = pyqtSignal(float, float)  # scene coordinates

    MARKER_RADIUS = 15
    SELECT_DIST = 30  # max pixel distance to select a marker

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._panning = False
        self._pan_start = QPointF()

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._overlay_items = []

    def set_image(self, pixmap: QPixmap):
        self._scene.clear()
        self._overlay_items.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def clear_overlays(self):
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()

    def add_marker(self, x: float, y: float, target_id: str, selected: bool = False):
        """Add a detection marker (circle + ID label) to the scene."""
        r = self.MARKER_RADIUS
        color = QColor(255, 255, 0) if selected else QColor(0, 255, 0)
        pen = QPen(color, 2)

        ellipse = self._scene.addEllipse(x - r, y - r, r * 2, r * 2, pen)
        self._overlay_items.append(ellipse)

        pen_cross = QPen(color, 1, Qt.DashLine)
        line_h = self._scene.addLine(x - r * 2, y, x + r * 2, y, pen_cross)
        line_v = self._scene.addLine(x, y - r * 2, x, y + r * 2, pen_cross)
        self._overlay_items.extend([line_h, line_v])

        label = QGraphicsTextItem(target_id)
        label.setDefaultTextColor(color)
        label.setPos(x + r + 2, y - r - 5)
        label.setFont(QFont("Consolas", 12, QFont.Bold))
        self._scene.addItem(label)
        self._overlay_items.append(label)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.left_clicked.emit(scene_pos.x(), scene_pos.y())
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        """Forward arrow keys to the main window for point nudging."""
        if event.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            QApplication.sendEvent(self.window(), event)
        else:
            super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# ID correction dialog
# ---------------------------------------------------------------------------

class IdCorrectionDialog(QDialog):

    def __init__(self, current_id: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Correct ID")
        layout = QFormLayout(self)

        self.id_input = QLineEdit(current_id)
        self.id_input.setMaxLength(3)
        self.id_input.setPlaceholderText("001-999")
        layout.addRow("New ID:", self.id_input)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_id(self) -> str:
        return self.id_input.text().strip().zfill(3)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Control Point Detection \u2014 Close Range Photogrammetry")
        self.resize(1600, 900)

        self._image_paths: List[str] = []
        self._results: dict[str, DetectionResult] = {}
        self._current_index = -1
        self._selected_index: int = -1   # index into current result.targets
        self._worker: Optional[DetectionWorker] = None
        self._adding_mode: bool = False  # True = waiting for user click to add
        self._control_field: dict[str, tuple[float, float, float]] = {}
        self._matched_points: list[MatchedPoint] = []
        self._current_image: np.ndarray | None = None  # cached for matching preview

        self._init_ui()
        self._init_menu()

    # --- UI setup ---

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        splitter = QSplitter(Qt.Horizontal, central)

        # Left: file list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)

        btn_open = QPushButton("Open Folder...")
        btn_open.clicked.connect(self._open_folder)
        left_layout.addWidget(btn_open)

        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        left_layout.addWidget(self.file_list)

        btn_detect_all = QPushButton("Detect All")
        btn_detect_all.clicked.connect(self._detect_all)
        left_layout.addWidget(btn_detect_all)

        btn_detect_current = QPushButton("Detect Current")
        btn_detect_current.clicked.connect(self._detect_current)
        left_layout.addWidget(btn_detect_current)

        btn_export = QPushButton("Export JSON...")
        btn_export.clicked.connect(self._export_json)
        left_layout.addWidget(btn_export)

        splitter.addWidget(left_widget)

        # Center: image viewer
        self.viewer = ImageViewer()
        self.viewer.left_clicked.connect(self._on_left_click)
        splitter.addWidget(self.viewer)

        # Right: parameter + info panels
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # Parameter group
        param_group = QGroupBox("Detection Parameters")
        param_form = QFormLayout(param_group)

        self.spin_target_size = QSpinBox()
        self.spin_target_size.setRange(50, 1000)
        self.spin_target_size.setValue(100)
        self.spin_target_size.setSuffix(" px")
        param_form.addRow("Target Size:", self.spin_target_size)

        self.spin_circ = QDoubleSpinBox()
        self.spin_circ.setRange(0.1, 1.0)
        self.spin_circ.setSingleStep(0.05)
        self.spin_circ.setValue(0.65)
        param_form.addRow("Circularity Min:", self.spin_circ)

        self.spin_area_tol = QDoubleSpinBox()
        self.spin_area_tol.setRange(0.1, 1.0)
        self.spin_area_tol.setSingleStep(0.05)
        self.spin_area_tol.setValue(0.5)
        param_form.addRow("Area Tolerance:", self.spin_area_tol)

        btn_reapply = QPushButton("Re-apply Parameters")
        btn_reapply.clicked.connect(self._detect_current)
        param_form.addRow(btn_reapply)

        right_layout.addWidget(param_group)

        # Selected point info
        info_group = QGroupBox("Selected Point")
        info_form = QFormLayout(info_group)

        self.lbl_id = QLabel("\u2014")
        info_form.addRow("ID:", self.lbl_id)

        self.lbl_coord = QLabel("\u2014")
        info_form.addRow("Pixel Coord:", self.lbl_coord)

        self.lbl_conf = QLabel("\u2014")
        info_form.addRow("Confidence:", self.lbl_conf)

        self.lbl_method = QLabel("\u2014")
        info_form.addRow("Method:", self.lbl_method)

        btn_correct = QPushButton("Correct ID...")
        btn_correct.clicked.connect(self._correct_id)
        info_form.addRow(btn_correct)

        btn_delete = QPushButton("Delete Point")
        btn_delete.clicked.connect(self._delete_point)
        info_form.addRow(btn_delete)

        btn_add = QPushButton("Add Point (click on image)")
        btn_add.clicked.connect(self._start_add_mode)
        info_form.addRow(btn_add)

        self.lbl_mode = QLabel("")
        self.lbl_mode.setStyleSheet("color: orange; font-weight: bold;")
        info_form.addRow(self.lbl_mode)

        right_layout.addWidget(info_group)
        right_layout.addStretch()

        splitter.addWidget(right_widget)
        splitter.setSizes([200, 1000, 280])

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(splitter)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(300)
        self.progress_bar.setVisible(False)
        self.status.addPermanentWidget(self.progress_bar)

    def _init_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        act_open = QAction("Open Folder...", self)
        act_open.triggered.connect(self._open_folder)
        file_menu.addAction(act_open)
        act_export = QAction("Export JSON...", self)
        act_export.triggered.connect(self._export_json)
        file_menu.addAction(act_export)

        det_menu = menubar.addMenu("Detection")
        act_detect_all = QAction("Detect All Images", self)
        act_detect_all.triggered.connect(self._detect_all)
        det_menu.addAction(act_detect_all)
        act_detect_cur = QAction("Detect Current Image", self)
        act_detect_cur.triggered.connect(self._detect_current)
        det_menu.addAction(act_detect_cur)

        # Photogrammetry menu
        photo_menu = menubar.addMenu("摄影测量")
        act_load_ctrl = QAction("加载控制场坐标...", self)
        act_load_ctrl.triggered.connect(self._load_control_field)
        photo_menu.addAction(act_load_ctrl)

        act_match = QAction("像点匹配...", self)
        act_match.triggered.connect(self._open_matching_dialog)
        photo_menu.addAction(act_match)

        act_resection = QAction("空间后方交会...", self)
        act_resection.triggered.connect(self._open_resection_panel)
        photo_menu.addAction(act_resection)

        act_dlt = QAction("直接线性变换...", self)
        act_dlt.triggered.connect(self._open_dlt_panel)
        photo_menu.addAction(act_dlt)

    # --- Properties ---

    def _get_params(self) -> DetectionParams:
        return DetectionParams(
            target_size_px=self.spin_target_size.value(),
            area_tolerance=self.spin_area_tol.value(),
            circularity_min=self.spin_circ.value(),
        )

    def _current_path(self) -> Optional[str]:
        row = self.file_list.currentRow()
        if 0 <= row < len(self._image_paths):
            return self._image_paths[row]
        return None

    def _current_result(self) -> Optional[DetectionResult]:
        path = self._current_path()
        if path and path in self._results:
            return self._results[path]
        return None

    # --- Click handling ---

    def _on_left_click(self, sx: float, sy: float):
        """Handle left-click on the image viewer in scene coordinates."""
        if self._adding_mode:
            self._add_point_at(sx, sy)
            return

        # Try to find nearest marker within threshold
        result = self._current_result()
        if not result or not result.targets:
            self._update_selection(-1)
            return

        best_idx = -1
        best_dist = ImageViewer.SELECT_DIST
        for i, tp in enumerate(result.targets):
            d = np.hypot(tp.pixel_x - sx, tp.pixel_y - sy)
            if d < best_dist:
                best_dist = d
                best_idx = i

        self._update_selection(best_idx)

    def _update_selection(self, idx: int):
        """Update the selected target index and refresh display."""
        self._selected_index = idx
        result = self._current_result()

        if idx < 0 or result is None or idx >= len(result.targets):
            self.lbl_id.setText("\u2014")
            self.lbl_coord.setText("\u2014")
            self.lbl_conf.setText("\u2014")
            self.lbl_method.setText("\u2014")
        else:
            tp = result.targets[idx]
            self.lbl_id.setText(tp.id)
            self.lbl_coord.setText(f"({tp.pixel_x:.2f}, {tp.pixel_y:.2f})")
            self.lbl_conf.setText(f"{tp.confidence:.4f}")
            self.lbl_method.setText(tp.subpixel_method)

        self._refresh_display()

    # --- Add point ---

    def _start_add_mode(self):
        """Enter add-point mode: next left-click adds a target."""
        path = self._current_path()
        if not path:
            return
        if path not in self._results:
            image = cv2.imread(path)
            if image is None:
                return
            h, w = image.shape[:2]
            self._results[path] = DetectionResult(
                image_path=path, image_width=w, image_height=h, targets=[]
            )

        self._adding_mode = True
        self.lbl_mode.setText("[ADD MODE] Click on image to add point")
        self.status.showMessage("Click on the image to add a new control point")

    def _add_point_at(self, sx: float, sy: float):
        """Add a new target at the clicked scene position."""
        self._adding_mode = False
        self.lbl_mode.setText("")

        result = self._current_result()
        if result is None:
            return

        # Get ID
        id_str, ok = QInputDialog.getText(
            self, "New Point ID", "Enter ID (001-999):", text="001"
        )
        if not ok:
            return
        id_str = id_str.strip().zfill(3)

        # Try to refine center using centroid on the grayscale image
        path = self._current_path()
        image = cv2.imread(path)
        if image is not None:
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            cx_ref, cy_ref = centroid_refine(gray, sx, sy, radius=30)
        else:
            cx_ref, cy_ref = sx, sy

        tp = TargetPoint(
            id=id_str,
            pixel_x=cx_ref,
            pixel_y=cy_ref,
            confidence=1.0,
            source="manual",
            subpixel_method="manual",
        )
        result.targets.append(tp)
        self._update_selection(len(result.targets) - 1)
        self.status.showMessage(f"Added target {id_str} at ({cx_ref:.1f}, {cy_ref:.1f})")

    # --- Correct ID ---

    def _correct_id(self):
        if self._selected_index < 0:
            QMessageBox.information(self, "No Selection", "Click on a target first.")
            return

        result = self._current_result()
        if result is None:
            return
        tp = result.targets[self._selected_index]

        dlg = IdCorrectionDialog(tp.id, self)
        if dlg.exec_() == QDialog.Accepted:
            tp.id = dlg.get_id()
            tp.source = "manual"
            self._update_selection(self._selected_index)

    # --- Delete point ---

    def _delete_point(self):
        if self._selected_index < 0:
            QMessageBox.information(self, "No Selection", "Click on a target first.")
            return

        result = self._current_result()
        if result is None:
            return

        removed = result.targets.pop(self._selected_index)
        self._selected_index = -1
        self._refresh_display()
        self.lbl_id.setText("\u2014")
        self.lbl_coord.setText("\u2014")
        self.lbl_conf.setText("\u2014")
        self.lbl_method.setText("\u2014")
        self.status.showMessage(f"Deleted target {removed.id}")

    # --- Display ---

    def _refresh_display(self):
        """Refresh only the overlay markers without reloading the image (preserves zoom/scroll)."""
        path = self._current_path()
        if not path:
            return
        self.viewer.clear_overlays()
        result = self._results.get(path)
        if result:
            for i, tp in enumerate(result.targets):
                selected = (i == self._selected_index)
                self.viewer.add_marker(tp.pixel_x, tp.pixel_y, tp.id, selected=selected)
        self.status.showMessage(
            f"[{self._current_index + 1}/{len(self._image_paths)}] "
            f"{os.path.basename(path)}  "
            f"Targets: {len(result.targets) if result else 0}"
        )

    def _display_image(self, path: str):
        """Full image reload (resets zoom). Use only on file change."""
        image = cv2.imread(path)
        if image is None:
            return

        self._current_image = image
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.viewer.set_image(pixmap)

        self._refresh_display()

    # --- Photogrammetry ---

    def _load_control_field(self):
        """Load control field coordinates from a text file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择控制场坐标文件", "docs/",
            "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return
        try:
            self._control_field = load_control_field(path)
            self.status.showMessage(
                f"已加载 {len(self._control_field)} 个控制场坐标点"
            )
        except Exception as e:
            QMessageBox.warning(self, "加载失败", str(e))

    def _get_intrinsics(self) -> CameraIntrinsics:
        """Get default camera intrinsics from current image dimensions."""
        result = self._current_result()
        if result:
            return CameraIntrinsics(
                img_width=result.image_width,
                img_height=result.image_height,
            )
        return CameraIntrinsics()

    def _open_matching_dialog(self):
        """Open the matching dialog to pair detected points with control field."""
        result = self._current_result()
        if not result or not result.targets:
            QMessageBox.information(self, "无数据", "请先检测当前影像的控制点。")
            return
        if not self._control_field:
            QMessageBox.information(self, "无控制场", "请先加载控制场坐标文件。")
            return

        intrinsics = self._get_intrinsics()
        solve_config = SolveConfig()  # default config for min_points calculation

        dlg = MatchingDialog(
            detected_points=result.targets,
            control_field=self._control_field,
            image_path=self._current_path() or "",
            image=self._current_image,
            intrinsics=intrinsics,
            solve_config=solve_config,
            parent=self,
        )
        if dlg.exec_() == QDialog.Accepted:
            self._matched_points = dlg.get_matched_points()
            self.status.showMessage(
                f"完成匹配：{len(self._matched_points)} 对像点-物方坐标"
            )

    def _open_resection_panel(self):
        """Open the space resection panel in a dialog."""
        if not self._matched_points:
            QMessageBox.information(
                self, "未匹配", "请先通过「像点匹配」建立像点-物方坐标对应关系。"
            )
            return

        from PyQt5.QtWidgets import QDialog, QVBoxLayout
        dlg = QDialog(self)
        dlg.setWindowTitle("空间后方交会")
        dlg.setMinimumSize(700, 800)
        layout = QVBoxLayout(dlg)
        panel = ResectionPanel(dlg)
        panel.set_matched_points(self._matched_points)
        layout.addWidget(panel)
        dlg.exec_()

    def _open_dlt_panel(self):
        """Open the DLT panel in a dialog."""
        if not self._matched_points:
            QMessageBox.information(
                self, "未匹配", "请先通过「像点匹配」建立像点-物方坐标对应关系。"
            )
            return

        from PyQt5.QtWidgets import QDialog, QVBoxLayout
        dlg = QDialog(self)
        dlg.setWindowTitle("直接线性变换 (DLT)")
        dlg.setMinimumSize(700, 800)
        layout = QVBoxLayout(dlg)
        panel = DLTPanel(dlg)
        panel.set_matched_points(self._matched_points)
        layout.addWidget(panel)
        dlg.exec_()

    # --- File management ---

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return

        self._image_paths = []
        self._results.clear()
        self.file_list.clear()

        for f in sorted(os.listdir(folder)):
            ext = Path(f).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                full = os.path.join(folder, f)
                self._image_paths.append(full)
                self.file_list.addItem(f)

        self.status.showMessage(f"Loaded {len(self._image_paths)} images from {folder}")
        if self._image_paths:
            self.file_list.setCurrentRow(0)

    def _on_file_selected(self, row: int):
        if row < 0 or row >= len(self._image_paths):
            return
        self._current_index = row
        self._selected_index = -1
        path = self._image_paths[row]
        self._display_image(path)

    # --- Detection ---

    def _detect_current(self):
        path = self._current_path()
        if not path:
            return

        params = self._get_params()
        prep = PreprocessingParams(target_size_px=params.target_size_px)

        image = cv2.imread(path)
        if image is None:
            QMessageBox.warning(self, "Error", f"Cannot read image: {path}")
            return

        t0 = time.time()
        h, w = image.shape[:2]
        gray, binary_inv = preprocess(image, prep)
        candidates = detect_candidates(binary_inv, params)

        targets = []
        for cand in candidates:
            tp = localize_target(gray, cand)
            targets.append(tp)

        assign_sequential_ids(targets)

        elapsed = time.time() - t0
        result = DetectionResult(
            image_path=path, image_width=w, image_height=h, targets=targets
        )
        self._results[path] = result
        self._selected_index = -1

        self._display_image(path)
        self.status.showMessage(f"Detected {len(targets)} targets in {elapsed:.2f}s")

    def _detect_all(self):
        if not self._image_paths:
            return

        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "Busy", "Detection is already running.")
            return

        params = self._get_params()
        self._worker = DetectionWorker(self._image_paths, params)
        self._worker.progress.connect(self._on_batch_progress)
        self._worker.image_done.connect(self._on_image_done)
        self._worker.finished_all.connect(self._on_batch_finished)

        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self._image_paths))
        self.progress_bar.setValue(0)
        self._worker.start()

    def _on_batch_progress(self, current: int, total: int):
        self.progress_bar.setValue(current)
        self.status.showMessage(f"Detecting... {current}/{total}")

    def _on_image_done(self, index: int, result: DetectionResult):
        self._results[result.image_path] = result
        if index == self._current_index:
            self._display_image(result.image_path)

    def _on_batch_finished(self):
        self.progress_bar.setVisible(False)
        total_targets = sum(len(r.targets) for r in self._results.values())
        self.status.showMessage(
            f"Batch done. {len(self._results)} images, {total_targets} total targets."
        )
        path = self._current_path()
        if path:
            self._display_image(path)

    # --- Export ---

    def _export_json(self):
        if not self._results:
            QMessageBox.information(self, "No Data", "No detection results to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "detection_results.json",
            "JSON Files (*.json)"
        )
        if not path:
            return

        results_list = list(self._results.values())
        save_results(results_list, path)
        self.status.showMessage(f"Exported {len(results_list)} results to {path}")

    # --- Keyboard nudge ---

    def keyPressEvent(self, event):
        """Arrow keys nudge the selected point by 1 pixel (Shift = 10px)."""
        if self._selected_index < 0:
            super().keyPressEvent(event)
            return

        result = self._current_result()
        if result is None or self._selected_index >= len(result.targets):
            super().keyPressEvent(event)
            return

        step = 0.1 if event.modifiers() & Qt.ShiftModifier else 1.0
        tp = result.targets[self._selected_index]

        key = event.key()
        if key == Qt.Key_Left:
            tp.pixel_x -= step
        elif key == Qt.Key_Right:
            tp.pixel_x += step
        elif key == Qt.Key_Up:
            tp.pixel_y -= step
        elif key == Qt.Key_Down:
            tp.pixel_y += step
        else:
            super().keyPressEvent(event)
            return

        tp.source = "manual"
        self._refresh_display()
        self.lbl_coord.setText(f"({tp.pixel_x:.2f}, {tp.pixel_y:.2f})")
        self.status.showMessage(
            f"Moved {tp.id} to ({tp.pixel_x:.2f}, {tp.pixel_y:.2f})"
        )
