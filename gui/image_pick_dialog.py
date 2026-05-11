"""Dialog for picking a point on an image by clicking."""

import cv2
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QDialogButtonBox,
    QGraphicsView, QGraphicsScene,
)
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPen, QColor, QPainter


class _PickViewer(QGraphicsView):
    """Image viewer with zoom/pan and left-click to pick a point."""

    left_clicked = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._panning = False
        self._pan_start = QPointF()
        self._crosshair_items = []

    def set_image(self, pixmap: QPixmap):
        self._scene.clear()
        self._crosshair_items.clear()
        self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def set_crosshair(self, x: float, y: float):
        for item in self._crosshair_items:
            self._scene.removeItem(item)
        self._crosshair_items.clear()

        pen = QPen(QColor(255, 0, 0), 1.5, Qt.DashLine)
        r = 500
        self._crosshair_items.append(self._scene.addLine(x - r, y, x + r, y, pen))
        self._crosshair_items.append(self._scene.addLine(x, y - r, x, y + r, pen))
        dot = self._scene.addEllipse(
            x - 6, y - 6, 12, 12, QPen(QColor(255, 0, 0), 2)
        )
        self._crosshair_items.append(dot)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            sp = self.mapToScene(event.pos())
            self.left_clicked.emit(sp.x(), sp.y())
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


class ImagePickDialog(QDialog):
    """Dialog to pick a single point on an image by left-clicking.

    Usage:
        dlg = ImagePickDialog("/path/to/image.jpg", "影像1")
        if dlg.exec_() == QDialog.Accepted:
            px, py = dlg.get_pixel_coords()
    """

    def __init__(self, image_path: str, point_label: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"刺点 — {point_label}" if point_label else "刺点")
        self.resize(1000, 800)

        self._pixel_x: float | None = None
        self._pixel_y: float | None = None

        layout = QVBoxLayout(self)

        self._coord_label = QLabel("点击影像选择刺点位置（滚轮缩放，中键平移）")
        layout.addWidget(self._coord_label)

        self._viewer = _PickViewer(self)
        self._viewer.left_clicked.connect(self._on_click)
        layout.addWidget(self._viewer)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # Load image
        image = cv2.imread(image_path)
        if image is not None:
            h, w = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self._viewer.set_image(pixmap)

    def _on_click(self, x: float, y: float):
        self._pixel_x = x
        self._pixel_y = y
        self._viewer.set_crosshair(x, y)
        self._coord_label.setText(f"已选: ({x:.1f}, {y:.1f}) px")

    def get_pixel_coords(self) -> tuple[float, float] | None:
        """Return the picked pixel coordinates, or None if not picked."""
        if self._pixel_x is not None and self._pixel_y is not None:
            return (self._pixel_x, self._pixel_y)
        return None
