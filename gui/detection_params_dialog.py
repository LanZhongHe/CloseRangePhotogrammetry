"""Dialog for configuring detection parameters."""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QGroupBox, QFormLayout,
    QDoubleSpinBox, QSpinBox, QPushButton, QDialogButtonBox,
)


class DetectionParamsDialog(QDialog):
    """Dialog for adjusting control point detection parameters."""

    def __init__(
        self,
        target_size: int = 100,
        circularity_min: float = 0.65,
        area_tolerance: float = 0.5,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("检测参数设置")
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)

        param_group = QGroupBox("检测参数")
        param_form = QFormLayout(param_group)

        self.spin_target_size = QSpinBox()
        self.spin_target_size.setRange(50, 1000)
        self.spin_target_size.setValue(target_size)
        self.spin_target_size.setSuffix(" px")
        param_form.addRow("目标大小:", self.spin_target_size)

        self.spin_circ = QDoubleSpinBox()
        self.spin_circ.setRange(0.1, 1.0)
        self.spin_circ.setSingleStep(0.05)
        self.spin_circ.setValue(circularity_min)
        param_form.addRow("最小圆度:", self.spin_circ)

        self.spin_area_tol = QDoubleSpinBox()
        self.spin_area_tol.setRange(0.1, 1.0)
        self.spin_area_tol.setSingleStep(0.05)
        self.spin_area_tol.setValue(area_tolerance)
        param_form.addRow("面积容差:", self.spin_area_tol)

        layout.addWidget(param_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self) -> dict:
        """Return current parameter values."""
        return {
            "target_size": self.spin_target_size.value(),
            "circularity_min": self.spin_circ.value(),
            "area_tolerance": self.spin_area_tol.value(),
        }
