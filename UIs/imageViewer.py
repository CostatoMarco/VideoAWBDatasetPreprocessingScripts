from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2

class FullImageWindow(QDialog):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("Full Image Viewer")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        scroll = QScrollArea()
        layout.addWidget(scroll)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image_label)

        # Load full-size image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.image_label.setPixmap(pixmap)