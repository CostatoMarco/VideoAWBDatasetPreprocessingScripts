# main.py

import sys
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QPushButton, QFileDialog, QScrollArea, QHBoxLayout, QGridLayout, QStackedWidget
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal
from imageViewer import FullImageWindow


class FileTile(QWidget):
    clicked = pyqtSignal(str)  # Emits path to the frames folder

    def __init__(self, folder_name, image_path, frames_path):
        super().__init__()
        self.frames_path = frames_path
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.label = QLabel(folder_name)
        self.label.setAlignment(Qt.AlignCenter)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)

        # Load and resize preview image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(qimg))

        layout.addWidget(self.img_label)
        layout.addWidget(self.label)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.frames_path)





class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Browser")
        self.resize(1000, 800)
        self.grid_columns = 6

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.btn_load_folder = QPushButton("Select Root Folder")
        self.btn_load_folder.clicked.connect(self.load_folder)

        self.stack = QStackedWidget()
        self.layout.addWidget(self.btn_load_folder)
        self.layout.addWidget(self.stack)

        # View 1: Root folder with FileTiles
        self.root_view = QWidget()
        self.root_layout = QGridLayout()
        self.root_view.setLayout(self.root_layout)
        self.root_scroll = QScrollArea()
        self.root_scroll.setWidgetResizable(True)
        self.root_scroll.setWidget(self.root_view)
        self.stack.addWidget(self.root_scroll)

        # View 2: Frames view
        self.frames_view = QWidget()
        self.frames_main_layout = QVBoxLayout()      
        self.frames_view.setLayout(self.frames_main_layout)

        self.btn_back = QPushButton("← Back to Folder View")
        self.btn_back.clicked.connect(self.back_to_root)

        self.frames_grid_container = QWidget()
        self.frames_layout = QGridLayout()
        self.frames_grid_container.setLayout(self.frames_layout)

        self.frames_scroll = QScrollArea()
        self.frames_scroll.setWidgetResizable(True)
        self.frames_scroll.setWidget(self.frames_grid_container)

        # Assemble the frames view
        self.frames_main_layout.addWidget(self.btn_back)
        self.frames_main_layout.addWidget(self.frames_scroll)

        self.stack.addWidget(self.frames_view)


    def load_folder(self):
        root_folder = QFileDialog.getExistingDirectory(self, "Select Root Folder")
        if not root_folder:
            return

        # Clear root view layout
        while self.root_layout.count():
            child = self.root_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        row = 0
        col = 0

        for entry in sorted(os.listdir(root_folder)):
            folder_path = os.path.join(root_folder, entry)
            frames_path = os.path.join(folder_path, "processed_labeled_RGB_png", "frames")

            if not os.path.isdir(frames_path):
                continue

            images = [f for f in os.listdir(frames_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                continue

            preview_path = os.path.join(frames_path, sorted(images)[0])
            tile = FileTile(entry, preview_path, frames_path)
            tile.clicked.connect(self.open_frames_folder)
            self.root_layout.addWidget(tile, row, col)
            col += 1
            if col >= self.grid_columns:
                col = 0
                row += 1

        self.stack.setCurrentWidget(self.root_scroll)
        
        
        
        
    def open_frames_folder(self, frames_path):
        # Clear previous frames
        while self.frames_layout.count():
            child = self.frames_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        row = 0
        col = 0

        images = [f for f in sorted(os.listdir(frames_path))
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in images:
            full_path = os.path.join(frames_path, img_file)
            thumb = self.ClickableImage(full_path)
            thumb.setAlignment(Qt.AlignCenter)

            img = cv2.imread(full_path)
            img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            thumb.setPixmap(QPixmap.fromImage(qimg))

            thumb.clicked.connect(self.open_full_image)
            self.frames_layout.addWidget(thumb, row, col)
            col += 1
            if col >= self.grid_columns:
                col = 0
                row += 1

        self.stack.setCurrentWidget(self.frames_view)
        
    def back_to_root(self):
        self.stack.setCurrentWidget(self.root_scroll)
        
    class ClickableImage(QLabel):
        clicked = pyqtSignal(str)

        def __init__(self, image_path):
            super().__init__()
            self.image_path = image_path
            self.setCursor(Qt.PointingHandCursor)

        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                self.clicked.emit(self.image_path)
                
    def open_full_image(self, image_path):
        viewer = FullImageWindow(image_path)
        viewer.exec_()






if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
