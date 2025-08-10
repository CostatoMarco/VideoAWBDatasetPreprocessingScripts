# main.py

import sys
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QPushButton, QFileDialog, QScrollArea, QHBoxLayout, QGridLayout, QStackedWidget, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from imageViewer import FullImageWindow
import pandas as pd


class FileTile(QWidget):
    clicked = pyqtSignal(str)  # Emits path to the frames folder

    def __init__(self, folder_name, image_path, frames_path):
        super().__init__()
        self.frames_path = frames_path
        self.current_frames_path = None  # Keep track of which folder is open
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
        self.btn_delete_mode = QPushButton("🗑 Delete Mode (Off)")
        self.btn_delete_mode.setCheckable(True)
        self.btn_delete_mode.clicked.connect(self.toggle_delete_mode)
        self.delete_mode = False
        


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
        self.frames_main_layout.addWidget(self.btn_back)
        self.frames_main_layout.addWidget(self.btn_delete_mode)
        self.frames_main_layout.addWidget(self.frames_scroll)

        self.stack.addWidget(self.frames_view)
        
    def toggle_delete_mode(self):
        self.delete_mode = self.btn_delete_mode.isChecked()
        if self.delete_mode:
            self.btn_delete_mode.setText("🗑 Delete Mode (On)")
        else:
            self.btn_delete_mode.setText("🗑 Delete Mode (Off)")


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
        self.current_frames_path = frames_path  # Save path for refresh
        self.video_folder = os.path.dirname(os.path.dirname(frames_path))  # Root/<Video>
        
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

            # Create container widget
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(5, 5, 5, 5)
            vbox.setSpacing(2)

            # Clickable image
            thumb = self.ClickableImage(full_path)
            thumb.setAlignment(Qt.AlignCenter)
            thumb.clicked.connect(lambda path=full_path: self.image_clicked(path))

            img = cv2.imread(full_path)
            img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            thumb.setPixmap(QPixmap.fromImage(qimg))

            # Name label
            name_label = QLabel(img_file)
            name_label.setAlignment(Qt.AlignCenter)

            vbox.addWidget(thumb)
            vbox.addWidget(name_label)

            self.frames_layout.addWidget(container, row, col)
            col += 1
            if col >= self.grid_columns:
                col = 0
                row += 1

        self.stack.setCurrentWidget(self.frames_view)
        
    def image_clicked(self, image_path):
        if self.delete_mode:
            reply = QMessageBox.question(
                self, "Confirm Delete",
                f"Delete {os.path.basename(image_path)} and related files?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.delete_image_and_data(image_path)
                self.open_frames_folder(self.current_frames_path)  # Refresh view
        else:
            self.open_full_image(image_path)
            
    def delete_image_and_data(self, image_path):
        img_name = os.path.basename(image_path)
        img_stem = os.path.splitext(img_name)[0]

        # 1) Delete image
        if os.path.exists(image_path):
            os.remove(image_path)

        # 2) Delete CSV in CSVs folder
        csvs_path = os.path.join(self.video_folder, "processed_labeled_RGB_png", "CSVs")
        csv_file = os.path.join(csvs_path, f"{img_stem}.csv")
        if os.path.exists(csv_file):
            os.remove(csv_file)

        # 3) Remove entry from labels.csv
        labels_csv_path = os.path.join(self.video_folder, "labels.csv")
        if os.path.exists(labels_csv_path):
            df = pd.read_csv(labels_csv_path)
            df = df[df["frame"] != img_name]
            df.to_csv(labels_csv_path, index=False)

        
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
        
    def changeEvent(self, event):
        if event.type() == QEvent.ActivationChange:
            if self.isActiveWindow() and self.stack.currentWidget() == self.frames_view:
                if self.current_frames_path:
                    self.open_frames_folder(self.current_frames_path)
        super().changeEvent(event)






if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
