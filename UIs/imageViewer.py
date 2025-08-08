from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea, QPushButton, QComboBox, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt
import cv2
import os
import numpy as np
import pandas as pd
import ast  # To safely parse stringified lists like "[y, x]"

class FullImageWindow(QDialog):
    def __init__(self, processed_image_path):
        super().__init__()
        self.setWindowTitle("Full Image Viewer")
        self.setMinimumSize(800, 600)

        self.processed_image_path = processed_image_path
        self.image_path = self.get_rgb_image_path()  # We'll load this instead
        self.coords_list = []
        self.loaded_pixmap = None
        self.original_image = None  # will store unprocessed RGB image

        # Layouts and widgets
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.btn_fix_detection = QPushButton("Fix Detection")
        self.btn_fix_detection.clicked.connect(self.load_and_show_coords)
        self.btn_select_coord = QPushButton("Select New Coordinate")
        self.btn_select_coord.clicked.connect(self.start_pick_mode)
        self.layout.addWidget(self.btn_select_coord)
        

        self.coord_selector = QComboBox()
        self.coord_selector.currentIndexChanged.connect(self.update_dot_display)
        self.coord_selector.setVisible(False)

        self.scroll = QScrollArea()
        self.image_label = self.ClickableLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.click_callback = self.handle_image_click

        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.image_label)

        # Assemble layout
        self.layout.addWidget(self.btn_fix_detection)
        self.layout.addWidget(self.coord_selector)
        self.layout.addWidget(self.scroll)
        
        self.white_triplets = []
        self.pick_mode = False  # True when waiting for user click
        self.selected_point_index = 0  # Which coord from the dropdown we are editing

        self.load_image()

    def load_image(self):
        rgb = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        self.original_image = rgb.copy()
        rgb = rgb.astype(np.float32) / 255.0  # Normalize to [0, 1]
        img_name = os.path.basename(self.image_path)
        parent_dir = os.path.dirname(os.path.dirname(self.image_path))  
        parent_csv_dir = os.path.join(parent_dir, "processed_labeled_RGB_png")
        csv_folder = os.path.join(parent_csv_dir, "CSVs")
        csv_file_path = os.path.join(csv_folder, img_name.replace('.png', '.csv'))
        df = pd.read_csv(csv_file_path)
        self.white_triplets = []
        for val in df["White"].dropna():
            try:
                triplet = np.array([float(x)/255 for x in val.strip("[]").split(",")], dtype=np.float32)
                if triplet.shape == (3,):
                    self.white_triplets.append(triplet)
            except Exception as e:
                print(f"Error parsing white triplet from {val}: {e}")

        if not self.white_triplets:
            QMessageBox.warning(self, "Missing Data", "No valid white triplets found.")
            return

        white_triplet = np.mean(self.white_triplets, axis=0)
        wbParameters = white_triplet / np.linalg.norm(white_triplet)
        
        rgb = rgb/(wbParameters+1e-6)
        rgb = np.clip(rgb, 0.0, 1.0)  # Ensure values are in [0, 1]
        rgb = (rgb * 255).astype(np.uint8)  # Convert back to [0, 255] for display
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # Convert to RGB format for display
        self.processed_image_array = rgb.copy()  # Save processed image for saving later
        
        
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.loaded_pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(self.loaded_pixmap)

        
        
    def save_processed_image(self):
        if not hasattr(self, 'processed_image_array') or self.processed_image_array is None:
            QMessageBox.warning(self, "No Image", "No processed image available to save.")
            return

        img_name = os.path.basename(self.image_path)
        parent_dir = os.path.dirname(os.path.dirname(self.image_path))
        save_path = os.path.join(parent_dir, "processed_labeled_RGB_png", "frames", img_name)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # processed_image_array is RGB, OpenCV expects BGR for saving
        bgr = cv2.cvtColor(self.processed_image_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, bgr)

        QMessageBox.information(self, "Image Saved", f"Processed image saved to:\n{save_path}")
    
    def load_and_show_coords(self):
        img_name = os.path.basename(self.image_path)
        parent_dir = os.path.dirname(os.path.dirname(self.image_path))  # gets to video folder
        parent_csv_dir = os.path.join(parent_dir, "processed_labeled_RGB_png")
        csv_folder = os.path.join(parent_csv_dir, "CSVs")
        csv_file_path = os.path.join(csv_folder, img_name.replace('.png', '.csv'))

        if not os.path.exists(csv_file_path):
            QMessageBox.warning(self, "CSV Not Found", f"No CSV found for {img_name}")
            return

        try:
            df = pd.read_csv(csv_file_path)
            if 'White_coords' not in df.columns:
                QMessageBox.warning(self, "Invalid CSV", f"No 'White_coords' column in {csv_file_path}")
                return

            self.coords_list = [ast.literal_eval(str(coord)) for coord in df['White_coords'].dropna()]
            if not self.coords_list:
                QMessageBox.information(self, "No Coordinates", "No valid white coordinates found.")
                return

            self.coord_selector.clear()
            for i, coord in enumerate(self.coords_list):
                self.coord_selector.addItem(f"Point {i+1}: {coord}")

            self.coord_selector.setVisible(True)
            self.update_dot_display(0)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {e}")
            
    def update_dot_display(self, index):
        if not self.loaded_pixmap or not self.coords_list:
            return

        pixmap_copy = self.loaded_pixmap.copy()
        painter = QPainter(pixmap_copy)
        pen = QPen(Qt.red)
        pen.setWidth(4)
        painter.setPen(pen)

        try:
            x,y = self.coords_list[index]  # assume coords as [y, x]
            painter.drawPoint(int(x), int(y))  # remember: drawPoint(x, y)
        except Exception as e:
            print(f"Error drawing point: {e}")

        painter.end()
        self.image_label.setPixmap(pixmap_copy)
        
    def get_rgb_image_path(self):
        filename = os.path.basename(self.processed_image_path)

        # This gets the base folder (e.g., FileName1, FileName2, ...)
        video_folder = os.path.dirname(os.path.dirname(self.processed_image_path))

        # Go up one more level to reach the root (the folder containing FileName1, etc.)
        root_folder = os.path.dirname(video_folder)

        # Now look for RGB_png at the same level as processed_labeled_RGB_png
        rgb_image_path = os.path.join(root_folder, "RGB_png", filename)

        if not os.path.exists(rgb_image_path):
            QMessageBox.warning(self, "Missing Image", f"RGB image not found:\n{rgb_image_path}")
            return self.processed_image_path  # fallback

        return rgb_image_path
    
    def start_pick_mode(self):
        if not self.coords_list:
            QMessageBox.warning(self, "No Points", "Load detection first.")
            return
        self.selected_point_index = self.coord_selector.currentIndex()
        self.pick_mode = True
        QMessageBox.information(self, "Pick Mode", "Click on the image to select a new coordinate.")
        
    def handle_image_click(self, x, y):
        if not self.pick_mode:
            return

        # Get clicked pixel value from the displayed image
        img = self.qpixmap_to_cvimg(self.loaded_pixmap)
        clicked_rgb = self.original_image[y, x].astype(np.float32) / 255.0

        # Update in-memory coords
        self.coords_list[self.selected_point_index] = [y, x]
        self.white_triplets[self.selected_point_index] = clicked_rgb

        # --- Update CSV in /CSVs/ ---
        img_name = os.path.basename(self.image_path)
        parent_dir = os.path.dirname(os.path.dirname(self.image_path))
        csv_path = os.path.join(parent_dir, "processed_labeled_RGB_png", "CSVs", img_name.replace(".png", ".csv"))
        df = pd.read_csv(csv_path)

        if self.selected_point_index < len(df):
            df.loc[self.selected_point_index, "White_coords"] = f"[{x}, {y}]"
            df.loc[self.selected_point_index, "White"] = f"[{int(clicked_rgb[0]*255)}, {int(clicked_rgb[1]*255)}, {int(clicked_rgb[2]*255)}]"
        df.to_csv(csv_path, index=False)

        # --- Update labels.csv ---
        labels_csv_path = os.path.join(parent_dir, "labels.csv")
        labels_df = pd.read_csv(labels_csv_path)
        mean_triplet = np.mean(self.white_triplets, axis=0)
        labels_df.loc[labels_df["frame"] == img_name, ["red", "green", "blue"]] = mean_triplet * 255
        labels_df.to_csv(labels_csv_path, index=False)

        # Refresh image
        self.pick_mode = False
        self.load_image()
        if hasattr(self, 'processed_image_array') and self.processed_image_array is not None:
            img_name = os.path.basename(self.image_path)
            parent_dir = os.path.dirname(os.path.dirname(self.image_path))
            save_path = os.path.join(parent_dir, "processed_labeled_RGB_png", "frames", img_name)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            bgr = cv2.cvtColor(self.processed_image_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, bgr)
        QMessageBox.information(self, "Updated", "Coordinate and CSV values updated.")
        
    def qpixmap_to_cvimg(self, pixmap):
        qimg = pixmap.toImage()
        qimg = qimg.convertToFormat(QImage.Format_RGB888)
        width = qimg.width()
        height = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        arr = np.array(ptr, dtype=np.uint8).reshape(height, width, 3)
        return arr
        
    class ClickableLabel(QLabel):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.click_callback = None

        def mousePressEvent(self, event):
            if self.click_callback:
                x = event.pos().x()
                y = event.pos().y()
                self.click_callback(x, y)
