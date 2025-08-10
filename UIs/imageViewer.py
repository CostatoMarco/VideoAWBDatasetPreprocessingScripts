from PyQt5.QtWidgets import QDialog, QVBoxLayout, QComboBox, QPushButton, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QEvent
import cv2
import os
import numpy as np
import pandas as pd
import ast


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._zoom = 0

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
            self._zoom += 1
        else:
            zoom_factor = zoom_out_factor
            self._zoom -= 1

        if self._zoom > 0:
            self.scale(zoom_factor, zoom_factor)
        elif self._zoom == 0:
            self.resetTransform()
        else:
            self._zoom = 0


class FullImageWindow(QDialog):
    def __init__(self, processed_image_path):
        super().__init__()
        self.setWindowTitle("Full Image Viewer")
        self.resize(1000, 800)

        self.processed_image_path = processed_image_path
        self.image_path = self.get_rgb_image_path()
        self.coords_list = []
        self.loaded_pixmap = None
        self.original_image = None
        self.white_triplets = []
        self.pick_mode = False
        self.selected_point_index = 0

        layout = QVBoxLayout(self)
        self.btn_fix_detection = QPushButton("Fix Detection")
        self.btn_fix_detection.clicked.connect(self.load_and_show_coords)
        self.btn_select_coord = QPushButton("Select New Coordinate")
        self.btn_select_coord.clicked.connect(self.start_pick_mode)

        self.coord_selector = QComboBox()
        self.coord_selector.currentIndexChanged.connect(self.update_dot_display)
        self.coord_selector.setVisible(False)

        layout.addWidget(self.btn_fix_detection)
        layout.addWidget(self.btn_select_coord)
        layout.addWidget(self.coord_selector)

        # Zoomable view
        self.scene = QGraphicsScene(self)
        self.view = ZoomableGraphicsView(self)
        self.view.setScene(self.scene)
        self.view.viewport().installEventFilter(self)
        layout.addWidget(self.view)

        self.load_image()

    def load_image(self):
        rgb = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        self.original_image = rgb.copy()
        rgb = rgb.astype(np.float32) / 255.0
        img_name = os.path.basename(self.image_path)

        parent_dir = os.path.dirname(os.path.dirname(self.image_path))
        csv_folder = os.path.join(parent_dir, "processed_labeled_RGB_png", "CSVs")
        csv_file_path = os.path.join(csv_folder, img_name.replace('.png', '.csv'))

        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path)
            self.white_triplets = []
            for val in df["White"].dropna():
                try:
                    triplet = np.array([float(x)/255 for x in val.strip("[]").split(",")], dtype=np.float32)
                    if triplet.shape == (3,):
                        self.white_triplets.append(triplet)
                except Exception as e:
                    print(f"Error parsing white triplet: {e}")

        if not self.white_triplets:
            QMessageBox.warning(self, "Missing Data", "No valid white triplets found.")
            return

        white_triplet = np.mean(self.white_triplets, axis=0)
        wbParameters = white_triplet / np.linalg.norm(white_triplet)
        rgb = rgb / (wbParameters + 1e-6)
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = (rgb * 255).astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        self.processed_image_array = rgb.copy()

        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.loaded_pixmap = QPixmap.fromImage(qimg)

        self.scene.clear()
        self.image_item = QGraphicsPixmapItem(self.loaded_pixmap)
        self.scene.addItem(self.image_item)

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton and self.pick_mode:
            pos = self.view.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())
            self.handle_image_click(x, y)
            return True
        return super().eventFilter(source, event)

    def load_and_show_coords(self):
        img_name = os.path.basename(self.image_path)
        parent_dir = os.path.dirname(os.path.dirname(self.image_path))
        csv_file_path = os.path.join(parent_dir, "processed_labeled_RGB_png", "CSVs", img_name.replace('.png', '.csv'))
        if not os.path.exists(csv_file_path):
            QMessageBox.warning(self, "CSV Not Found", f"No CSV found for {img_name}")
            return

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

    def update_dot_display(self, index):
        if not self.loaded_pixmap or not self.coords_list:
            return
        pixmap_copy = self.loaded_pixmap.copy()
        painter = QPainter(pixmap_copy)
        pen = QPen(Qt.red)
        pen.setWidth(4)
        painter.setPen(pen)
        try:
            x, y = self.coords_list[index]
            painter.drawPoint(int(x), int(y))
        except Exception as e:
            print(f"Error drawing point: {e}")
        painter.end()
        self.image_item.setPixmap(pixmap_copy)

    def get_rgb_image_path(self):
        filename = os.path.basename(self.processed_image_path)
        video_folder = os.path.dirname(os.path.dirname(self.processed_image_path))
        root_folder = os.path.dirname(video_folder)
        rgb_image_path = os.path.join(root_folder, "RGB_png", filename)
        if not os.path.exists(rgb_image_path):
            QMessageBox.warning(self, "Missing Image", f"RGB image not found:\n{rgb_image_path}")
            return self.processed_image_path
        return rgb_image_path

    def start_pick_mode(self):
        if not self.coords_list:
            QMessageBox.warning(self, "No Points", "Load detection first.")
            return
        self.selected_point_index = self.coord_selector.currentIndex()
        self.pick_mode = True
        QMessageBox.information(self, "Pick Mode", "Click on the image to select a new coordinate.")

    def handle_image_click(self, x, y):
        clicked_rgb = self.original_image[y, x].astype(np.float32) / 255.0
        self.coords_list[self.selected_point_index] = [x, y]
        self.white_triplets[self.selected_point_index] = clicked_rgb

        img_name = os.path.basename(self.image_path)
        parent_dir = os.path.dirname(os.path.dirname(self.image_path))
        csv_path = os.path.join(parent_dir, "processed_labeled_RGB_png", "CSVs", img_name.replace(".png", ".csv"))
        df = pd.read_csv(csv_path)
        if self.selected_point_index < len(df):
            df.loc[self.selected_point_index, "White_coords"] = f"[{x}, {y}]"
            df.loc[self.selected_point_index, "White"] = f"[{int(clicked_rgb[0]*255)}, {int(clicked_rgb[1]*255)}, {int(clicked_rgb[2]*255)}]"
        df.to_csv(csv_path, index=False)

        labels_csv_path = os.path.join(parent_dir, "labels.csv")
        labels_df = pd.read_csv(labels_csv_path)
        mean_triplet = np.mean(self.white_triplets, axis=0)
        labels_df.loc[labels_df["frame"] == img_name, ["red", "green", "blue"]] = mean_triplet * 255
        labels_df.to_csv(labels_csv_path, index=False)

        self.pick_mode = False
        self.load_image()
        self.save_processed_image()
        QMessageBox.information(self, "Updated", "Coordinate and CSV values updated.")

    def save_processed_image(self):
        img_name = os.path.basename(self.image_path)
        parent_dir = os.path.dirname(os.path.dirname(self.image_path))
        save_path = os.path.join(parent_dir, "processed_labeled_RGB_png", "frames", img_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        bgr = cv2.cvtColor(self.processed_image_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, bgr)
