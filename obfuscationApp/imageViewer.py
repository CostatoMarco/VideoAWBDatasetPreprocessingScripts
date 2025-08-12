import os
import pandas as pd
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsItem
)
from PyQt5.QtGui import QPixmap, QPen, QColor, QBrush, QCursor
from PyQt5.QtCore import Qt, QRectF, QPointF


class ZoomableGraphicsView(QGraphicsView):
    """QGraphicsView that supports mouse wheel zooming and drag panning."""
    def __init__(self, parent=None, max_zoom=10):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._zoom = 0
        self.max_zoom = max_zoom

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:  # zoom in
            if self._zoom < self.max_zoom:
                self.scale(zoom_in_factor, zoom_in_factor)
                self._zoom += 1
        else:  # zoom out
            if self._zoom > 0:
                self.scale(zoom_out_factor, zoom_out_factor)
                self._zoom -= 1

        if self._zoom == 0:
            self.resetTransform()


from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsItem
from PyQt5.QtGui import QPen, QBrush, QColor, QCursor
from PyQt5.QtCore import Qt, QRectF, QPointF


class HandleItem(QGraphicsRectItem):
    SIZE = 10

    def __init__(self, cursor_type, position_flag, parent=None):
        super().__init__(-HandleItem.SIZE/2, -HandleItem.SIZE/2, HandleItem.SIZE, HandleItem.SIZE, parent)
        self.setBrush(QBrush(QColor('red')))
        self.setPen(QPen(Qt.NoPen))
        self.setCursor(cursor_type)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

        self.position_flag = position_flag
        self._ignore_position_change = False  # Guard flag

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.parentItem():
            if self._ignore_position_change:
                # Ignore changes when flag is set
                return value
            # Call parent's resize handler
            self.parentItem().handle_moved(self, value)
            # Ignore this movement; parent will reposition handle properly
            return self.pos()
        return super().itemChange(change, value)

    def setPos(self, *args, **kwargs):
        # Override setPos to set guard flag during repositioning
        self._ignore_position_change = True
        super().setPos(*args, **kwargs)
        self._ignore_position_change = False


class ResizableRectItem(QGraphicsRectItem):
    HANDLE_POSITIONS = {
        "top_left": Qt.SizeFDiagCursor,
        "top": Qt.SizeVerCursor,
        "top_right": Qt.SizeBDiagCursor,
        "right": Qt.SizeHorCursor,
        "bottom_right": Qt.SizeFDiagCursor,
        "bottom": Qt.SizeVerCursor,
        "bottom_left": Qt.SizeBDiagCursor,
        "left": Qt.SizeHorCursor,
    }

    def __init__(self, rect):
        super().__init__(rect)
        self.setFlags(
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self.brush = QBrush(QColor(0, 0, 0))  # Solid black fill
        self.pen = QPen(Qt.NoPen)  # No border
        self.setBrush(self.brush)
        self.setPen(self.pen)

        self.handles = {}
        self.handle_selected = None
        self.mouse_press_pos = None
        self.mouse_press_rect = None

        # Create handle items
        for pos_name, cursor in self.HANDLE_POSITIONS.items():
            handle = HandleItem(cursor, pos_name, self)
            self.handles[pos_name] = handle

        self.update_handles_pos()

    def update_handles_pos(self):
        rect = self.rect()
        for pos_name, handle in self.handles.items():
            if pos_name == "top_left":
                handle.setPos(rect.topLeft())
            elif pos_name == "top":
                handle.setPos(rect.left() + rect.width()/2, rect.top())
            elif pos_name == "top_right":
                handle.setPos(rect.topRight())
            elif pos_name == "right":
                handle.setPos(rect.right(), rect.top() + rect.height()/2)
            elif pos_name == "bottom_right":
                handle.setPos(rect.bottomRight())
            elif pos_name == "bottom":
                handle.setPos(rect.left() + rect.width()/2, rect.bottom())
            elif pos_name == "bottom_left":
                handle.setPos(rect.bottomLeft())
            elif pos_name == "left":
                handle.setPos(rect.left(), rect.top() + rect.height()/2)

    def handle_moved(self, handle, new_pos):
        # Called when a handle is moved; resize the rect accordingly
        rect = self.rect()
        old_rect = QRectF(rect)

        pos_name = handle.position_flag
        delta = new_pos - handle.pos()  # Will be (0,0) because we ignore handle move itself
        # Instead, calculate based on new_pos relative to rect coords:
        p = new_pos

        if pos_name == "top_left":
            new_rect = QRectF(p.x(), p.y(), rect.right() - p.x(), rect.bottom() - p.y())
        elif pos_name == "top":
            new_rect = QRectF(rect.left(), p.y(), rect.width(), rect.bottom() - p.y())
        elif pos_name == "top_right":
            new_rect = QRectF(rect.left(), p.y(), p.x() - rect.left(), rect.bottom() - p.y())
        elif pos_name == "right":
            new_rect = QRectF(rect.left(), rect.top(), p.x() - rect.left(), rect.height())
        elif pos_name == "bottom_right":
            new_rect = QRectF(rect.left(), rect.top(), p.x() - rect.left(), p.y() - rect.top())
        elif pos_name == "bottom":
            new_rect = QRectF(rect.left(), rect.top(), rect.width(), p.y() - rect.top())
        elif pos_name == "bottom_left":
            new_rect = QRectF(p.x(), rect.top(), rect.right() - p.x(), p.y() - rect.top())
        elif pos_name == "left":
            new_rect = QRectF(p.x(), rect.top(), rect.right() - p.x(), rect.height())
        else:
            return  # Unknown handle

        # Enforce minimum size
        min_size = 20
        if new_rect.width() < min_size:
            if pos_name in ["top_left", "left", "bottom_left"]:
                new_rect.setLeft(new_rect.right() - min_size)
            else:
                new_rect.setRight(new_rect.left() + min_size)
        if new_rect.height() < min_size:
            if pos_name in ["top_left", "top", "top_right"]:
                new_rect.setTop(new_rect.bottom() - min_size)
            else:
                new_rect.setBottom(new_rect.top() + min_size)

        # Prevent inverted rects (should be handled by min size, but just in case)
        if new_rect.left() > new_rect.right():
            new_rect.setLeft(new_rect.right())
        if new_rect.top() > new_rect.bottom():
            new_rect.setTop(new_rect.bottom())

        self.prepareGeometryChange()
        self.setRect(new_rect)
        self.update_handles_pos()
        self.update()

    def mouseMoveEvent(self, event):
        if self.handle_selected is None:
            super().mouseMoveEvent(event)
            self.update()

    def mousePressEvent(self, event):
        # Disable selecting handles separately; handled inside HandleItem
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)


class FullImageWindow(QDialog):
    def __init__(self, processed_image_path):
        super().__init__()
        self.setWindowTitle("Full Image Viewer")
        self.resize(1000, 800)
        self.processed_image_path = processed_image_path

        layout = QVBoxLayout(self)
        self.scene = QGraphicsScene(self)
        self.view = ZoomableGraphicsView(self)
        self.view.setScene(self.scene)
        layout.addWidget(self.view)

        self._load_image_and_boxes()

    def _load_image_and_boxes(self):
        if not os.path.exists(self.processed_image_path):
            print(f"Image not found: {self.processed_image_path}")
            return

        pixmap = QPixmap(self.processed_image_path)
        if pixmap.isNull():
            print(f"Failed to load image: {self.processed_image_path}")
            return

        # Add the base image
        self.scene.clear()
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)

        # Load boxes CSV
        csv_path = self._get_boxes_csv_path()
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    y1 = int(row['bbox_y1'])
                    x1 = int(row['bbox_x1'])
                    y2 = int(row['bbox_y2'])
                    x2 = int(row['bbox_x2'])
                    rect_item = ResizableRectItem(QRectF(x1, y1, x2 - x1, y2 - y1))
                    self.scene.addItem(rect_item)
            except Exception as e:
                print(f"Error reading CSV {csv_path}: {e}")

    def _get_boxes_csv_path(self):
        filename = os.path.basename(self.processed_image_path).replace(".png", ".csv")
        video_folder = os.path.dirname(os.path.dirname(self.processed_image_path))
        return os.path.join(video_folder, "CSVs", "BoxesCSVs", filename)
