import sys
from collections import Counter
from ultralytics import YOLO
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton

class TrafficUI:
    def __init__(self, target_classes, model_names):
        self.target_classes = target_classes
        self.model_names = model_names

    def classify_density(self, total):
        if total < 6:
            return "Low", (0, 255, 0)
        elif total < 15:
            return "Medium", (0, 255, 255)
        else:
            return "High", (0, 0, 255)

    def draw(self, frame, total, class_counts):
        height, width, _ = frame.shape

        # Semi-transparent panel
        overlay = frame.copy()
        panel_width = 320
        panel_height = 200
        panel_x = 20
        panel_y = 20
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (50, 50, 50), -1)

        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Density label
        density_label, color = self.classify_density(total)
        cv2.putText(frame, f"Traffic Density: {density_label}", (panel_x + 15, panel_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Class counts
        y_offset = 70
        for class_name in self.target_classes:
            class_id = next((i for i, name in self.model_names.items() if name == class_name), None)
            count = class_counts.get(class_id, 0) if class_id is not None else 0
            text = f"{class_name}: {count}"
            cv2.putText(frame, text, (panel_x + 15, panel_y + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30

        return frame

    def draw_button_area(self, frame):
        height, width, _ = frame.shape
        button_area_height = 80
        button_y_start = height
        new_frame = np.zeros((height + button_area_height, width, 3), dtype=np.uint8)
        new_frame[:height] = frame

        # Draw button background
        cv2.rectangle(new_frame, (0, height), (width, height + button_area_height), (30, 30, 30), -1)

        # Draw a sample button (e.g., Report)
        button_color = (70, 70, 200)
        button_text = "Report"
        button_x1, button_y1 = 50, height + 20
        button_x2, button_y2 = 200, height + 60
        cv2.rectangle(new_frame, (button_x1, button_y1), (button_x2, button_y2), button_color, -1)
        cv2.putText(new_frame, button_text, (button_x1 + 20, button_y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return new_frame

    def on_report_button_click(self):
        # Placeholder for report button action
        print("Report button clicked")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Traffic Detection")
        self.setGeometry(100, 100, 1024, 720)

        # Create the model and UI
        self.model = YOLO("models/v5_yolov8n-50-epoches-new_database/best.pt")
        self.target_classes = ['bus', 'car', 'motorbike', 'motorcycle', 'truck']
        self.ui = TrafficUI(self.target_classes, self.model.names)

        # Create a label for displaying the video
        self.image_label = QLabel(self)
        self.image_label.resize(1024, 640)

        # Create a report button
        self.report_button = QPushButton("Report", self)
        self.report_button.clicked.connect(self.ui.on_report_button_click)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(self.report_button)

        # Video capture
        self.cap = cv2.VideoCapture("sources/road_traffic.mp4")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        success, frame = self.cap.read()
        if not success:
            return

        # Process frame
        results = self.model.predict(frame, conf=0.4, verbose=False)
        class_ids = results[0].boxes.cls.int().tolist()
        class_counts = Counter(class_ids)

        total = sum(class_counts.get(next((i for i, name in self.model.names.items() if name == name_), None), 0)
                    for name_ in self.target_classes)

        annotated = results[0].plot()
        annotated = self.ui.draw(annotated, total, class_counts)
        annotated_with_buttons = self.ui.draw_button_area(annotated)

        # Convert OpenCV image (BGR) to QImage (RGB)
        rgb_image = cv2.cvtColor(annotated_with_buttons, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Update label with the new image
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
