import os
import sys
import cv2
import time
import numpy as np
from collections import deque
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer
from ultralytics import YOLO
from ui import TrafficUI
from plot import DensityPlotCanvas
from database_manager import DatabaseManager

# Define relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SOURCES_DIR = os.path.join(BASE_DIR, "sources")

MODEL_PATHS = {
    "Model1": os.path.join(MODELS_DIR, "v1-yolov8s-25-epochs-weak_dataset", "best.pt"),
    "Model2": os.path.join(MODELS_DIR, "v2-yolov8m-50-epochs-normal_database", "best.pt"),
    "Model3": os.path.join(MODELS_DIR, "v3-yolov8n-50-epochs-normal_database", "best.pt"),
    "Model4": os.path.join(MODELS_DIR, "v4-yolov8n-10-epochs-normal_database", "best.pt"),
    "Model5": os.path.join(MODELS_DIR, "v5_yolov8n-50-epoches-new_database", "best.pt"),
}
DEFAULT_VIDEO = os.path.join(SOURCES_DIR, "road_traffic.mp4")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
TARGET_CLASSES = ['bus', 'car', 'motorbike', 'motorcycle', 'truck']

class TrafficApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.ui = TrafficUI()
        self.cap = None
        self.db_manager = DatabaseManager()

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

        self.log_timer = QTimer() # new timer for satabase records
        self.log_timer.timeout.connect(self.log_data_to_db)

        self.frame_count = 0
        self.start_time = time.time()
        self.last_plot_update = 0
        self.current_frame = None
        self.fps = 0
        self.last_fps_time = time.time()
        self.frame_counter = 0
        self.density_history = deque(maxlen=100)
        self.avg_density = 0
        self.last_processed_vehicle_count = 0
        self.last_processed_density_label = "N/A"
        self.last_processed_density_percentage = 0.0

        # Create directories if needed
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR, exist_ok=True)
        if not os.path.exists(SOURCES_DIR):
            os.makedirs(SOURCES_DIR, exist_ok=True)

        self.ui.show()
        self.setup_connections()
        self.initialize_system()

    def setup_connections(self):
        self.ui.start_button.clicked.connect(self.start_video)
        self.ui.stop_button.clicked.connect(self.stop_video)
        self.ui.save_button.clicked.connect(self.save_snapshot)
        self.ui.model_combo.currentIndexChanged.connect(self.change_model)

    def initialize_system(self):
        # Check if models exist
        self.available_models = {}
        for name, path in MODEL_PATHS.items():
            if os.path.exists(path):
                self.available_models[name] = path

        if not self.available_models:
            QMessageBox.critical(self.ui, "Error",
                                "No model files found in the models directory!")
            self.ui.model_combo.setEnabled(False)
            self.ui.start_button.setEnabled(False)
        else:
            self.ui.model_combo.clear()
            self.ui.model_combo.addItems(self.available_models.keys())
            self.change_model()  # Load initial model

        self.density_canvas = DensityPlotCanvas()
        self.ui.graph_layout.addWidget(self.density_canvas)


    def change_model(self):
        model_name = self.ui.model_combo.currentText()
        model_path = self.available_models.get(model_name)

        if not model_path or not os.path.exists(model_path):
            self.ui.status_label.setText(f"Model file not found: {model_name}")
            return

        try:
            self.model = YOLO(model_path)
            self.ui.status_label.setText(f"Model loaded: {model_name}")
            print(f"Model changed to: {model_name}")
        except Exception as e:
            self.ui.status_label.setText(f"Error loading model: {str(e)}")
            print(f"Error loading model: {e}")

    def cleanup_resources(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

        if self.db_manager:
            self.db_manager.close()

    def __del__(self):
        self.cleanup_resources()

    def start_video(self):
        video_path = self.ui.path_input.text().strip() or DEFAULT_VIDEO

        if not os.path.exists(video_path):
            QMessageBox.warning(self.ui, "File Not Found",
                               f"Video file not found:\n{video_path}")
            return

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self.ui, "Error",
                                "Failed to open video file")
            return

        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.ui.video_info.setText(
            f"Resolution: {width}x{height} | FPS: {fps:.1f} | Model: {self.ui.model_combo.currentText()}"
        )

        self.frame_count = 0
        self.start_time = time.time()
        self.last_plot_update = self.start_time
        self.last_fps_time = self.start_time
        self.frame_counter = 0
        self.timer.start(30)  # ~33 FPS
        self.log_timer.start(1000)
        self.ui.status_label.setText("Processing video...")

    def stop_video(self):
        if self.cap and self.cap.isOpened():
            self.timer.stop()
            self.log_timer.stop()
            self.cap.release()
            self.ui.status_label.setText("Video stopped")
            print("Video stopped")
        else:
            self.ui.status_label.setText("No active video")

    # new method added for database
    def log_data_to_db(self):
        if self.db_manager.conn:
            density_label, _, adjusted = self.classify_density(
                self.avg_density)
            vehicle_count = self.frame_count

            current_vehicle_count = getattr(self, 'last_processed_vehicle_count', 0)

            self.db_manager.insert_log(
                density_label=density_label,
                density_percentage=adjusted,
                vehicle_count=current_vehicle_count,
                fps=self.fps
            )
            print(
                "***Logged***")

    def save_snapshot(self):
        if self.current_frame is None:
            QMessageBox.warning(self.ui, "No Frame",
                               "No frame available to save")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(SCREENSHOT_DIR, f"snapshot_{timestamp}.jpg")

        # Convert to BGR format before saving
        save_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, save_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        self.ui.status_label.setText(f"Snapshot saved: {filename}")
        print(f"Snapshot saved: {filename}")

    def calculate_area_density(self, boxes, frame_area):
        total_area = sum(
            max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))
            for x1, y1, x2, y2 in boxes
        )
        return round((total_area / frame_area) * 100, 2)

    def classify_density(self, percent):
        adjusted = min(100.0, percent * 3)
        if adjusted < 30.0:
            return "Low", (50, 205, 50), adjusted  # LimeGreen
        elif adjusted < 60.0:
            return "Medium", (255, 215, 0), adjusted   # Gold
        else:
            return "High", (220, 20, 60), adjusted  # Crimson

    def process_frame(self):
        if not self.cap or not self.cap.isOpened():
            self.timer.stop()
            return

        success, frame = self.cap.read()
        if not success:
            self.timer.stop()
            self.cap.release()
            self.ui.status_label.setText("Video processing completed")
            return

        # Calculate FPS
        self.frame_counter += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self.frame_counter / elapsed
            self.frame_counter = 0
            self.last_fps_time = current_time

        # Process frame
        results = self.model.predict(frame, conf=0.1, verbose=False)

        boxes = []
        if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
            if hasattr(results[0].boxes, "xyxy") and hasattr(results[0].boxes, "cls"):
                boxes = [
                    box for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls)
                    if self.model.names[int(cls)] in TARGET_CLASSES
                ]

        self.last_processed_vehicle_count = len(boxes)

        frame_area = frame.shape[0] * frame.shape[1]
        density_percent = self.calculate_area_density(boxes, frame_area)
        self.density_history.append(density_percent)
        self.avg_density = sum(self.density_history) / len(self.density_history)

        label, color, adjusted = self.classify_density(density_percent)

        # Annotate frame
        annotated = results[0].plot()
        h, w, _ = annotated.shape

        # Create info panel
        panel_height = 90
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)

        # Add text info
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, f"Density: {label} ({adjusted:.1f}%)", (20, 30),
                    font, 0.9, color, 2)
        cv2.putText(panel, f"FPS: {self.fps:.1f} | Vehicles: {len(boxes)}", (20, 65),
                    font, 0.7, (220, 220, 220), 1)
        cv2.putText(panel, f"Avg: {self.avg_density:.1f}%", (w - 200, 65),
                    font, 0.7, (180, 180, 255), 1)

        # Add density bar
        bar_x, bar_y = 20, panel_height - 25
        bar_width = w - 40
        fill = int(bar_width * (adjusted / 100))
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + 12),
                      (100, 100, 100), 1)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill, bar_y + 12),
                      color, -1)

        # Combine panel with frame
        combined = np.vstack([annotated, panel])

        # Store current frame
        self.current_frame = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        self.ui.update_image(combined)

        # Update plot
        elapsed_time = current_time - self.start_time
        self.density_canvas.update_plot(elapsed_time, adjusted)

    def run(self):
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    app = TrafficApp()
    app.run()