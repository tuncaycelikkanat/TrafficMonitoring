import os
import sys
import cv2
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from ultralytics import YOLO
from collections import Counter
from ui import TrafficUI

model1 = r"models/v1-yolov8s-25-epochs-weak_dataset/best.pt"
model2 = r"models/v2-yolov8m-50-epochs-normal_database/best.pt"
model3 = r"models/v3-yolov8n-50-epochs-normal_database/best.pt"
model4 = r"models/v4-yolov8n-10-epochs-normal_database/best.pt"
model5 = r"models/v5_yolov8n-50-epoches-new_database/best.pt"

model = YOLO(model5)
video_path = r"sources/road_traffic.mp4"

target_classes = ['bus', 'car', 'motorbike', 'motorcycle', 'truck']

class TrafficApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.ui = TrafficUI()
        self.ui.show()

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

        self.ui.start_button.clicked.connect(self.start_video)
        self.ui.stop_button.clicked.connect(self.stop_video)

        self.frame_count = 0
        self.log_file = open("traffic_density_log.txt", "w")

    def __del__(self):
        if self.log_file:
            self.log_file.close()

    def start_video(self):
        video_path = self.ui.path_input.text()
        if not os.path.exists(video_path):
            print("The video could not be found:", video_path)
            return

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("The video could not be opened")
            return

        self.timer.start(10)

    def stop_video(self):
        if self.cap and self.cap.isOpened():
            self.timer.stop()
            self.cap.release()
            print("The video is stopped.")
        else:
            print("The video has stopped already.")

    def calculate_area_density(self, boxes, frame_area):
        total_area = 0
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            total_area += area
        percent = (total_area / frame_area) * 100
        return round(percent, 2)

    def classify_density(self, percent):
        adjusted_percent = min(100.0, percent * 3)  
        if adjusted_percent < 30.0:
            return "Low", (0, 255, 0), adjusted_percent
        elif adjusted_percent < 60.0:
            return "Medium", (0, 255, 255), adjusted_percent
        else:
            return "High", (0, 0, 255), adjusted_percent

    def process_frame(self):
        success, frame = self.cap.read()
        if not success:
            print("The video finished.")
            self.timer.stop()
            self.cap.release()
            return

        self.frame_count += 1
        results = model.predict(frame, conf=0.1, verbose=False)
        boxes = []
        class_ids = []

        for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            class_name = model.names[int(cls)]
            if class_name in target_classes:
                boxes.append(box)
                class_ids.append(int(cls))

        frame_area = frame.shape[0] * frame.shape[1]
        density_percent = self.calculate_area_density(boxes, frame_area)
        density_label, color, adjusted_percent = self.classify_density(density_percent)

        annotated = results[0].plot()

        # Backup for the original frame
        overlay = annotated.copy()
        cv2.rectangle(overlay, (5, 10), (500, 80), (0, 0, 0), -1)
        alpha = 0.4 
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
        cv2.putText(annotated, f"Density: {density_label} ({round(adjusted_percent,1)}%)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Draw the density bar
        bar_x = 10
        bar_y = 85
        bar_width = 400
        bar_height = 20
        fill_width = int(bar_width * (adjusted_percent / 100.0))
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 2)
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)

        self.ui.update_image(annotated)

        if self.frame_count % 25 == 0:
            timestamp = time.strftime("%H:%M:%S")
            self.log_file.write(f"[{timestamp}] Density: {density_label} - {round(adjusted_percent,1)}%\n")
            self.log_file.flush()

    def run(self):
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    TrafficApp().run()
