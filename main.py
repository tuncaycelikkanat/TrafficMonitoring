import os
import sys
import cv2
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from ultralytics import YOLO
from collections import Counter
from ui import TrafficUI

model1 = r"models/v1-yolov8s-25-epochs-weak_dataset/best.pt"
model2 = r"models/v2-yolov8m-50-epochs-normal_database/best.pt" #slowest
model3 = r"models/v3-yolov8n-50-epochs-normal_database/best.pt" #best for normal database
model4 = r"models/v4-yolov8n-10-epochs-normal_database/best.pt"
model5 = r"models/v5_yolov8n-50-epoches-new_database/best.pt"

model = YOLO(model5)
video_path = r"sources/road_traffic.mp4"

target_classes = ['bus', 'car', 'motorbike', 'motorcycle', 'truck']

def classify_density(total):
    if total < 6:
        return "Low", (0, 255, 0)
    elif total < 15:
        return "Medium", (0, 255, 255)
    else:
        return "High", (0, 0, 255)

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

    # Start button
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

    # Stop button
    def stop_video(self):
        if self.cap and self.cap.isOpened():
            self.timer.stop()
            self.cap.release()
            print("The video is stopped.")
        else:
            print("The video has stopped already.")

    def classify_density(self, total):
        if total < 6:
            return "Low", (0, 255, 0)
        elif total < 15:
            return "Medium", (0, 255, 255)
        else:
            return "High", (0, 0, 255)

    def process_frame(self):
        success, frame = self.cap.read()
        if not success:
            print("The video finished.")
            self.timer.stop()
            self.cap.release()
            return

        results = model.predict(frame, conf=0.1, verbose=False)
        class_ids = results[0].boxes.cls.int().tolist()
        class_counts = Counter(class_ids)

        total = 0
        for class_name in target_classes:
            class_id = next((i for i, name in model.names.items() if name == class_name), None)
            count = class_counts.get(class_id, 0) if class_id is not None else 0
            total += count

        density_label, color = self.classify_density(total)

        annotated = results[0].plot()
        cv2.rectangle(annotated, (5, 10), (300, 50), (0, 0, 0), -1)
        cv2.putText(annotated, f"Density: {density_label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        self.ui.update_image(annotated)

    def run(self):
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    TrafficApp().run()