from collections import Counter
from ultralytics import YOLO
import cv2


model1 = r"models/v1-yolov8s-25-epochs-weak_dataset/best.pt"
model2 = r"models/v2-yolov8m-50-epochs-normal_database/best.pt" #slowest
model3 = r"models/v3-yolov8n-50-epochs-normal_database/best.pt" #best for normal database
model4 = r"models/v4-yolov8n-10-epochs-normal_database/best.pt"
model5 = r"models/v5_yolov8n-50-epoches-new_database/best.pt"

model = YOLO(model5)
video = "sources/road_traffic.mp4"
cap = cv2.VideoCapture(video)
cv2.namedWindow("Traffic Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Detection", 1024, 640)

target_classes = ['bus', 'car', 'motorbike', 'motorcycle', 'truck']

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    results = model.predict(frame, conf=0.4, verbose=False)
    class_ids = results[0].boxes.cls.int().tolist()
    class_counts = Counter(class_ids)

    total = sum(class_counts.get(next((i for i, name in model.names.items() if name == name_), None), 0)
                for name_ in target_classes)

    annotated = results[0].plot()
    annotated = ui.draw(annotated, total, class_counts)
    annotated_with_buttons = ui.draw_button_area(annotated)

    cv2.imshow("Traffic Detection", annotated_with_buttons)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()