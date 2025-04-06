from collections import Counter
from ultralytics import YOLO
import cv2

model1 = r"models/v1-yolov8s-25-epochs-weak_dataset/best.pt"
model2 = r"models/v2-yolov8m-50-epochs/best.pt"
model3 = r"models/v3-yolov8n-50-epochs/best.pt"
model4 = r"models/v4-yolov8n-10-epochs/best.pt"

model = YOLO(model3)

video = r"sources/movingCars.mp4"

cap = cv2.VideoCapture(video)
cv2.namedWindow("Traffic Detection",cv2.WINDOW_FULLSCREEN)

target_classes = ['auto rickshaw', 'bus', 'car', 'motorbike', 'truck']

def classify_density(total):
    if total < 6:
        return "Low", (0, 255, 0)
    elif total < 30:
        return "Medium", (0, 255, 255)
    else:
        return "High", (0, 0, 255)

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break
    results = model.predict(frame, conf=0.4, verbose=False)

    class_ids = results[0].boxes.cls.int().tolist()
    class_counts = Counter(class_ids)

    print("Detected Vehicles:")
    total = 0
    for class_name in target_classes:
        class_id = next((i for i, name in model.names.items() if name == class_name), None)
        count = class_counts.get(class_id, 0) if class_id is not None else 0
        total += count
        print(f"  {class_name}: {count}")
    print("-" * 40)
    print(f"Total Vehicles: {total}")

    density_label, color = classify_density(total)

    annotated = results[0].plot()
    cv2.rectangle(annotated, (5, 10), (300, 50), (0, 0, 0), -1)
    cv2.putText(annotated, f"Density: {density_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Traffic Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""
auto rickshaw
bus
car
motorbike
truck
"""
