import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO("yolov8n.pt")  # lightweight model

# Initialize tracker
tracker = DeepSort(max_age=30)

# Start video capture
cap = cv2.VideoCapture(0)  # 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = model(frame)

    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])

            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # Track objects
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = map(int, track.to_ltrb())

        # Draw bounding box
        cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
