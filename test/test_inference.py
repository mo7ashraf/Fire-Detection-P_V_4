from ultralytics import YOLO

model = YOLO("runs/detect/exp_y10n_t1000_stable7/weights/best.pt")
results = model.predict(source="test/fire.24.png", imgsz=512, conf=0.3)

for r in results:
    print(r)
    r.show()  # Display the image with detections