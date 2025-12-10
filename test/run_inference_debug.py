from ultralytics import YOLO
from PIL import Image
model = YOLO("runs/detect/exp_y10n_t1000_stable7/weights/best.pt")

# print class names
print("Model class names:", model.names)

# optionally load and upscale image programmatically
img_path = "test/fire.125.png"
img = Image.open(img_path)
print("Original size:", img.size)

# Run inference (low conf to surface weak detections), save results
results = model.predict(source=img_path, imgsz=512, conf=0.01, save=True)

# Print boxes/conf/class for each result
for i, r in enumerate(results):
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        print(f"Result {i}: no boxes")
    else:
        print(f"Result {i}: {len(boxes)} boxes")
        # boxes.xyxy and boxes.conf and boxes.cls may be available depending on ultralytics version
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            for b, c, cl in zip(xyxy, confs, clss):
                print(" box xyxy:", b, "conf:", float(c), "class:", int(cl), "name:", model.names[int(cl)])
        except Exception:
            # fallback: print r.boxes
            print("Boxes object:", boxes)