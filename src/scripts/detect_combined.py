from ultralytics import YOLO
from pathlib import Path

def main():
    # path to your best model
    model_path = r"../../runs/detect/exp_y10n_combined4/weights/best.pt"
    model = YOLO(model_path)

    # 1) Single image
    img_path = r"../../test/fire.24.png"  # put some custom fire/smoke images here
    results = model(
        source=img_path,
        imgsz=640,
        conf=0.30,   # from your F1 curve: best F1 around 0.30–0.35
        iou=0.5,
        save=True,
        project="../../runs/predict",
        name="exp_fire_smoke",
        exist_ok=True,
    )

    for r in results:
        print("Image:", r.path)
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            xyxy = box.xyxy[0].tolist()
            label = r.names[cls_id]
            print(f"  {label}: conf={conf:.3f}, box={xyxy}")

    # 2) Folder of images
    folder = r"../../test/images"  # put a batch of test images here
    model(
        source=folder,
        imgsz=640,
        conf=0.30,
        iou=0.5,
        save=True,
        project="../../runs/predict",
        name="exp_fire_smoke_batch",
        exist_ok=True,
    )

if __name__ == "__main__":
    main()
