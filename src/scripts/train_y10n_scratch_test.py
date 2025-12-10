from ultralytics import YOLO

def main():
    # 1) Start from official YOLOv10n pretrained weights
    # If this fails, try 'yolov8n.pt' just to test dataset stability.
    model = YOLO("yolov10n.pt")  # will download if not present

    results = model.train(
        data="../../configs/fire_smoke_combined_new.yaml",  # make sure this points to the new dataset
        epochs=5,            # short test, just to see if losses are numbers
        imgsz=640,
        batch=8,             # smaller batch to be gentle
        lr0=0.001,           # safe LR
        amp=False,           # 🔴 disable mixed precision to avoid NaNs
        workers=2,
        project="runs",
        name="y10n_scratch_nan_test",
        exist_ok=True,
    )

    print("✅ Test training finished. Results dir:", results.save_dir)

if __name__ == "__main__":
    main()
