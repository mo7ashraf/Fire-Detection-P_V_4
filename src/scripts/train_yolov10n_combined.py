# src/scripts/train_y10n_dfire_pyro.py

from ultralytics import YOLO
from pathlib import Path


def main():
    # Paths are relative to the directory you run the script from (repo root)
    data_config = "../../configs/fire_smoke_combined.yaml"

    # Old experiment best model
    # If your best.pt is directly under exp_y10n_t1000_stable7, change this path
    old_exp_best = Path("../../best_model/exp_y10n_t1000_stable7/weights/best.pt")

    if not old_exp_best.exists():
        raise FileNotFoundError(
            f"Could not find old best model at: {old_exp_best}. "
            "Update the path in train_y10n_dfire_pyro.py if your structure is different."
        )

    # Load old best model (YOLOv10n) and fine-tune on combined dataset
    model = YOLO(str(old_exp_best))

    results = model.train(
        data=data_config,
        epochs=100,          # you can change
        imgsz=640,
        batch=16,
        patience=30,
        optimizer="SGD",     # or "AdamW"
        lr0=0.001,
        cos_lr=True,
        workers=4,
        project="../../runs",
        name="y10n_dfire_pyro",
        exist_ok=True,
        pretrained=False,    # we already loaded a trained model
    )

    print("✅ Training finished.")
    print("📁 Best model and results saved in:", results.save_dir)


if __name__ == "__main__":
    main()
