from ultralytics import YOLO
from pathlib import Path


def main():
    data_config = "../../configs/fire_smoke_combined_new.yaml"

    old_exp_best = Path("../../best_model/exp_y10n_t1000_stable7/weights/best.pt")
    if not old_exp_best.exists():
        raise FileNotFoundError(f"Old best model not found at {old_exp_best}")

    model = YOLO(str(old_exp_best))

    results = model.train(
        data=data_config,
        epochs=100,
        imgsz=640,
        batch=16,
        patience=30,
        optimizer="SGD",
        lr0=0.001,      # safer LR for fine-tuning
        cos_lr=True,
        workers=4,
        project="runs",
        name="y10n_dfire_pyro",
        exist_ok=True,
        pretrained=False,
    )

    print("✅ Training finished. Saved to:", results.save_dir)


if __name__ == "__main__":
    main()
