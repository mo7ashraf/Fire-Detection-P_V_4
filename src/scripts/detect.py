# scripts/infer_images.py
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "../../runs/detect/y10n_dfire_pyro/weights/best.pt"
SOURCE_DIR = "test_images"
OUT_DIR = "test_results"

def main():
    model = YOLO(MODEL_PATH)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = list(Path(SOURCE_DIR).glob("*.jpg"))
    for img in imgs:
        results = model(img)
        # save image with boxes
        results[0].save(filename=str(out_dir / img.name))

    print("✅ Inference done, check:", out_dir)


if __name__ == "__main__":
    main()
