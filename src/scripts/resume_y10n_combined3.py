from ultralytics import YOLO
from pathlib import Path


def main():
    ckpt = Path("../../runs/detect/exp_y10n_combined3/weights/last.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt}")

    # Load the checkpoint
    model = YOLO(str(ckpt))

    # Resume training with the saved args (data, epochs, etc.)
    results = model.train(
        resume=True
    )

    print("✅ Resumed training finished. Final results at:", results.save_dir)


if __name__ == "__main__":
    main()
