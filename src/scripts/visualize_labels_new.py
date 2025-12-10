import cv2
from pathlib import Path

DATASET_ROOT = Path("../../dataset/fire_smoke_combined")
IMG_DIR = DATASET_ROOT / "images/train"
LBL_DIR = DATASET_ROOT / "labels/train"

# MATCHES YAML: 0=fire, 1=smoke
CLASS_NAMES = {0: "fire", 1: "smoke"}

def draw_boxes(image_path, label_path):
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.read().strip().splitlines()

    for ln in lines:
        if not ln:
            continue
        cls, x, y, bw, bh = ln.split()
        cls = int(cls)
        x, y, bw, bh = map(float, [x, y, bw, bh])

        cx, cy = x * w, y * h
        bw, bh = bw * w, bh * h
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)

        # fire = red, smoke = green
        color = (0, 0, 255) if cls == 0 else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, CLASS_NAMES[cls], (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img


def main():
    #images = list(IMG_DIR.glob("*.jpg"))[16192:16250]  # show first 100 images
    images = list(IMG_DIR.glob("*.jpg"))[4444:4460]  # show first 100 images
    for img_path in images:
        lbl_path = LBL_DIR / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        out_img = draw_boxes(img_path, lbl_path)
        cv2.imshow("Labeled Image", out_img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
