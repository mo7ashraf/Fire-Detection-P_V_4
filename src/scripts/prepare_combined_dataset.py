import shutil
from pathlib import Path

# ====== CONFIG ======
# These paths are correct for: repo_root/src/scripts/prepare_combined_dataset.py
PYRO_IMAGES_TRAIN = Path("../../dataset/pyro_sdis_yolo/images/train")
PYRO_LABELS_TRAIN = Path("../../dataset/pyro_sdis_yolo/labels/train")
PYRO_IMAGES_VAL   = Path("../../dataset/pyro_sdis_yolo/images/val")
PYRO_LABELS_VAL   = Path("../../dataset/pyro_sdis_yolo/labels/val")

DFIRE_IMAGES_TRAIN = Path("../../dataset/dfire/train/images")
DFIRE_LABELS_TRAIN = Path("../../dataset/dfire/train/labels")
DFIRE_IMAGES_VAL   = Path("../../dataset/dfire/val/images")
DFIRE_LABELS_VAL   = Path("../../dataset/dfire/val/labels")

# Optional: if D-Fire has a test split
DFIRE_IMAGES_TEST  = Path("../../dataset/dfire/test/images")
DFIRE_LABELS_TEST  = Path("../../dataset/dfire/test/labels")
OUT_ROOT = Path("../../dataset/fire_smoke_combined")
OUT_IMG_TRAIN = OUT_ROOT / "images/train"
OUT_IMG_VAL   = OUT_ROOT / "images/val"
OUT_IMG_TEST  = OUT_ROOT / "images/test"
OUT_LBL_TRAIN = OUT_ROOT / "labels/train"
OUT_LBL_VAL   = OUT_ROOT / "labels/val"
OUT_LBL_TEST  = OUT_ROOT / "labels/test"


def ensure_dirs():
    for p in [OUT_IMG_TRAIN, OUT_IMG_VAL, OUT_IMG_TEST,
              OUT_LBL_TRAIN, OUT_LBL_VAL, OUT_LBL_TEST]:
        p.mkdir(parents=True, exist_ok=True)


def copy_pair(src_img, src_lbl, dst_img_dir, dst_lbl_dir,
              class_remap=None):
    """Copy img + label, optionally remapping class ids."""
    img_name = src_img.name
    lbl_name = src_lbl.name

    dst_img = dst_img_dir / img_name
    dst_lbl = dst_lbl_dir / lbl_name

    shutil.copy2(src_img, dst_img)

    with open(src_lbl, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()

    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        cls = int(parts[0])
        if class_remap is not None:
            if cls not in class_remap:
                # skip unknown classes if any
                continue
            cls = class_remap[cls]
        parts[0] = str(cls)
        new_lines.append(" ".join(parts))

    with open(dst_lbl, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))


def sync_split(img_dir, lbl_dir, out_img_dir, out_lbl_dir, class_remap):
    # handle .jpg and .png in case D-Fire uses both
    exts = ["*.jpg", "*.jpeg", "*.png"]
    for ext in exts:
        for img_path in img_dir.glob(ext):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            copy_pair(img_path, lbl_path, out_img_dir, out_lbl_dir,
                      class_remap=class_remap)


def main():
    ensure_dirs()

    # ---- 1) Add D-Fire (0=fire, 1=smoke) ----
    dfire_remap = {0: 0, 1: 1}  # D-Fire already matches our global [0=fire,1=smoke]

    sync_split(DFIRE_IMAGES_TRAIN, DFIRE_LABELS_TRAIN,
               OUT_IMG_TRAIN, OUT_LBL_TRAIN, dfire_remap)
    sync_split(DFIRE_IMAGES_VAL, DFIRE_LABELS_VAL,
               OUT_IMG_VAL, OUT_LBL_VAL, dfire_remap)
    if DFIRE_IMAGES_TEST.exists():
        sync_split(DFIRE_IMAGES_TEST, DFIRE_LABELS_TEST,
                   OUT_IMG_TEST, OUT_LBL_TEST, dfire_remap)

    # ---- 2) Add Pyro-SDIS (0=smoke, 1=fire) with remap → [0=fire, 1=smoke] ----
    pyro_remap = {
        0: 1,  # smoke -> 1
        1: 0,  # fire  -> 0
    }

    sync_split(PYRO_IMAGES_TRAIN, PYRO_LABELS_TRAIN,
               OUT_IMG_TRAIN, OUT_LBL_TRAIN, pyro_remap)
    sync_split(PYRO_IMAGES_VAL, PYRO_LABELS_VAL,
               OUT_IMG_VAL, OUT_LBL_VAL, pyro_remap)

    print("✅ Combined dataset prepared at:", OUT_ROOT)


if __name__ == "__main__":
    main()
