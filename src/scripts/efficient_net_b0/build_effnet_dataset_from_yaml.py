#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from tqdm import tqdm
import yaml
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_dataset_yaml_to_file(data_arg: str) -> str:
    print("Dataset YAML being used:", data_arg)
    p = Path(data_arg)
    if p.suffix.lower() not in {".yaml", ".yml"} or not p.exists():
        return data_arg

    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    repo_root = Path(__file__).resolve().parent.parent  # src/
    path_entry = cfg.get("path", "")
    dataset_root = (repo_root / path_entry).resolve() if path_entry else repo_root

    def _resolve(entry: Any) -> Any:
        if entry is None or entry == "":
            return None
        if isinstance(entry, list):
            out = []
            for e in entry:
                ep = Path(e)
                out.append(str((dataset_root / ep).resolve()) if not ep.is_absolute() else str(ep))
            return out
        if isinstance(entry, str):
            ep = Path(entry)
            return str((dataset_root / ep).resolve()) if not ep.is_absolute() else str(ep)
        return entry

    resolved_cfg: Dict[str, Any] = {}
    train_abs = _resolve(cfg.get("train"))
    val_abs = _resolve(cfg.get("val"))
    test_abs = _resolve(cfg.get("test")) if cfg.get("test") else None

    if train_abs:
        resolved_cfg["train"] = train_abs
    if val_abs:
        resolved_cfg["val"] = val_abs
    if test_abs:
        resolved_cfg["test"] = test_abs
    if "names" in cfg:
        resolved_cfg["names"] = cfg["names"]
    if "nc" in cfg:
        resolved_cfg["nc"] = cfg["nc"]

    if not (resolved_cfg.get("train") and resolved_cfg.get("val")):
        return data_arg

    out_path = p.with_name(p.stem + ".resolved.yaml")
    out_path.write_text(yaml.safe_dump(resolved_cfg, sort_keys=False), encoding="utf-8")
    return str(out_path)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def expand_box(x1, y1, x2, y2, pad_ratio: float, w: int, h: int):
    bw = x2 - x1
    bh = y2 - y1
    pad_w = bw * pad_ratio
    pad_h = bh * pad_ratio
    nx1 = clamp(int(math.floor(x1 - pad_w)), 0, w - 1)
    ny1 = clamp(int(math.floor(y1 - pad_h)), 0, h - 1)
    nx2 = clamp(int(math.ceil(x2 + pad_w)), 0, w - 1)
    ny2 = clamp(int(math.ceil(y2 + pad_h)), 0, h - 1)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return nx1, ny1, nx2, ny2


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def read_yolo_label_file(label_path: str) -> List[List[float]]:
    if not os.path.exists(label_path):
        return []
    out = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])
            out.append([cls, xc, yc, bw, bh])
    return out


def yolo_norm_to_xyxy(lbl, w: int, h: int):
    cls, xc, yc, bw, bh = lbl
    x1 = (xc - bw / 2) * w
    y1 = (yc - bh / 2) * h
    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h
    return cls, x1, y1, x2, y2


def save_patch(img, xyxy, out_path: str, size: int = 224) -> bool:
    x1, y1, x2, y2 = xyxy
    patch = img[y1:y2, x1:x2]
    if patch is None or patch.size == 0:
        return False
    patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.imwrite(out_path, patch)


def map_image_to_label(img_path: str) -> str:
    p = Path(img_path)
    parts = list(p.parts)
    try:
        i = parts.index("images")
        parts[i] = "labels"
        return str(Path(*parts).with_suffix(".txt"))
    except ValueError:
        return str(p.with_suffix(".txt")).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)


def collect_image_paths(entry: Any) -> List[str]:
    paths: List[str] = []
    if isinstance(entry, list):
        for e in entry:
            paths.extend(collect_image_paths(e))
        return paths

    if not isinstance(entry, str):
        return paths

    p = Path(entry)
    if p.is_dir():
        for f in p.rglob("*"):
            if f.suffix.lower() in IMG_EXTS:
                paths.append(str(f))
        paths.sort()
        return paths

    if p.is_file():
        if p.suffix.lower() in IMG_EXTS:
            return [str(p)]
        if p.suffix.lower() == ".txt":
            lines = p.read_text(encoding="utf-8").splitlines()
            for line in lines:
                line = line.strip()
                if line:
                    paths.append(str(Path(line).resolve()) if Path(line).exists() else line)
            return paths

    return paths


def main():
    ap = argparse.ArgumentParser()

    # ✅ Defaults you requested
    ap.add_argument("--data", default="../../../configs/fire_smoke_combined_new.yaml")
    ap.add_argument("--yolo_weights", default="../../../runs/detect/exp_y10n_combined3/weights/best.pt")
    ap.add_argument("--out_dir", default="../../../runs/detect/data_effnet")

    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--pad_ratio", type=float, default=0.15)
    ap.add_argument("--conf_pred", type=float, default=0.20)
    ap.add_argument("--iou_match", type=float, default=0.30)
    ap.add_argument("--bg_random_per_image", type=int, default=2)
    ap.add_argument("--max_hardneg_per_image", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    resolved_yaml = resolve_dataset_yaml_to_file(args.data)
    #D:\Master\Updated Work\Proposed Paper\Paper V3_1\FireDetect_Pro_Suite_20250904_1102\Paper_V4_Code\configs\fire_smoke_combined_new.resolved.yaml
    cfg = yaml.safe_load(Path(resolved_yaml).read_text(encoding="utf-8")) or {}

    # Expected YOLO class mapping: 0 fire, 1 smoke
    CLASS_NAMES = {0: "fire", 1: "smoke"}
    BG_NAME = "background"

    # output dirs
    for split in ["train", "val"]:
        for cname in ["fire", "smoke", "background"]:
            ensure_dir(os.path.join(args.out_dir, split, cname))

    yolo = YOLO(args.yolo_weights)

    def process_split(split_name: str, split_entry: Any):
        img_files = collect_image_paths(split_entry)
        stats = {"fire": 0, "smoke": 0, "background": 0, "hardneg": 0, "randbg": 0, "images": len(img_files)}
        if not img_files:
            print(f"⚠️ No images found for {split_name} from entry:", split_entry)
            return stats

        for img_path in tqdm(img_files, desc=f"Build {split_name}"):
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            stem = Path(img_path).stem

            label_path = map_image_to_label(img_path)
            gt = read_yolo_label_file(label_path)

            gt_boxes = []
            # GT crops
            for i, lbl in enumerate(gt):
                cls, x1, y1, x2, y2 = yolo_norm_to_xyxy(lbl, w, h)
                if cls not in CLASS_NAMES:
                    continue
                exp = expand_box(x1, y1, x2, y2, args.pad_ratio, w, h)
                if exp is None:
                    continue
                cname = CLASS_NAMES[cls]
                out_name = f"{stem}_gt_{i}.jpg"
                out_path = os.path.join(args.out_dir, split_name, cname, out_name)
                if save_patch(img, exp, out_path, size=args.img_size):
                    stats[cname] += 1
                gt_boxes.append([int(exp[0]), int(exp[1]), int(exp[2]), int(exp[3])])

            # Random background crops
            for j in range(args.bg_random_per_image):
                bw = random.randint(max(32, w // 10), max(64, w // 3))
                bh = random.randint(max(32, h // 10), max(64, h // 3))
                x1 = random.randint(0, max(0, w - bw - 1))
                y1 = random.randint(0, max(0, h - bh - 1))
                x2 = x1 + bw
                y2 = y1 + bh
                box = [x1, y1, x2, y2]

                ok = True
                for gb in gt_boxes:
                    if iou_xyxy(box, gb) > 0.05:
                        ok = False
                        break
                if not ok:
                    continue

                out_name = f"{stem}_bg_{j}.jpg"
                out_path = os.path.join(args.out_dir, split_name, BG_NAME, out_name)
                if save_patch(img, box, out_path, size=args.img_size):
                    stats["background"] += 1
                    stats["randbg"] += 1

            # Hard negatives from YOLO false positives
            hardneg_count = 0
            try:
                res = yolo.predict(img_path, conf=args.conf_pred, verbose=False)
            except Exception:
                continue

            if res and len(res) > 0 and res[0].boxes is not None:
                boxes = res[0].boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)

                for k in range(len(xyxy)):
                    if hardneg_count >= args.max_hardneg_per_image:
                        break
                    pred_cls = int(clss[k])
                    if pred_cls not in CLASS_NAMES:
                        continue

                    px1, py1, px2, py2 = xyxy[k]
                    exp = expand_box(px1, py1, px2, py2, args.pad_ratio, w, h)
                    if exp is None:
                        continue
                    pbox = [exp[0], exp[1], exp[2], exp[3]]

                    best_iou = 0.0
                    for gb in gt_boxes:
                        best_iou = max(best_iou, iou_xyxy(pbox, gb))

                    if best_iou < args.iou_match:
                        out_name = f"{stem}_hardneg_{hardneg_count}_c{pred_cls}_p{confs[k]:.2f}.jpg"
                        out_path = os.path.join(args.out_dir, split_name, BG_NAME, out_name)
                        if save_patch(img, pbox, out_path, size=args.img_size):
                            stats["background"] += 1
                            stats["hardneg"] += 1
                            hardneg_count += 1

        with open(os.path.join(args.out_dir, f"stats_{split_name}.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        return stats

    train_stats = process_split("train", cfg.get("train"))
    val_stats = process_split("val", cfg.get("val"))

    print("\n✅ Done. Dataset created at:", args.out_dir)
    print("[train] stats:", train_stats)
    print("[val]   stats:", val_stats)
    print("Next: python src/scripts/train_effnet_b0_verifier.py")


if __name__ == "__main__":
    main()

'''#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import yaml
from ultralytics import YOLO


# -----------------------------
# YAML resolver (same spirit as yours)
# -----------------------------
def resolve_dataset_yaml_to_file(data_arg: str) -> str:
    print("Dataset YAML being used:", data_arg)
    p = Path(data_arg)
    if p.suffix.lower() not in {".yaml", ".yml"} or not p.exists():
        return data_arg

    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    repo_root = Path(__file__).resolve().parent.parent.parent  # src/
    path_entry = cfg.get("path", "")
    dataset_root = (repo_root / path_entry).resolve() if path_entry else repo_root

    def _resolve(entry: Any) -> Any:
        if entry is None or entry == "":
            return None
        # Ultralytics allows train/val as:
        # - string path
        # - list of paths
        # - .txt file listing images
        if isinstance(entry, list):
            return [str((dataset_root / Path(e)).resolve()) if not Path(e).is_absolute() else str(Path(e)) for e in entry]
        if isinstance(entry, str):
            ep = Path(entry)
            return str((dataset_root / ep).resolve()) if not ep.is_absolute() else str(ep)
        return entry

    resolved_cfg: Dict[str, Any] = {}
    train_abs = _resolve(cfg.get("train"))
    val_abs = _resolve(cfg.get("val"))
    test_abs = _resolve(cfg.get("test")) if cfg.get("test") else None

    if train_abs:
        resolved_cfg["train"] = train_abs
    if val_abs:
        resolved_cfg["val"] = val_abs
    if test_abs:
        resolved_cfg["test"] = test_abs
    if "names" in cfg:
        resolved_cfg["names"] = cfg["names"]
    if "nc" in cfg:
        resolved_cfg["nc"] = cfg["nc"]

    if not (resolved_cfg.get("train") and resolved_cfg.get("val")):
        return data_arg

    out_path = p.with_name(p.stem + ".resolved.yaml")
    out_path.write_text(yaml.safe_dump(resolved_cfg, sort_keys=False), encoding="utf-8")
    return str(out_path)


# -----------------------------
# Dataset helpers
# -----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def expand_box(x1, y1, x2, y2, pad_ratio: float, w: int, h: int):
    bw = x2 - x1
    bh = y2 - y1
    pad_w = bw * pad_ratio
    pad_h = bh * pad_ratio
    nx1 = clamp(int(math.floor(x1 - pad_w)), 0, w - 1)
    ny1 = clamp(int(math.floor(y1 - pad_h)), 0, h - 1)
    nx2 = clamp(int(math.ceil(x2 + pad_w)), 0, w - 1)
    ny2 = clamp(int(math.ceil(y2 + pad_h)), 0, h - 1)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return nx1, ny1, nx2, ny2


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def read_yolo_label_file(label_path: str) -> List[List[float]]:
    if not os.path.exists(label_path):
        return []
    out = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])
            out.append([cls, xc, yc, bw, bh])
    return out


def yolo_norm_to_xyxy(lbl, w: int, h: int):
    cls, xc, yc, bw, bh = lbl
    x1 = (xc - bw / 2) * w
    y1 = (yc - bh / 2) * h
    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h
    return cls, x1, y1, x2, y2


def save_patch(img, xyxy, out_path: str, size: int = 224) -> bool:
    x1, y1, x2, y2 = xyxy
    patch = img[y1:y2, x1:x2]
    if patch is None or patch.size == 0:
        return False
    patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.imwrite(out_path, patch)


def map_image_to_label(img_path: str) -> str:
    """
    Standard YOLO layout:
      .../images/train/xxx.jpg  -> .../labels/train/xxx.txt
    If 'images' isn't present, it still tries best-effort replacing directory name.
    """
    p = Path(img_path)
    parts = list(p.parts)
    try:
        i = parts.index("images")
        parts[i] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
        return str(label_path)
    except ValueError:
        # fallback: sibling labels folder
        label_path = p.with_suffix(".txt")
        return str(label_path).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)


def collect_image_paths(entry: Any) -> List[str]:
    """
    entry can be:
      - directory
      - image file
      - .txt list file (one image path per line)
      - list of any of the above
    """
    paths: List[str] = []
    if isinstance(entry, list):
        for e in entry:
            paths.extend(collect_image_paths(e))
        return paths

    if not isinstance(entry, str):
        return paths

    p = Path(entry)
    if p.is_dir():
        for f in p.rglob("*"):
            if f.suffix.lower() in IMG_EXTS:
                paths.append(str(f))
        paths.sort()
        return paths

    if p.is_file():
        if p.suffix.lower() in IMG_EXTS:
            return [str(p)]
        if p.suffix.lower() == ".txt":
            # list file
            lines = p.read_text(encoding="utf-8").splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # allow relative paths inside list file
                lp = Path(line)
                paths.append(str(lp.resolve()) if lp.exists() else line)
            return paths

    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset YAML (same used for YOLO)")
    ap.add_argument("--yolo_weights", required=True, help="stage-1 weights (best.pt)")
    ap.add_argument("--out_dir", default="data_effnet", help="output root")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--pad_ratio", type=float, default=0.15)

    ap.add_argument("--conf_pred", type=float, default=0.20, help="YOLO conf for mining hard negatives")
    ap.add_argument("--iou_match", type=float, default=0.30, help="if pred IoU < iou_match with any GT -> hard negative")
    ap.add_argument("--bg_random_per_image", type=int, default=2, help="random background crops per image")
    ap.add_argument("--max_hardneg_per_image", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    resolved_yaml = resolve_dataset_yaml_to_file(args.data)
    cfg = yaml.safe_load(Path(resolved_yaml).read_text(encoding="utf-8")) or {}

    # Expected mapping from your detection dataset:
    # 0 fire, 1 smoke (same as your YAML names)
    CLASS_NAMES = {0: "fire", 1: "smoke"}
    BG_NAME = "background"

    # output dirs
    for split in ["train", "val"]:
        for cname in ["fire", "smoke", "background"]:
            ensure_dir(os.path.join(args.out_dir, split, cname))

    yolo = YOLO(args.yolo_weights)

    def process_split(split_name: str, split_entry: Any):
        img_files = collect_image_paths(split_entry)
        stats = {"fire": 0, "smoke": 0, "background": 0, "hardneg": 0, "randbg": 0, "images": len(img_files)}
        if not img_files:
            print(f"⚠️ No images found for {split_name} from entry:", split_entry)
            return stats

        for img_path in tqdm(img_files, desc=f"Build {split_name}"):
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            stem = Path(img_path).stem

            label_path = map_image_to_label(img_path)
            gt = read_yolo_label_file(label_path)

            gt_boxes = []
            # --- GT crops
            for i, lbl in enumerate(gt):
                cls, x1, y1, x2, y2 = yolo_norm_to_xyxy(lbl, w, h)
                if cls not in CLASS_NAMES:
                    continue
                exp = expand_box(x1, y1, x2, y2, args.pad_ratio, w, h)
                if exp is None:
                    continue
                cname = CLASS_NAMES[cls]
                out_name = f"{stem}_gt_{i}.jpg"
                out_path = os.path.join(args.out_dir, split_name, cname, out_name)
                if save_patch(img, exp, out_path, size=args.img_size):
                    stats[cname] += 1
                gt_boxes.append([int(exp[0]), int(exp[1]), int(exp[2]), int(exp[3])])

            # --- random background crops (low overlap with GT)
            for j in range(args.bg_random_per_image):
                bw = random.randint(max(32, w // 10), max(64, w // 3))
                bh = random.randint(max(32, h // 10), max(64, h // 3))
                x1 = random.randint(0, max(0, w - bw - 1))
                y1 = random.randint(0, max(0, h - bh - 1))
                x2 = x1 + bw
                y2 = y1 + bh
                box = [x1, y1, x2, y2]

                ok = True
                for gb in gt_boxes:
                    if iou_xyxy(box, gb) > 0.05:
                        ok = False
                        break
                if not ok:
                    continue

                out_name = f"{stem}_bg_{j}.jpg"
                out_path = os.path.join(args.out_dir, split_name, BG_NAME, out_name)
                if save_patch(img, box, out_path, size=args.img_size):
                    stats["background"] += 1
                    stats["randbg"] += 1

            # --- hard negatives from YOLO preds that don't match any GT
            hardneg_count = 0
            try:
                res = yolo.predict(img_path, conf=args.conf_pred, verbose=False)
            except Exception:
                continue

            if res and len(res) > 0 and res[0].boxes is not None:
                boxes = res[0].boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)

                for k in range(len(xyxy)):
                    if hardneg_count >= args.max_hardneg_per_image:
                        break
                    pred_cls = int(clss[k])
                    if pred_cls not in CLASS_NAMES:
                        continue

                    px1, py1, px2, py2 = xyxy[k]
                    exp = expand_box(px1, py1, px2, py2, args.pad_ratio, w, h)
                    if exp is None:
                        continue
                    pbox = [exp[0], exp[1], exp[2], exp[3]]

                    best_iou = 0.0
                    for gb in gt_boxes:
                        best_iou = max(best_iou, iou_xyxy(pbox, gb))

                    if best_iou < args.iou_match:
                        out_name = f"{stem}_hardneg_{hardneg_count}_c{pred_cls}_p{confs[k]:.2f}.jpg"
                        out_path = os.path.join(args.out_dir, split_name, BG_NAME, out_name)
                        if save_patch(img, pbox, out_path, size=args.img_size):
                            stats["background"] += 1
                            stats["hardneg"] += 1
                            hardneg_count += 1

        with open(os.path.join(args.out_dir, f"stats_{split_name}.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        return stats

    train_stats = process_split("train", cfg.get("train"))
    val_stats = process_split("val", cfg.get("val"))

    print("\n✅ Done. Dataset created at:", args.out_dir)
    print("[train] stats:", train_stats)
    print("[val]   stats:", val_stats)
    print("Next: train EfficientNet with train_effnet_b0_verifier.py")


if __name__ == "__main__":
    main()
'''