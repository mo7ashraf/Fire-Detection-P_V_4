#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO
import yaml


def _resolve_dataset_yaml_to_file(data_arg: str) -> str:
    print("Dataset YAML being used:", data_arg)
    p = Path(data_arg)
    if not p.suffix.lower() in {".yaml", ".yml"} or not p.exists():
        return data_arg

    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    repo_root = Path(__file__).resolve().parent.parent
    path_entry = cfg.get("path", "")
    dataset_root = (repo_root / path_entry).resolve() if path_entry else repo_root

    def _resolve(entry: str | None) -> str | None:
        if not entry:
            return None
        ep = Path(entry)
        return str((dataset_root / ep).resolve()) if not ep.is_absolute() else str(ep)

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../configs/fire_smoke_combined_new.yaml")
    ap.add_argument("--model", default="yolov10n.pt")  # or point to your best.pt *later*
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--name", default="exp_y10n_combined")
    a = ap.parse_args()

    data_cfg = _resolve_dataset_yaml_to_file(a.data)

    m = YOLO(a.model)
    m.train(
        data=data_cfg,
        imgsz=a.img,
        epochs=a.epochs,
        batch=a.batch,
        seed=42,
        lr0=0.001,          # safer for combined dataset
        lrf=0.01,
        weight_decay=0.0005,
        mosaic=1.0,
        mixup=0.0,
        fliplr=0.5,
        workers=a.workers,
        amp=False,          # 🔴 avoid NaNs
        optimizer="auto",   # let Ultralytics choose AdamW etc.
        project="../../runs/detect",
        name=a.name,
    )


if __name__ == "__main__":
    main()
