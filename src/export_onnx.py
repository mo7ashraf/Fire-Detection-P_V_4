#!/usr/bin/env python3
import argparse, os
from ultralytics import YOLO
ap = argparse.ArgumentParser(); ap.add_argument("--weights", required=True); ap.add_argument("--img", type=int, default=640); ap.add_argument("--out", default="models/best.onnx")
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
YOLO(a.weights).export(format="onnx", imgsz=a.img, opset=13, half=False, dynamic=False, device=0, save_dir=os.path.dirname(a.out))
