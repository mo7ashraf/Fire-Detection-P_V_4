#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# Ensure project root is on sys.path so 'src.*' imports work when running this file directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.tracker_iou import IOUTracker  # noqa: E402
from src.rules import should_alert  # noqa: E402


def preprocess(frame, sz: int):
    H, W = frame.shape[:2]
    scale = min(sz / W, sz / H)
    nw, nh = int(W * scale), int(H * scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((sz, sz, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized
    x = canvas[:, :, ::-1].astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    return x


def xywh_to_xyxy(xc, yc, w, h):
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return (x1, y1, x2, y2)


def main(a):
    cap = cv2.VideoCapture(0 if a.source == "0" else a.source)
    assert cap.isOpened()

    sess = ort.InferenceSession(
        a.onnx,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    tracker = IOUTracker(0.3, 30, 3)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        x = preprocess(frame, a.img)
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]

        dets = []
        for row in out:
            xc, yc, w, h, conf, cls = row.tolist()
            if conf < a.conf:
                continue
            dets.append((int(cls), conf, xywh_to_xyxy(xc, yc, w, h)))

        tracks = tracker.update(dets)

        for tr in tracks:
            color = (0, 255, 0)
            label = f"id{tr.id} c{tr.cls} h{tr.hits}"
            if should_alert(tr, a.min_persist, a.min_growth):
                color = (0, 0, 255)
                label += " ALERT"
            x1, y1, x2, y2 = map(int, tr.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 5)), 0, 0.5, color, 1)

        cv2.imshow("Fire/Smoke ONNX", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--source", default="0")
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min_persist", type=int, default=8)
    ap.add_argument("--min_growth", type=float, default=0.15)
    main(ap.parse_args())

