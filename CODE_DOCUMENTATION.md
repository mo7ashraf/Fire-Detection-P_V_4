# Code Documentation (R2)

## Layout
```
configs/
src/
  train_yolov10.py
  export_onnx.py
  run_eval_export.py
  infer_onnx_tracker.py
  update_word.py
  update_latex.py
  utils/tracker_iou.py
  rules.py
data/   models/   runs/
```

## Environments
- Python 3.10/3.11, PyTorch CUDA 12.1, onnxruntime-gpu
- Windows DataLoader: `--workers 0`

## Training
```powershell
python src\train_yolov10.py --data configs\dataset.yaml --model yolov10n.pt --img 640 --epochs 50 --batch 8 --workers 0 --name exp_y10n_t1000
```

## Export & Realtime
```powershell
python src\export_onnx.py --weights runs\detect\EXP\weights\best.pt --img 640 --out models\best.onnx
python src\infer_onnx_tracker.py --onnx models\best.onnx --source 0 --img 640 --conf 0.25 --min_persist 8 --min_growth 0.15
```

## Metrics Export and Document Updates
```powershell
python src\run_eval_export.py --run runs\detect\EXP --out out
python src\update_word.py  --doc paper\FireSmoke_Professional_Rich.docx --metrics out\metrics_summary.json --pr out\pr_curve.png
python src\update_latex.py --latex_dir paper\latex               --metrics out\metrics_summary.json
```

## Troubleshooting
- Absolute `path:` in YAML or set `yolo settings datasets_dir`.
- Keep `--workers 0` on Windows.
- Verify CUDA providers for torch and ORT.
