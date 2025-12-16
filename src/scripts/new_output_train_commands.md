## Clean Labels

(venv_fire_detect) PS D:\Master\Updated Work\Proposed Paper\Paper V3_1\FireDetect_Pro_Suite_20250904_1102\Paper_V4_Code\src\scripts> python clean_labels_new.py

=== Cleaning ..\..\dataset\fire_smoke_combined_new\labels\train ===
Found 46758 label files

=== Cleaning ..\..\dataset\fire_smoke_combined_new\labels\val ===
Found 7543 label files

=== Cleaning ..\..\dataset\fire_smoke_combined_new\labels\test ===
Found 4306 label files

✅ Cleaning finished.
Total label files processed: 58607
Total lines seen: 62906
Total invalid lines removed:2

## Validate Labels

(venv_fire_detect) PS D:\Master\Updated Work\Proposed Paper\Paper V3_1\FireDetect_Pro_Suite_20250904_1102\Paper_V4_Code\src\scripts> python validate_labels_new.py

=== Checking ..\..\dataset\fire_smoke_combined_new\labels\train ===
Found 46758 label files

=== Checking ..\..\dataset\fire_smoke_combined_new\labels\val ===
Found 7543 label files

=== Checking ..\..\dataset\fire_smoke_combined_new\labels\test ===
Found 4306 label files

✅ All checked labels look OK (format & ranges).

## Visualize Labels

Done

## Train Model

(venv_fire_detect) PS D:\Master\Updated Work\Proposed Paper\Paper V3_1\FireDetect_Pro_Suite_20250904_1102\Paper_V4_Code\src\scripts> python .\train_yolov10n_last.py
Dataset YAML being used: ../../configs/fire_smoke_combined_new.yaml
New https://pypi.org/project/ultralytics/8.3.238 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.0 🚀 Python-3.10.18 torch-2.3.1+cu121 CUDA:0 (Quadro T1000 with Max-Q Design, 4096MiB)
engine\trainer: task=detect, mode=train, model=yolov10n.pt, data=..\..\configs\fire_smoke_combined_new.resolved.yaml, epochs=100, time=None, patience=100, batch=8, imgsz=640, save=True, save_period=-1, cache=False, device=None, workers=2, project=../../runs/detect, name=exp_y10n_combined5, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=42, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=False, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=4, nms=False, lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=..\..\runs\detect\exp_y10n_combined5
Overriding model.yaml nc=80 with nc=2

                   from  n    params  module                                       arguments

0 -1 1 464 ultralytics.nn.modules.conv.Conv [3, 16, 3, 2]

1 -1 1 4672 ultralytics.nn.modules.conv.Conv [16, 32, 3, 2]

2 -1 1 7360 ultralytics.nn.modules.block.C2f [32, 32, 1, True]

3 -1 1 18560 ultralytics.nn.modules.conv.Conv [32, 64, 3, 2]

4 -1 2 49664 ultralytics.nn.modules.block.C2f [64, 64, 2, True]

5 -1 1 9856 ultralytics.nn.modules.block.SCDown [64, 128, 3, 2]

6 -1 2 197632 ultralytics.nn.modules.block.C2f [128, 128, 2, True]

7 -1 1 36096 ultralytics.nn.modules.block.SCDown [128, 256, 3, 2]

8 -1 1 460288 ultralytics.nn.modules.block.C2f [256, 256, 1, True]

9 -1 1 164608 ultralytics.nn.modules.block.SPPF [256, 256, 5]

10 -1 1 249728 ultralytics.nn.modules.block.PSA [256, 256]

11 -1 1 0 torch.nn.modules.upsampling.Upsample [None, 2, 'nearest']

12 [-1, 6] 1 0 ultralytics.nn.modules.conv.Concat [1]

13 -1 1 148224 ultralytics.nn.modules.block.C2f [384, 128, 1]

14 -1 1 0 torch.nn.modules.upsampling.Upsample [None, 2, 'nearest']

15 [-1, 4] 1 0 ultralytics.nn.modules.conv.Concat [1]

16 -1 1 37248 ultralytics.nn.modules.block.C2f [192, 64, 1]

17 -1 1 36992 ultralytics.nn.modules.conv.Conv [64, 64, 3, 2]

18 [-1, 13] 1 0 ultralytics.nn.modules.conv.Concat [1]

19 -1 1 123648 ultralytics.nn.modules.block.C2f [192, 128, 1]

20 -1 1 18048 ultralytics.nn.modules.block.SCDown [128, 128, 3, 2]

21 [-1, 10] 1 0 ultralytics.nn.modules.conv.Concat [1]

22 -1 1 282624 ultralytics.nn.modules.block.C2fCIB [384, 256, 1, True, True]

23 [16, 19, 22] 1 862108 ultralytics.nn.modules.head.v10Detect [2, [64, 128, 256]]

YOLOv10n summary: 385 layers, 2,707,820 parameters, 2,707,804 gradients, 8.4 GFLOPs

Transferred 493/595 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
train: Scanning D:\Master\Updated Work\Proposed Paper\Paper V3_1\FireDetect_Pro_Suite_20250904_1102\Paper_V4_Co

Class Images Instances Box(P R mAP50 mAP50-95): 100%|██████████| 472/
all 7543 8206 0.197 0.186 0.13 0.0578

Epoch GPU_mem box_loss cls_loss dfl_loss Instances Size 1/100 2.93G 3.616 8.263 2.534 2 640: 100%|██████████| 5415/5415 [31:2 Class Images Instances Box(P R mAP50 mAP50-95): 100%|██████████| 472/472 [01:15<00:00, 6.22it/s] all 7543 8206 0.182 0.177 0.103 0.0452
Epoch GPU_mem box_loss cls_loss dfl_loss Instances Size
3/100 2.93G 3.952 4.369 2.648 22 640: 3%|▎ | 179/5415 [00:58
Epoch GPU_mem box_loss cls_loss dfl_loss Instances Size
3/100 2.99G 3.916 4.324 2.715 2 640: 100%|██████████| 5415/5415 [16:2
Class Images Instances Box(P R mAP50 mAP50-95): 100%|██████████| 472/
all 7543 8206 0.177 0.178 0.117 0.0533

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/100      3.03G      3.821      4.097      2.713          3        640: 100%|██████████| 5415/5415 [29:3
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 472/
                   all       7543       8206      0.226      0.202       0.15     0.0706

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/100      2.99G      3.644      3.756      2.613          3        640: 100%|██████████| 5415/5415 [29:3
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 472/
                   all       7543       8206      0.237      0.211       0.17     0.0853

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/100      2.99G      3.551      3.541      2.572          3        640: 100%|██████████| 5415/5415 [29:3
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 472/
                   all       7543       8206      0.235      0.216      0.168      0.088

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/100      2.99G      3.469      3.411      2.527          3        640: 100%|██████████| 5415/5415 [29:3
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 472/
                   all       7543       8206      0.247      0.215      0.174     0.0923

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/100      2.99G      3.426      3.299      2.513          3        640: 100%|██████████| 5415/5415 [29:3
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 472/
                   all       7543       8206      0.245      0.215       0.17     0.0912

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/100         3G       3.38      3.193      2.483          4        640: 100%|██████████| 5415/5415 [29:3
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 472/
                   all       7543       8206      0.252      0.224      0.172     0.0926

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/100      2.99G      3.343      3.132      2.459          2        640: 100%|██████████| 5415/5415 [29:3
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 472/
                   all       7543       8206      0.243      0.226      0.169     0.0925
