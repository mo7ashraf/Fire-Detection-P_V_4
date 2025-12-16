المسارات اللى جوا بتبقى مرتبطه بالمسار اللى بنعمل منه تشغيل الاسكريبت
cd .\src\scripts\efficient_net_b0\
(venv_fire_detect) PS D:\Master\Updated Work\Proposed Paper\Paper V3_1\FireDetect_Pro_Suite_20250904_1102\Paper_V4_Code\src\scripts\efficient_net_b0> python .\build_effnet_dataset_from_yaml.py
Dataset YAML being used: ../../../configs/fire_smoke_combined_new.yaml
Build train: 100%|█████████████████████████████████████████████████████████| 46758/46758 [32:21<00:00, 24.08it/s]
Build val: 100%|█████████████████████████████████████████████████████████████| 4099/4099 [02:40<00:00, 25.61it/s]

✅ Done. Dataset created at: ../../../runs/detect/data_effnet
[train] stats: {'fire': 11807, 'smoke': 37710, 'background': 91547, 'hardneg': 6709, 'randbg': 84838, 'images': 46758}
[val] stats: {'fire': 0, 'smoke': 3942, 'background': 9167, 'hardneg': 1205, 'randbg': 7962, 'images': 4099}  
Next: python src/scripts/train_effnet_b0_verifier.py

دلا بتجهز فولدر بيحتوى على صور لبدء تدريب الموديل والصور دى من الداتاست اللى جهزناها
لتدريب موديل اليولو وبتحط الداتا داخل جوا الrun\data_effnet
