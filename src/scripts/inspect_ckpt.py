# inspect_ckpt.py
import torch
p = r'../../runs\detect/exp_y10n_combined3/weights/last.pt'
ck = torch.load(p, map_location='cpu')
print(type(ck))
print('KEYS:', list(ck.keys()) if isinstance(ck, dict) else 'not-dict')
print('EPOCH:', ck.get('epoch', ck.get('epochs', 'N/A')) if isinstance(ck, dict) else 'N/A')
print('HAS_OPTIMIZER:', 'optimizer' in ck if isinstance(ck, dict) else 'N/A')