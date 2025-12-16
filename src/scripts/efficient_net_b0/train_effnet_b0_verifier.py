#!/usr/bin/env python3
import argparse
import json
import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm


def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_cm(cm):
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    f1 = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision[c] = tp / (tp + fp + 1e-12)
        recall[c] = tp / (tp + fn + 1e-12)
        f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c] + 1e-12)

    acc = np.trace(cm) / (cm.sum() + 1e-12)
    macro_f1 = f1.mean()
    return acc, precision, recall, f1, macro_f1


def main():
    ap = argparse.ArgumentParser()

    # ✅ Defaults you requested
    ap.add_argument("--data_dir", default="../../../runs/detect/data_effnet")
    ap.add_argument("--out_dir", default="../../../runs/detect/effnet_verifier")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    # ✅ Make AMP ON by default
    ap.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    ap.add_argument("--freeze_backbone_epochs", type=int, default=2)
    args = ap.parse_args()

    use_amp = not args.no_amp

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    normalize = weights.transforms().transforms[-1]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    if num_classes != 3:
        raise ValueError(f"Expected 3 classes (background/fire/smoke). Found: {class_to_idx}")

    with open(os.path.join(args.out_dir, "class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    def set_backbone_trainable(trainable: bool):
        for p in model.features.parameters():
            p.requires_grad = trainable

    # class weights
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y in train_ds.samples:
        counts[y] += 1
    wts = counts.sum() / (counts + 1e-12)
    wts = (wts / wts.mean()).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(wts, device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_macro_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if epoch <= args.freeze_backbone_epochs:
            set_backbone_trainable(False)
        else:
            set_backbone_trainable(True)

        # Train
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            tr_correct += (preds == y).sum().item()
            tr_total += x.size(0)
            pbar.set_postfix(loss=float(loss.item()))

        scheduler.step()
        tr_loss /= max(1, tr_total)
        tr_acc = tr_correct / max(1, tr_total)

        # Val
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)

                val_loss += loss.item() * x.size(0)
                preds = torch.argmax(logits, dim=1)

                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        val_loss /= max(1, len(val_ds))
        cm = confusion_matrix_np(y_true, y_pred, num_classes)
        val_acc, prec, rec, f1, macro_f1 = metrics_from_cm(cm)

        row = {
            "epoch": epoch,
            "train_loss": float(tr_loss),
            "train_acc": float(tr_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "macro_f1": float(macro_f1),
            "cm": cm.tolist(),
            "class_to_idx": class_to_idx,
            "seconds": float(time.time() - t0),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        print(
            f"\nEpoch {epoch}: train_acc={tr_acc:.4f} val_acc={val_acc:.4f} macro_f1={macro_f1:.4f} "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f}"
        )

        # Save best
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            ckpt = {
                "model": model.state_dict(),
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "best_macro_f1": best_macro_f1,
                "epoch": epoch,
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            with open(os.path.join(args.out_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)
            print("✅ Saved new best checkpoint:", os.path.join(args.out_dir, "best.pt"))

        with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("\nDone. Best macro_f1 =", best_macro_f1)
    print("Best checkpoint:", os.path.join(args.out_dir, "best.pt"))


if __name__ == "__main__":
    main()

'''#!/usr/bin/env python3
import argparse
import json
import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm


def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_cm(cm):
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    f1 = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision[c] = tp / (tp + fp + 1e-12)
        recall[c] = tp / (tp + fn + 1e-12)
        f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c] + 1e-12)

    acc = np.trace(cm) / (cm.sum() + 1e-12)
    macro_f1 = f1.mean()
    return acc, precision, recall, f1, macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_effnet", help="root with train/ and val/")
    ap.add_argument("--out_dir", default="runs/effnet_verifier", help="save checkpoints/metrics")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--freeze_backbone_epochs", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    # Use ImageNet normalization
    normalize = weights.transforms().transforms[-1]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    if num_classes != 3:
        raise ValueError(f"Expected 3 classes (background/fire/smoke). Found: {class_to_idx}")

    with open(os.path.join(args.out_dir, "class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    def set_backbone_trainable(trainable: bool):
        for p in model.features.parameters():
            p.requires_grad = trainable

    # class weights (helps if background is huge)
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y in train_ds.samples:
        counts[y] += 1
    wts = counts.sum() / (counts + 1e-12)
    wts = (wts / wts.mean()).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(wts, device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    best_macro_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if epoch <= args.freeze_backbone_epochs:
            set_backbone_trainable(False)
        else:
            set_backbone_trainable(True)

        # -------- Train --------
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            tr_correct += (preds == y).sum().item()
            tr_total += x.size(0)

            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        tr_loss /= max(1, tr_total)
        tr_acc = tr_correct / max(1, tr_total)

        # -------- Val --------
        model.eval()
        val_loss = 0.0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)

                val_loss += loss.item() * x.size(0)
                preds = torch.argmax(logits, dim=1)

                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        val_loss /= max(1, len(val_ds))
        cm = confusion_matrix_np(y_true, y_pred, num_classes)
        val_acc, prec, rec, f1, macro_f1 = metrics_from_cm(cm)

        row = {
            "epoch": epoch,
            "train_loss": float(tr_loss),
            "train_acc": float(tr_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "macro_f1": float(macro_f1),
            "cm": cm.tolist(),
            "class_to_idx": class_to_idx,
            "seconds": float(time.time() - t0),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        print(
            f"\nEpoch {epoch}: train_acc={tr_acc:.4f} val_acc={val_acc:.4f} macro_f1={macro_f1:.4f} "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f}"
        )

        # save best
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            ckpt = {
                "model": model.state_dict(),
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "best_macro_f1": best_macro_f1,
                "epoch": epoch,
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            with open(os.path.join(args.out_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)
            print("✅ Saved new best checkpoint:", os.path.join(args.out_dir, "best.pt"))

        with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("\nDone. Best macro_f1 =", best_macro_f1)
    print("Best checkpoint:", os.path.join(args.out_dir, "best.pt"))


if __name__ == "__main__":
    main()
'''