from pathlib import Path

LABELS_ROOT = Path("../../dataset/fire_smoke_combined_new/labels")  # adjust if needed
SPLITS = ["train", "val", "test"]  # 'test' is optional

VALID_CLASSES = {0, 1}


def check_label_file(path: Path):
    errors = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for i, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"Line {i}: expected 5 values, got {len(parts)} -> '{line}'")
            continue

        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"Line {i}: non-numeric value -> '{line}'")
            continue

        if cls not in VALID_CLASSES:
            errors.append(f"Line {i}: invalid class id {cls}, allowed {VALID_CLASSES}")

        for name, v in zip(["x", "y", "w", "h"], [x, y, w, h]):
            if not (0.0 <= v <= 1.0):
                errors.append(f"Line {i}: {name}={v} out of [0,1] in '{line}'")

        if w <= 0 or h <= 0:
            errors.append(f"Line {i}: non-positive width/height w={w}, h={h} in '{line}'")

    return errors


def main():
    any_errors = False
    for split in SPLITS:
        split_dir = LABELS_ROOT / split
        if not split_dir.exists():
            print(f"⚠️  Split '{split}' not found, skipping.")
            continue

        print(f"\n=== Checking {split_dir} ===")
        txt_files = list(split_dir.glob("*.txt"))
        print(f"Found {len(txt_files)} label files")

        for lbl_file in txt_files:
            errors = check_label_file(lbl_file)
            if errors:
                any_errors = True
                print(f"\n❌ Problems in: {lbl_file}")
                for e in errors[:10]:
                    print("   -", e)
                if len(errors) > 10:
                    print(f"   ... and {len(errors)-10} more issues in this file")

    if not any_errors:
        print("\n✅ All checked labels look OK (format & ranges).")


if __name__ == "__main__":
    main()
