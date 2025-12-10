from pathlib import Path

LABELS_ROOT = Path("../../dataset/fire_smoke_combined_new/labels")
SPLITS = ["train", "val", "test"]
VALID_CLASSES = {0, 1}


def clean_label_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    new_lines = []
    removed = 0

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            removed += 1
            continue

        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            removed += 1
            continue

        # check class
        if cls not in VALID_CLASSES:
            removed += 1
            continue

        # check ranges
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            removed += 1
            continue
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            removed += 1
            continue

        new_lines.append(f"{cls} {x} {y} {w} {h}")

    # overwrite file
    with open(path, "w", encoding="utf-8") as f:
        if new_lines:
            f.write("\n".join(new_lines))
        else:
            # leave empty file -> treated as image with no objects
            f.write("")

    return len(lines), removed


def main():
    total_files = 0
    total_lines = 0
    total_removed = 0

    for split in SPLITS:
        split_dir = LABELS_ROOT / split
        if not split_dir.exists():
            print(f"⚠️ Split '{split}' not found, skipping.")
            continue

        print(f"\n=== Cleaning {split_dir} ===")
        txt_files = list(split_dir.glob("*.txt"))
        print(f"Found {len(txt_files)} label files")

        for lbl_file in txt_files:
            n_lines, n_removed = clean_label_file(lbl_file)
            total_files += 1
            total_lines += n_lines
            total_removed += n_removed

    print("\n✅ Cleaning finished.")
    print(f"Total label files processed: {total_files}")
    print(f"Total lines seen:           {total_lines}")
    print(f"Total invalid lines removed:{total_removed}")


if __name__ == "__main__":
    main()
