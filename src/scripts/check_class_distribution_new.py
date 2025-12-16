# src/scripts/check_class_distribution.py

from pathlib import Path
from collections import Counter

# Labels root relative to this script (assuming you run from repo root)
LABELS_ROOT = Path("../../dataset/fire_smoke_combined_new/labels")


def count_classes_in_dir(lbl_dir: Path, num_classes=2):
    counts = Counter()
    files = list(lbl_dir.glob("*.txt"))
    for lbl_path in files:
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                if 0 <= cls < num_classes:
                    counts[cls] += 1
    return counts, len(files)


def main():
    splits = ["train", "val", "test"]

    total_counts = Counter()
    total_files = 0

    for split in splits:
        split_dir = LABELS_ROOT / split
        if not split_dir.exists():
            print(f"⚠️  Split '{split}' not found, skipping.")
            continue

        counts, n_files = count_classes_in_dir(split_dir, num_classes=2)
        total_counts.update(counts)
        total_files += n_files

        print(f"\n=== {split.upper()} ===")
        print(f"Label files: {n_files}")
        print(f"fire (0):  {counts.get(0, 0)}")
        print(f"smoke (1): {counts.get(1, 0)}")

    print("\n=== TOTAL (all splits found) ===")
    print(f"Total label files: {total_files}")
    print(f"fire (0):  {total_counts.get(0, 0)}")
    print(f"smoke (1): {total_counts.get(1, 0)}")

    if total_counts.get(0, 0) and total_counts.get(1, 0):
        ratio = total_counts[0] / total_counts[1]
        print(f"\nfire/smoke ratio: {ratio:.3f}")


if __name__ == "__main__":
    main()
