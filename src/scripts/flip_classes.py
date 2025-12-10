from pathlib import Path

LABELS_ROOT = Path("../../dataset/fire_smoke_combined/labels")
SPLITS = ["train", "val", "test"]

FLIP = {0: 1, 1: 0}  # swap fire ↔ smoke

def flip_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()

    new_lines = []
    for line in lines:
        parts = line.split()
        cls = int(parts[0])
        x, y, w, h = parts[1:]
        new_cls = FLIP[cls]
        new_lines.append(f"{new_cls} {x} {y} {w} {h}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))


def main():
    for split in SPLITS:
        split_dir = LABELS_ROOT / split
        if not split_dir.exists():
            continue

        for file in split_dir.glob("*.txt"):
            flip_file(file)

    print("✅ Finished flipping classes (0 ↔ 1).")


if __name__ == "__main__":
    main()
