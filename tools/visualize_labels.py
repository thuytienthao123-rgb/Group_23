import argparse
from pathlib import Path
import sys
import io
import cv2
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_image(path: Path):
    """Safely load image with unicode path on Windows."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None

def save_image(path: Path, img):
    """Safely save image with unicode path on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _, buf = cv2.imencode(path.suffix if path.suffix else ".jpg", img)
    buf.tofile(str(path))

def draw_yolo_labels(image, label_path: Path, class_names=None):
    if class_names is None:
        class_names = {0: "GreenSM"}
        
    h, w = image.shape[:2]
    if not label_path.exists():
        return image, 0

    lines = label_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    box_count = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, bw, bh = [float(x) for x in parts[1:]]
        except ValueError:
            continue

        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w - 1, int((cx + bw / 2) * w))
        y2 = min(h - 1, int((cy + bh / 2) * h))

        name = class_names.get(cls_id, f"cls_{cls_id}")
        color = (255, 200, 0) # Cyan/Bgr color

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_text = f"{name} ({cls_id})"
        cv2.putText(image, label_text, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        box_count += 1

    return image, box_count

def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO labeled images.")
    parser.add_argument("--dataset", default="dataset_v2", help="Path to dataset root folder containing images/ and labels/")
    parser.add_argument("--output", default="preview_samples", help="Output directory to save sample visualized images")
    parser.add_argument("--num", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--all", action="store_true", help="Visualize all labeled images")
    args = parser.parse_args()

    root = Path(args.dataset)
    img_dir = root / "images"
    lbl_dir = root / "labels"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"Error: {root} must contain images/ and labels/ directories.")
        return

    label_files = sorted(lbl_dir.glob("*.txt"))
    non_empty = [p for p in label_files if p.stat().st_size > 0]
    
    print(f"Dataset: {root}")
    print(f"Total labels: {len(label_files)} (Non-empty: {len(non_empty)})")

    count = 0
    targets = non_empty if args.all else non_empty[:args.num]
    
    for lbl_file in targets:
        img_file = img_dir / (lbl_file.stem + ".jpg")
        if not img_file.exists():
            # Try png/jpeg
            for ext in [".png", ".jpeg", ".webp"]:
                cand = img_dir / (lbl_file.stem + ext)
                if cand.exists():
                    img_file = cand
                    break
        if not img_file.exists():
            continue

        img = load_image(img_file)
        if img is None:
            continue

        annotated_img, n_boxes = draw_yolo_labels(img, lbl_file)
        out_path = out_dir / f"view_{count+1:03d}_{lbl_file.stem[:30]}.jpg"
        save_image(out_path, annotated_img)
        count += 1
        print(f"[{count}/{len(targets)}] Saved {out_path.name} ({n_boxes} boxes)")

    print(f"\nDone! Saved {count} visualized images in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
