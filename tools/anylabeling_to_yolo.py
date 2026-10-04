"""
Convert edited AnyLabeling JSON files back to YOLO format and package for Colab training.
"""

import sys
import io
import json
import zipfile
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

def convert_json_to_yolo(json_path: Path, class_mapping=None):
    if class_mapping is None:
        class_mapping = {"greensm": 0, "green_sm": 0, "taxi": 0, "car": 0, "0": 0}
        
    try:
        data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return []

    w = float(data.get("imageWidth", 1920))
    h = float(data.get("imageHeight", 1080))
    if w <= 0 or h <= 0:
        w, h = 1920.0, 1080.0

    yolo_lines = []
    shapes = data.get("shapes", [])
        label = str(shape.get("label", "GreenSM")).strip().lower()
        # Drop person / pedestrian labels completely
        if label in ["person", "people", "pedestrian", "nguoi", "người", "human"]:
            continue
        if label not in class_mapping:
            continue
        cls_id = class_mapping[label]

        if len(points) < 2:
            continue
            
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)
        
        # Clamp to image boundaries
        x1 = max(0.0, min(w, x1))
        x2 = max(0.0, min(w, x2))
        y1 = max(0.0, min(h, y1))
        y2 = max(0.0, min(h, y2))
        
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 1.0 or bh <= 1.0:
            continue
            
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        norm_w = bw / w
        norm_h = bh / h
        
        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")

    return yolo_lines

def main():
    root_dir = Path(__file__).resolve().parent.parent
    workspace_dir = root_dir / "anylabeling_workspace"
    out_dir = root_dir / "dataset_final"
    out_zip = root_dir / "du_lieu_final.zip"

    out_img_dir = out_dir / "images"
    out_lbl_dir = out_dir / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning edited JSON annotations from: {workspace_dir}")

    json_files = list(workspace_dir.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found in {workspace_dir}. Check path.")
        return

    print(f"Found {len(json_files)} JSON files. Converting to YOLO format...")

    total_boxes = 0
    non_empty_count = 0

    for jf in json_files:
        prefix = jf.parent.name
        unified_stem = f"{prefix}_{jf.stem}"
        
        # Corresponding image
        img_cand = jf.with_suffix(".jpg")
        if not img_cand.exists():
            img_cand = jf.with_suffix(".png")
            
        if img_cand.exists():
            target_img = out_img_dir / f"{unified_stem}{img_cand.suffix}"
            if not target_img.exists():
                target_img.write_bytes(img_cand.read_bytes())
                
        # Convert label
        lines = convert_json_to_yolo(jf)
        target_lbl = out_lbl_dir / f"{unified_stem}.txt"
        target_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        
        if lines:
            non_empty_count += 1
            total_boxes += len(lines)

    print("\n=== KẾT QUẢ CHUYỂN ĐỔI ===")
    print(f"Tổng số ảnh: {len(json_files)}")
    print(f"Ảnh có xe GreenSM: {non_empty_count}")
    print(f"Ảnh Negative (không có xe): {len(json_files) - non_empty_count}")
    print(f"Tổng số hộp nhãn: {total_boxes}")

    print(f"\nĐang nén vào {out_zip.name}...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in sorted(out_img_dir.glob("*")):
            zf.write(img, arcname=f"images/{img.name}")
        for lbl in sorted(out_lbl_dir.glob("*.txt")):
            zf.write(lbl, arcname=f"labels/{lbl.name}")

    mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Xong! File sẵn sàng nộp bài / train Colab: {out_zip.name} ({mb:.1f} MB)")

if __name__ == "__main__":
    main()
