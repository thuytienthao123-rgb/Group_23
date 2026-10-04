"""
Prepare dataset for quick editing in AnyLabeling / X-AnyLabeling.
- Converts YOLO txt annotations into native LabelMe/AnyLabeling JSON files.
- Places .jpg, .json, and classes.txt together in subfolders so AnyLabeling loads boxes instantly.
- Generates member packages for parallel review (Hiếu, Linh, Chứ, Kiên).
"""

import sys
import io
import json
import zipfile
from pathlib import Path
import cv2
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

def get_image_size(img_path: Path):
    try:
        buf = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            return img.shape[0], img.shape[1] # h, w
    except Exception:
        pass
    return 1080, 1920

def yolo_to_anylabeling_json(txt_content: str, img_name: str, h: int, w: int, class_name: str = "GreenSM"):
    shapes = []
    lines = [l.strip() for l in txt_content.strip().splitlines() if l.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, bw, bh = [float(x) for x in parts[1:]]
        except ValueError:
            continue
            
        x1 = round(max(0.0, (cx - bw / 2.0) * w), 2)
        y1 = round(max(0.0, (cy - bh / 2.0) * h), 2)
        x2 = round(min(float(w), (cx + bw / 2.0) * w), 2)
        y2 = round(min(float(h), (cy + bh / 2.0) * h), 2)

        shapes.append({
            "label": class_name,
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {}
        })

    return {
        "version": "0.4.15",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }

def main():
    root_dir = Path(__file__).resolve().parent.parent
    zip_label_cand = [
        root_dir / "ảnh đã pre-label-20261004T062307Z-1-001.zip",
        root_dir / "anh da pre-label-20261004T062307Z-1-001.zip"
    ]
    zip_path = None
    for z in zip_label_cand:
        if z.exists():
            zip_path = z
            break
    if not zip_path:
        for z in root_dir.glob("*.zip"):
            if "pre-label" in z.name.lower():
                zip_path = z
                break
                
    if not zip_path:
        print("Error: Could not find pre-label zip file.")
        return

    img_base = root_dir / "raw_new_images" / "Ảnh đã che"
    if not img_base.exists():
        img_base = root_dir / "raw_new_images"
    
    out_dir = root_dir / "anylabeling_workspace"
    out_dir.mkdir(exist_ok=True)

    print(f"Reading pre-labels from: {zip_path.name}")
    print(f"Reading images from: {img_base}")
    print(f"Output workspace: {out_dir}")

    # Read all txt files from zip into memory: dict key: (subfolder, stem) -> content
    labels_map = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".txt"):
                parts = name.replace("\\", "/").split("/")
                # usually ảnh đã pre-label/<folder>/<frame>.txt
                if len(parts) >= 3:
                    folder = parts[-2]
                    stem = Path(parts[-1]).stem
                    content = zf.read(name).decode("utf-8", errors="replace")
                    labels_map[(folder, stem)] = content

    print(f"Loaded {len(labels_map)} label files from zip.")

    # Convert each image and label into AnyLabeling JSON in anylabeling_workspace/<subfolder>/
    total_converted = 0
    subdirs = sorted([d.name for d in img_base.glob("*") if d.is_dir()])
    
    for sub in subdirs:
        sub_img_dir = img_base / sub
        sub_out_dir = out_dir / sub
        sub_out_dir.mkdir(exist_ok=True, parents=True)
        
        # Write classes.txt for AnyLabeling
        (sub_out_dir / "classes.txt").write_text("GreenSM\n", encoding="utf-8")

        for img_p in sorted(sub_img_dir.glob("*.jpg")):
            stem = img_p.stem
            txt_content = labels_map.get((sub, stem), "")
            
            # Read image size
            h, w = get_image_size(img_p)
            
            # Create JSON
            json_obj = yolo_to_anylabeling_json(txt_content, img_p.name, h, w, "GreenSM")
            
            # Copy or link image
            target_img = sub_out_dir / img_p.name
            if not target_img.exists():
                target_img.write_bytes(img_p.read_bytes())
                
            # Write JSON
            target_json = sub_out_dir / f"{stem}.json"
            target_json.write_text(json.dumps(json_obj, indent=2, ensure_ascii=False), encoding="utf-8")
            
            # Also keep .txt for reference
            target_txt = sub_out_dir / f"{stem}.txt"
            target_txt.write_text(txt_content, encoding="utf-8")
            
            total_converted += 1

    print(f"Successfully generated {total_converted} AnyLabeling JSON files in: {out_dir}")

    # Create team packages
    team_assignments = {
        "member_Hieu_Video0_1": ["Video_0", "Video_1"],
        "member_Linh_Video2_3": ["Video_2", "Video_3"],
        "member_Chu_Video4_5_occluded": ["Video_4", "Video_5", "vid_occluded"],
        "member_Kien_xanhsm_di_re_thang_van": ["vid_xanhsm_di_re", "vid_xanhsm_di_thang", "vid_xe_van"]
    }

    team_pkg_dir = root_dir / "team_label_packages"
    team_pkg_dir.mkdir(exist_ok=True)
    
    for member_name, assigned_folders in team_assignments.items():
        zip_pkg = team_pkg_dir / f"{member_name}.zip"
        with zipfile.ZipFile(zip_pkg, "w", zipfile.ZIP_DEFLATED) as zout:
            for fold in assigned_folders:
                fold_p = out_dir / fold
                if fold_p.exists():
                    for f in fold_p.glob("*"):
                        zout.write(f, arcname=f"{fold}/{f.name}")
        mb = zip_pkg.stat().st_size / (1024 * 1024)
        print(f"Created package for {member_name}: {zip_pkg.name} ({mb:.1f} MB)")

if __name__ == "__main__":
    main()
