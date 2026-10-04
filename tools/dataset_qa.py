import os
import zipfile
from pathlib import Path
import argparse
import shutil

def validate_and_package(dataset_dir, output_zip="du_lieu.zip"):
    dataset_dir = Path(dataset_dir)
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    # Check structure: either flat (images/ and labels/) or already split (train/val)
    images = sorted(p for p in dataset_dir.rglob("*") if p.suffix.lower() in img_exts and "__MACOSX" not in p.parts)
    if not images:
        print(f"Error: No images found in {dataset_dir}")
        return False
        
    print(f"Found {len(images)} images in {dataset_dir}. Checking labels...")
    
    issues = []
    box_count = 0
    negative_images = 0
    small_boxes = 0
    oversized_boxes = 0
    
    for img in images:
        # Locate label
        label_file = None
        parts = list(img.parts)
        if "images" in parts:
            i = len(parts) - 1 - parts[::-1].index("images")
            parts[i] = "labels"
            cand = Path(*parts).with_suffix(".txt")
            if cand.is_file():
                label_file = cand
        if label_file is None:
            cand = img.with_suffix(".txt")
            if cand.is_file():
                label_file = cand
                
        if label_file is None:
            # Treated as negative sample
            negative_images += 1
            continue
            
        lines = [line.strip() for line in label_file.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        if not lines:
            negative_images += 1
            continue
            
        for line_no, line in enumerate(lines, 1):
            cols = line.split()
            if len(cols) != 5:
                issues.append(f"{label_file.name}:{line_no} Invalid column count ({len(cols)} instead of 5)")
                continue
            try:
                cls_id = int(cols[0])
                cx, cy, w, h = [float(x) for x in cols[1:]]
            except ValueError:
                issues.append(f"{label_file.name}:{line_no} Non-numeric values found")
                continue
                
            if cls_id != 0:
                issues.append(f"{label_file.name}:{line_no} Invalid class {cls_id}! Only class 0 (GreenSM) allowed.")
            if not all(0.0 <= v <= 1.0 for v in [cx, cy, w, h]):
                issues.append(f"{label_file.name}:{line_no} Coordinates out of [0, 1] range: {cols}")
            if w <= 0 or h <= 0:
                issues.append(f"{label_file.name}:{line_no} Degenerate box dimensions w={w}, h={h}")
                
            box_area = w * h
            if box_area < 0.0005:
                small_boxes += 1
            if box_area > 0.90:
                oversized_boxes += 1
                
            box_count += 1
            
    print("\n--- DATASET QA REPORT ---")
    print(f"Total Images: {len(images)}")
    print(f"Total GreenSM Bounding Boxes: {box_count}")
    print(f"Negative Images (no GreenSM): {negative_images} ({negative_images/len(images)*100:.1f}%)")
    print(f"Very Small Boxes (<0.05% image area): {small_boxes}")
    print(f"Very Large Boxes (>90% image area): {oversized_boxes}")
    
    if negative_images / len(images) < 0.05:
        print("\n[WARNING] Negative images are < 5%. It is recommended to include 10-15% negative images (other cars, street backgrounds) to prevent false positives.")
    elif negative_images / len(images) > 0.35:
        print("\n[NOTE] Negative images > 35%. Check that images with GreenSM were not accidentally left unlabeled.")

    if issues:
        print(f"\n[FAILED] Found {len(issues)} critical format errors:")
        for issue in issues[:15]:
            print(f"  - {issue}")
        if len(issues) > 15:
            print(f"  ... and {len(issues) - 15} more issues.")
        return False
        
    print("\n[PASSED] All labels conform to YOLO detection format (class 0, coords in [0, 1]).")
    
    # Packaging
    out_zip = Path(output_zip)
    print(f"\nPackaging dataset into {out_zip}...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in dataset_dir.rglob("*"):
            if f.is_file() and not f.name.startswith(".") and "__MACOSX" not in f.parts and f != out_zip:
                rel = f.relative_to(dataset_dir)
                zf.write(f, rel)
                
    zip_size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Successfully created {out_zip} ({zip_size_mb:.2f} MB). Ready to upload to Google Drive or Colab!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, default="dataset", help="Dataset root directory")
    parser.add_argument("--out", "-o", type=str, default="du_lieu.zip", help="Output zip filename")
    args = parser.parse_args()
    
    validate_and_package(args.dir, args.out)
