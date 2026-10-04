import zipfile
from pathlib import Path
import argparse
import os

def export_cvat_yolo_single(labels_dir, zip_path, class_name="greensm"):
    labels_dir = Path(labels_dir)
    zip_path = Path(zip_path)
    
    txt_files = list(labels_dir.rglob("*.txt"))
    if not txt_files:
        print(f"  -> Skipping {labels_dir.name}: No .txt files found.")
        return False
        
    print(f"  -> {labels_dir.name}: Found {len(txt_files)} labels. Creating {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("obj.names", f"{class_name}\n")
        obj_data = f"classes = 1\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n"
        zf.writestr("obj.data", obj_data)
        
        train_lines = []
        for txt_file in txt_files:
            rel_path = txt_file.relative_to(labels_dir)
            arc_name = f"obj_train_data/{rel_path}"
            zf.write(txt_file, arcname=arc_name)
            
            img_rel_path = rel_path.with_suffix('.jpg')
            train_lines.append(f"data/obj_train_data/{img_rel_path}")
            
        zf.writestr("train.txt", "\n".join(train_lines) + "\n")
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", "-l", required=True, help="Folder containing the subfolders with pre-labeled .txt files")
    parser.add_argument("--output-dir", "-o", required=True, help="Directory to save the output .zip files")
    args = parser.parse_args()
    
    labels_base = Path(args.labels)
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning {labels_base} for subdirectories...")
    for sub_dir in labels_base.iterdir():
        if sub_dir.is_dir():
            zip_name = f"{sub_dir.name}_annotations.zip"
            zip_path = output_base / zip_name
            export_cvat_yolo_single(sub_dir, zip_path)
            
    print("Done generating all zip files!")
