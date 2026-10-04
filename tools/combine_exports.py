import zipfile
from pathlib import Path
import argparse
import os

def combine_cvat_exports(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create the output directory
    output_obj_dir = output_dir / "obj_train_data"
    output_obj_dir.mkdir(parents=True, exist_ok=True)
    
    zip_files = list(input_dir.glob("*.zip"))
    
    if not zip_files:
        print(f"No .zip files found in {input_dir}")
        return
        
    print(f"Found {len(zip_files)} zip files.")
    
    total_labels = 0
    train_lines = []
    
    for zip_path in zip_files:
        # Generate a clean prefix from the zip filename 
        # (e.g. "vid_troi_mua_1_export.zip" -> "vid_troi_mua_1")
        prefix = zip_path.stem.replace("_export", "").replace("_annotations", "")
        print(f"Processing {zip_path.name} (prefix: {prefix}_) ...")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for file_info in zf.infolist():
                # We only want the .txt label files in obj_train_data
                if file_info.filename.endswith(".txt") and not file_info.filename.endswith("train.txt"):
                    # Original filename like "frame_0001.txt"
                    orig_name = Path(file_info.filename).name
                    
                    # New filename like "vid_troi_mua_1_frame_0001.txt"
                    new_name = f"{prefix}_{orig_name}"
                    new_path = output_obj_dir / new_name
                    
                    # Read content and write to new file
                    content = zf.read(file_info.filename)
                    new_path.write_bytes(content)
                    
                    # Add to train_lines for the combined train.txt
                    img_name = new_name.replace('.txt', '.jpg')
                    train_lines.append(f"data/obj_train_data/{img_name}")
                    total_labels += 1
                    
    # Write the combined obj.names, obj.data, and train.txt
    (output_dir / "obj.names").write_text("greensm\n")
    (output_dir / "obj.data").write_text("classes = 1\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n")
    (output_dir / "train.txt").write_text("\n".join(train_lines) + "\n")
    
    # Check for RAR files
    rar_files = list(input_dir.glob("*.rar"))
    if rar_files:
        print("\nWARNING: Found .rar file(s) which cannot be processed automatically by this script:")
        for r in rar_files:
            print(f" - {r.name}")
        print("Please extract them manually and re-zip them as .zip if you want to include them.")
    
    print(f"\nDone! Combined {total_labels} label files into: {output_dir}")
    print(f"NOTE: Remember to also rename your original images with the same prefix to match!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Folder containing the CVAT export .zip files")
    parser.add_argument("--output", "-o", required=True, help="Output folder for the combined dataset")
    args = parser.parse_args()
    
    combine_cvat_exports(args.input, args.output)
