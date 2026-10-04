import os
import shutil
from pathlib import Path

def combine_images(images_dir, output_dir):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    # Output directory for images - we can place it next to obj_train_data 
    # so they can be merged easily, or just a separate folder
    # YOLO typically looks for images in the same folder as labels or an 'images' folder.
    # The train.txt generated earlier references "data/obj_train_data/..."
    # So we should put the images right inside Combined_Labels_Final/obj_train_data!
    
    # Wait, the prompt says "đổi tên toàn bộ các ảnh gốc".
    # Let's put them in a dedicated folder or directly inside Combined_Labels_Final/obj_train_data.
    # Putting them in Combined_Labels_Final/obj_train_data is best because train.txt expects them there.
    
    output_obj_dir = output_dir / "obj_train_data"
    output_obj_dir.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    
    # Mapping for any zip name discrepancies (e.g. user named the zip video4.zip instead of Video_4...)
    folder_to_prefix = {
        "Video_4": "video4",
        "Video_3": "Video_3",
    }
    
    print(f"Scanning original images in {images_dir} ...")
    
    for sub_dir in images_dir.iterdir():
        if sub_dir.is_dir():
            prefix = folder_to_prefix.get(sub_dir.name, sub_dir.name)
            
            # Find all images in this folder
            for img_path in sub_dir.rglob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    new_name = f"{prefix}_{img_path.name}"
                    new_path = output_obj_dir / new_name
                    
                    if not new_path.exists():
                        shutil.copy2(img_path, new_path)
                        total_copied += 1
                        
    print(f"\nSuccessfully copied and renamed {total_copied} images to:")
    print(f" -> {output_obj_dir}")
    print("Now your images and .txt labels are perfectly mapped in the same folder!")

if __name__ == "__main__":
    images_in = "I:\\My Drive\\Mini Hackathon\\Ảnh đã che"
    dataset_out = "I:\\My Drive\\Mini Hackathon\\Combined_Labels_Final"
    combine_images(images_in, dataset_out)
