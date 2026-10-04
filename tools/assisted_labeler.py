import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO

def is_greensm_color(crop_bgr, min_cyan_ratio=0.08):
    """
    Check if the cropped vehicle contains characteristic GreenSM cyan/teal paint.
    GreenSM color in HSV:
    Hue roughly 80 to 105 (OpenCV Hue scale is 0-179, so 80-105 is Cyan/Aquamarine/Teal)
    Saturation: >= 50
    Value: >= 50
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False
        
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    
    # Mask for GreenSM cyan/teal
    lower_teal = np.array([75, 45, 45])
    upper_teal = np.array([105, 255, 255])
    
    mask = cv2.inRange(hsv, lower_teal, upper_teal)
    cyan_pixel_count = np.count_nonzero(mask)
    total_pixels = crop_bgr.shape[0] * crop_bgr.shape[1]
    
    ratio = cyan_pixel_count / max(1, total_pixels)
    return ratio >= min_cyan_ratio, ratio

def auto_detect_candidates(images_dir, output_labels_dir, conf_thresh=0.25, filter_color=True):
    images_dir = Path(images_dir)
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    model = YOLO("yolov8n.pt")  # COCO pretrained
    vehicle_classes = [2, 7] # 2: car, 7: truck (COCO ids)
    
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in img_exts)
    
    print(f"Running candidate detection on {len(images)} images...")
    
    greensm_box_count = 0
    negative_image_count = 0
    
    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        
        results = model.predict(img_bgr, conf=conf_thresh, classes=vehicle_classes, verbose=False)[0]
        
        lines = []
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(w, xyxy[2]), min(h, xyxy[3])
            
            crop = img_bgr[y1:y2, x1:x2]
            
            is_gsm = True
            if filter_color:
                is_gsm, ratio = is_greensm_color(crop)
                
            if is_gsm:
                # Convert xyxy to YOLO format: 0 cx cy nw nh (normalized)
                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                nw = (x2 - x1) / w
                nh = (y2 - y1) / h
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                greensm_box_count += 1
                
        label_file = output_labels_dir / f"{img_path.stem}.txt"
        label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        
        if not lines:
            negative_image_count += 1
            
    print(f"\nDone auto-detection!")
    print(f"Total images processed: {len(images)}")
    print(f"GreenSM candidate boxes proposed: {greensm_box_count}")
    print(f"Negative samples (0 boxes, empty .txt): {negative_image_count}")
    print(f"\nIMPORTANT: Human review is required per competition rules! Open in CVAT, Labelme, or Roboflow to verify.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", "-i", type=str, default="raw_images", help="Folder with raw images")
    parser.add_argument("--labels", "-l", type=str, default="raw_labels", help="Folder to save generated labels")
    parser.add_argument("--conf", "-c", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-color-filter", action="store_true", help="Propose all cars (useful if you want to select manually)")
    args = parser.parse_args()
    
    auto_detect_candidates(args.images, args.labels, conf_thresh=args.conf, filter_color=not args.no_color_filter)
