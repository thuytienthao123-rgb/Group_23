import cv2
import numpy as np
from pathlib import Path
import argparse
import shutil
from ultralytics import YOLO

def apply_anonymization(image, x1, y1, x2, y2, method="blur", factor=0.08):
    """
    Apply blur, pixelate, or black rectangle to anonymize the region [x1:x2, y1:y2].
    """
    h, w = image.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    
    if x2 <= x1 or y2 <= y1:
        return image
        
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return image
        
    rw, rh = x2 - x1, y2 - y1
    
    if method == "blur":
        # Gaussian blur with large kernel
        kw = max(3, int(rw / 2) | 1)
        kh = max(3, int(rh / 2) | 1)
        blurred = cv2.GaussianBlur(roi, (kw, kh), sigmaX=30)
        image[y1:y2, x1:x2] = blurred
        
    elif method == "pixelate":
        # Mosaic effect
        small_w = max(1, int(rw * factor))
        small_h = max(1, int(rh * factor))
        temp = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(temp, (rw, rh), interpolation=cv2.INTER_NEAREST)
        image[y1:y2, x1:x2] = pixelated
        
    elif method == "black":
        # Solid black box
        image[y1:y2, x1:x2] = 0
        
    return image

def find_plate_candidates_in_crop(crop_bgr):
    """
    Find Vietnamese license plate candidates inside a car bumper region.
    Taxis in Vietnam (including GreenSM) have YELLOW plates, some white plates.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return []
        
    ch, cw = crop_bgr.shape[:2]
    if ch < 10 or cw < 20:
        return []
        
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # Yellow plate mask (HSV: Hue 15-38, Sat 70-255, Val 80-255)
    lower_yellow = np.array([12, 60, 70])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # White plate / high contrast mask
    _, white_thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Sobel edge density to find license plate characters
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.uint8(np.absolute(sobelx))
    _, edge_thresh = cv2.threshold(sobelx, 60, 255, cv2.THRESH_BINARY)
    
    combined_mask = cv2.bitwise_or(yellow_mask, cv2.bitwise_and(white_thresh, edge_thresh))
    
    # Morphological close to bridge text characters into single plate block
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    crop_area = cw * ch
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(1, h)
        area = w * h
        
        # Plate dimensions check: aspect ratio between 1.5 and 5.5
        # Area should be reasonable portion of bumper (between 0.8% and 25%)
        if 1.4 <= aspect <= 5.8 and (0.008 * crop_area <= area <= 0.28 * crop_area):
            boxes.append((x, y, w, h))
            
    return boxes

def process_image(img_path, output_path, model=None, method="blur", fallback_bumper=True):
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
        
    ih, iw = img.shape[:2]
    plates_blurred = 0
    
    if model is None:
        model = YOLO("yolov8n.pt")
        
    # Detect vehicles: 2: car, 3: motorcycle, 5: bus, 7: truck
    results = model.predict(img, conf=0.25, classes=[2, 3, 5, 7], verbose=False)[0]
    
    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cx1, cy1, cx2, cy2 = max(0, xyxy[0]), max(0, xyxy[1]), min(iw, xyxy[2]), min(ih, xyxy[3])
        cw = cx2 - cx1
        ch = cy2 - cy1
        
        if cw < 30 or ch < 30:
            continue
            
        # Target bumper ROI: bottom 45% of car, middle 80% width
        by1 = cy1 + int(ch * 0.55)
        by2 = cy2
        bx1 = cx1 + int(cw * 0.10)
        bx2 = cx2 - int(cw * 0.10)
        
        bumper_crop = img[by1:by2, bx1:bx2]
        plate_candidates = find_plate_candidates_in_crop(bumper_crop)
        
        if plate_candidates:
            for px, py, pw, ph in plate_candidates:
                # Add slight margin around plate
                mx = int(pw * 0.15)
                my = int(ph * 0.20)
                gx1 = bx1 + px - mx
                gy1 = by1 + py - my
                gx2 = bx1 + px + pw + mx
                gy2 = by1 + py + ph + my
                apply_anonymization(img, gx1, gy1, gx2, gy2, method=method)
                plates_blurred += 1
        elif fallback_bumper:
            # Fallback: blur estimated license plate zone in center bottom of bumper
            plate_w = int(cw * 0.28)
            plate_h = int(ch * 0.12)
            plate_cx = cx1 + int(cw * 0.5)
            plate_cy = cy1 + int(ch * 0.82)
            
            gx1 = plate_cx - int(plate_w / 2)
            gy1 = plate_cy - int(plate_h / 2)
            gx2 = plate_cx + int(plate_w / 2)
            gy2 = plate_cy + int(plate_h / 2)
            apply_anonymization(img, gx1, gy1, gx2, gy2, method=method)
            plates_blurred += 1
            
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return plates_blurred

def blur_directory(input_dir, output_dir, method="blur", copy_labels=True):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in img_exts and "__MACOSX" not in p.parts)
    
    if not images:
        print(f"No images found in {input_dir}")
        return
        
    print(f"Loading YOLOv8 detector for vehicle license plate detection...")
    model = YOLO("yolov8n.pt")
    
    print(f"Processing {len(images)} images with method='{method}'...")
    total_blurred = 0
    for idx, img_p in enumerate(images, 1):
        rel = img_p.relative_to(input_dir)
        out_p = output_dir / rel
        blurred = process_image(img_p, out_p, model=model, method=method)
        total_blurred += blurred
        
        # If labels exist next to image or in corresponding labels/ directory, copy them
        if copy_labels:
            label_p = img_p.with_suffix(".txt")
            if label_p.is_file():
                out_lbl = out_p.with_suffix(".txt")
                out_lbl.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(label_p, out_lbl)
                
        if idx % 20 == 0 or idx == len(images):
            print(f"  [{idx}/{len(images)}] Processed {img_p.name} (Plates masked: {total_blurred})")
            
    print(f"\nDone! Anonymized {len(images)} images saved to {output_dir}")
    print(f"Total license plate regions blurred/masked: {total_blurred}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blur/anonymize vehicle license plates")
    parser.add_argument("--input", "-i", type=str, default="raw_images", help="Input image or directory")
    parser.add_argument("--output", "-o", type=str, default="anonymized_images", help="Output image or directory")
    parser.add_argument("--method", "-m", type=str, choices=["blur", "pixelate", "black"], default="blur",
                        help="Anonymization method: blur (Gaussian), pixelate (Mosaic), black (solid box)")
    parser.add_argument("--no-labels", action="store_true", help="Do not copy .txt label files")
    args = parser.parse_args()
    
    in_path = Path(args.input)
    out_path = Path(args.output)
    
    if in_path.is_file():
        model = YOLO("yolov8n.pt")
        process_image(in_path, out_path, model=model, method=args.method)
        print(f"Saved anonymized image to {out_path}")
    elif in_path.is_dir():
        blur_directory(in_path, out_path, method=args.method, copy_labels=not args.no_labels)
    else:
        print(f"Error: {args.input} does not exist.")
