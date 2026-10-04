import os
import cv2
import argparse
from pathlib import Path

def extract_frames(video_path, output_dir, interval_sec=1.0, quality=95):
    """
    Extract frames from video at fixed time intervals, avoiding redundant frames.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return 0
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_step = max(1, int(fps * interval_sec))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing {video_path.name}: {total_frames} frames, FPS: {fps:.1f}, taking 1 frame every {interval_sec}s...")
    
    saved_count = 0
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_step == 0:
            out_filename = f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
            out_path = output_dir / out_filename
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            saved_count += 1
            
        frame_idx += 1
        
    cap.release()
    print(f"Done {video_path.name}: Saved {saved_count} frames to {output_dir}")
    return saved_count

def process_all_videos(input_dir, output_dir, interval_sec=1.0):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    
    video_files = [p for p in input_dir.glob("*") if p.suffix.lower() in video_exts]
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
        
    total_saved = 0
    for v in video_files:
        total_saved += extract_frames(v, output_dir, interval_sec=interval_sec)
    print(f"\nAll done! Total extracted frames: {total_saved} in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos for GreenSM dataset")
    parser.add_argument("--input", "-i", type=str, default="raw_videos", help="Input video file or folder")
    parser.add_argument("--output", "-o", type=str, default="raw_images", help="Output directory for frames")
    parser.add_argument("--interval", "-t", type=float, default=1.0, help="Interval in seconds between frames (e.g. 1.0 or 0.5)")
    args = parser.parse_args()
    
    in_path = Path(args.input)
    if in_path.is_file():
        extract_frames(in_path, args.output, args.interval)
    elif in_path.is_dir():
        process_all_videos(in_path, args.output, args.interval)
    else:
        print(f"Path not found: {args.input}. Create folder 'raw_videos' and put your MP4/MOV videos there.")
