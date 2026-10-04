# GreenSM Data Centric Challenge - Group 23
**VinUni AI In Action (AI20k) · Day 15 Capstone Hackathon**

Mã số đội / Seed huấn luyện: **23** (hoặc seed được cấp bởi BTC)  
Đối tượng nhận diện: Duy nhất lớp `0 = GreenSM` (Taxi điện GSM trên đường phố Việt Nam).

---

## Cấu trúc thư mục

```
Group_23/
├── train_and_export.ipynb          # Notebook chính thức huấn luyện YOLOv8n và xuất ONNX
├── GreenSM Data Centric Challenge.pdf # Đề bài & thể lệ cuộc thi
├── bao_cao_slide_va_chien_luoc.md  # Kế hoạch tác chiến 4 vòng & kịch bản 6 Slide thuyết trình
└── tools/
    ├── extract_frames.py           # Cắt video thành ảnh thô tự động theo chu kỳ giây
    ├── assisted_labeler.py         # Gán nhãn nháp (YOLO COCO + bộ lọc màu Cyan/Teal)
    ├── blur_license_plates.py      # Che/làm mờ biển số xe tự động (bảo vệ quyền riêng tư)
    └── dataset_qa.py               # Kiểm tra định dạng nhãn, tọa độ, và đóng gói du_lieu.zip
```

---

## Hướng dẫn sử dụng bộ công cụ

### 1. Cắt video thành ảnh thô
```bash
python tools/extract_frames.py -i raw_videos -o raw_images -t 1.0
```

### 2. Che/làm mờ biển số xe (Anonymization)
Hỗ trợ 3 phương pháp làm mờ: `blur` (Gaussian Blur), `pixelate` (Mosaic), `black` (khung đen).
```bash
python tools/blur_license_plates.py -i raw_images -o anonymized_images -m blur
```

### 3. Gán nhãn bán tự động (Assisted Labeling)
Tự phát hiện vị trí ô tô và lọc màu đặc trưng GreenSM để tạo trước các file nhãn `.txt`:
```bash
python tools/assisted_labeler.py -i anonymized_images -l raw_labels
```
*Lưu ý: Sau khi chạy, kiểm tra và chỉnh khít lại viền hộp trên Roboflow / CVAT.*

### 4. Kiểm tra chất lượng và đóng gói dữ liệu
```bash
python tools/dataset_qa.py -d dataset -o du_lieu.zip
```

### 5. Huấn luyện trên Google Colab
Mở `train_and_export.ipynb` trên Colab T4 GPU, tải lên `du_lieu.zip` và chạy theo thứ tự để nhận `submission.zip`.
