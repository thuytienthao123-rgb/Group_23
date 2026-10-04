$folders = @("vid_troi_mua_1", "vid_troi_mua_2")
foreach ($f in $folders) {
    Write-Host "============================="
    Write-Host "Processing folder: $f"
    Write-Host "============================="
    
    Write-Host "[1/3] Blurring license plates..."
    python tools\blur_license_plates.py --input "I:\My Drive\Mini Hackathon\video đã chia frame\$f" --output "I:\My Drive\Mini Hackathon\Ảnh đã che\$f"
    
    Write-Host "[2/3] Generating pre-labels..."
    python tools\assisted_labeler.py --images "I:\My Drive\Mini Hackathon\Ảnh đã che\$f" --labels "I:\My Drive\Mini Hackathon\ảnh đã pre-label\$f"
}

Write-Host "============================="
Write-Host "[3/3] Generating CVAT zip files..."
python tools\export_cvat_multiple.py -l "I:\My Drive\Mini Hackathon\ảnh đã pre-label" -o "I:\My Drive\Mini Hackathon\CVAT_Zips"
Write-Host "Done!"
