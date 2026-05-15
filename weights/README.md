# Model Weights

The trained YOLOv11 weights are **not included** in this repository due to file size.  
Download them from Google Drive and place them in the correct sub-folders as shown below.

---

## Download Links

| Stage | Model | Val mAP@50 | Size | Download |
|-------|-------|-----------|------|----------|
| Stage 1 | YOLOv11n | 0.9715 | 5.5 MB | [Download stage1_yolo11n/best.pt]https://drive.google.com/file/d/1A9NuYtFzQMiz4tWRFG4XYSoLID00xiQ7/view?usp=sharing|
| Stage 2 | YOLOv11s | 0.9563 | 19.2 MB | [Download stage2_yolo11s/best.pt]https://drive.google.com/file/d/1EFra7o1jtuAjZgrsiEnmwMRK7klnVTue/view?usp=sharing|
| Stage 3 | YOLOv11m | 0.9708 | 40.5 MB | [Download stage3_yolo11m/best.pt]https://drive.google.com/file/d/1Xds62KMTFq68m_q7hSHh0ElGyC8XBUFe/view?usp=sharing|

---

## Expected Folder Structure

After downloading, your `weights/` directory should look like this:
weights/
├── README.md
├── stage1_yolo11n/
│   └── weights/
│       └── best.pt
├── stage2_yolo11s/
│   └── weights/
│       └── best.pt
└── stage3_yolo11m/
└── weights/
└── best.pt
