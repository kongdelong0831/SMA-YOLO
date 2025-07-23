# SMA-YOLO
## Introduction
An improved YOLOv8-based model for stent strut detection in IVOCT images, optimized for multi-frame spatio-temporal fusion and small object detection accuracy.

## Key Features
- **Improvements**: Multi-frame input fusion, Backbone/Neck/Head module optimizations
- **Dependencies**: Python 3.8+, PyTorch 2.0+, ultralytics
- **Usage**:
  1. Configure dataset paths in `data.yaml`
  2. Train: `yolo task=detect mode=train model=weights/yolov8n.pt data=data.yaml`
  3. Infer: `yolo task=detect mode=predict model=weights/best.pt source=test_images/`
