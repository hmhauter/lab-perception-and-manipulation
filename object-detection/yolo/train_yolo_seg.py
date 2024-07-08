from ultralytics import YOLO

# Script to train the YOLO model based on a YOLOv8 nano segmentation model using the YOLO API

# Load a model
model = YOLO('yolov8n-seg.pt')

folder_path = ''

# Train the model
results = model.train(data='object-detection/dataset-segm/data.yaml', epochs=100, imgsz=1024, device=0, patience=10, seed=0, plots=True, val=True)

