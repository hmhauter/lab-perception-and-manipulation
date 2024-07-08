from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')

folder_path = ''

# Train the model
results = model.train(data='./dataset_segm/data.yaml', epochs=100, imgsz=1024, device=0, patience=10, seed=0, plots=True, val=True)

