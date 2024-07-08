from ultralytics import YOLO

# Load a model
model = YOLO("./runs/segment/train/weights/best.pt")

# Validation metrics 
metrics = model.val(data="./dataset_segm/data.yaml", save_json=True, split="val", iou=0.7, plots=True, conf=0.7)
metrics.confusion_matrix.iou_threshold = 0.7
print(metrics.confusion_matrix.matrix)

print(metrics.box.maps)