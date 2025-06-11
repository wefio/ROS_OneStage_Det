from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")
data_path = r"datasets/coco8/coco8.yaml"

if __name__ == "__main__":
    # Validate the model
    metrics = model.val(data=data_path, imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")
    print(metrics.box.map)  # map50-95