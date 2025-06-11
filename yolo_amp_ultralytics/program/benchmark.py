from ultralytics.utils.benchmarks import benchmark
from pathlib import Path

if __name__ == '__main__':
    benchmark(model=Path("yolo11n.pt"), data=r"datasets/coco8/coco8.yaml", imgsz=640, half=False, device="cuda:0")

# Benchmark specific export format
#benchmark(model="yolo11n.pt", data=r"datasets/coco8/coco8.yaml", imgsz=640, format="onnx")