from ultralytics import YOLO
import os

model_path = "yolov8n.pt"
data_path = r"datasets/coco8/coco8.yaml"

def get_inference_speed(results):
    """提取并计算速度指标"""
    speed = results.speed
    return {
        "inference_time_ms": speed.get("inference", 0),
        "fps": 1000 / (
            speed.get("preprocess", 0) +
            speed.get("inference", 0) +
            speed.get("postprocess", 0)
        ) if speed else 0
    }

if __name__ == "__main__":
    # 强化路径验证
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集路径 {data_path} 不存在")

    try:
        # 初始化模型
        model = YOLO(model_path)
        print("✅ 模型加载成功")

        # 执行验证
        results = model.val(
            data=data_path,
            imgsz=640,
            batch=64,
            device="cuda",
            save=True,  # 关闭结果保存
            verbose=True  # 关闭详细输出
        )

        # 获取速度指标
        speed_metrics = get_inference_speed(results)
        
        # 格式化输出
        print("\n=== 验证结果 ===")
        print(f"推理时间: {speed_metrics['inference_time_ms']:.1f} ms/im")
        print(f"FPS: {speed_metrics['fps']:.1f}")

    except Exception as e:
        print(f"❌ 验证失败: {str(e)}")
        import traceback
        traceback.print_exc()