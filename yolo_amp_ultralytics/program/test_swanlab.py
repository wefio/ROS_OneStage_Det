from ultralytics import YOLO
from swanlab.integration.ultralytics import add_swanlab_callback
from email_notifier import initialize_swanlab  # 确保导入正确的函数名

if __name__ == "__main__":
    try:
        # 初始化 SwanLab 并集成邮件通知（指定 project 和 experiment_name）
        initialize_swanlab(
            project_name="ultralytics",
            experiment_name=""
        )

        # 创建 YOLO 模型
        model = YOLO('models/yolo11n.pt', task='detect')
        model('output2.mp4',save=True)
        model.info()

        # 添加 SwanLab 回调
        add_swanlab_callback(model)

        # 开始训练
        '''model.train(
            data=r"ultralytics-8.3.55/ultralytics/cfg/datasets/coco.yaml",  # 确保这个文件存在
            epochs=3,
            imgsz=320,
        )'''


    except Exception as e:  # 捕获更广泛的异常类型
        print(f"训练异常: {str(e)}")