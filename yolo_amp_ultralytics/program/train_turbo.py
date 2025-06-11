from ultralytics import YOLO
import torch

def train_yolo_voc():    
    # 硬件配置
    # 检查GPU是否可用，优先使用GPU进行训练
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')  # 输出当前使用的设备
    
    # 模型加载
    # 加载YOLOv8n预训练模型
    model = YOLO('yolov8n.pt')  
    
    # 训练参数配置
    train_args = {
        'data': 'datasets/VOC/VOC.yaml',  # 数据集配置文件路径
        'epochs': 30,                     # 训练总轮次
        'imgsz': 640,                     # 输入图像尺寸
        'batch': 96,                      # 批量大小（根据显存调整）
        'device': device,                 # 使用的设备
        'workers': 16,                    # 数据加载线程数
        'project': 'runs/train',          # 训练结果保存目录
        'name': 'yolov8n_voc_fast',       # 实验名称
        'save': True,                     # 是否保存训练结果
        'exist_ok': True,                 # 允许覆盖已有实验
        'pretrained': True,               # 使用预训练权重
        'optimizer': 'auto',              # 自动选择优化器
        'lr0': 0.01,                      # 初始学习率
        'cos_lr': True,                   # 使用余弦学习率调度
        'patience': 15,                   # 早停耐心值
        'amp': True,                      # 启用混合精度训练
        'plots': False,                   # 不保存训练曲线图（节省空间）
        'val': True,                      # 启用验证
        'cache': False,                   # 禁用数据缓存（避免内存不足）
        'close_mosaic': 5                 # 最后5个epoch关闭马赛克增强
    }
    
    # --- 开始训练 ---
    print("Starting training...")
    results = model.train(**train_args)  # 解参传递训练配置
    
    # --- 模型验证 ---
    print("\nValidating model...")
    val_results = model.val()  # 在验证集上评估模型
    
    # --- 输出验证结果 ---
    print('\nValidation results:')
    print(f"mAP50-95: {val_results.box.map:.4f}")  # COCO标准mAP
    print(f"mAP50: {val_results.box.map50:.4f}")   # VOC标准mAP
    
    # --- 模型导出 ---
    print("\nExporting model to ONNX...")
    # 导出为ONNX格式，进行模型简化，指定opset版本12
    model.export(format='onnx', simplify=True, opset=12)
    
    return model  # 返回训练好的模型对象

if __name__ == '__main__':
    # 主程序入口
    trained_model = train_yolo_voc()