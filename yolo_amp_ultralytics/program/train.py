from ultralytics import YOLO
import torch

def train_yolo_voc():
    # 检查GPU可用性
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # 加载模型 (YOLOv8n是轻量级版本，适合6GB显存)
    model = YOLO('yolov8n.pt')  # 会自动下载预训练权重
    
    # 训练配置 (更新为兼容v8.3.0的参数)
    train_args = {
        'data': 'datasets/VOC/VOC.yaml',  # 确保yaml文件路径正确
        'epochs': 50,                     # VOC数据集较小，建议更多epochs
        'imgsz': 640,
        'batch': 32 if device == '0' else 4,  # 根据显存调整
        'device': device,
        'workers': 8,                     # 根据CPU核心数调整
        'project': 'runs/train',
        'name': 'yolov8n_voc',
        'save': True,                     # 保存训练结果
        'exist_ok': True,                 # 允许覆盖现有项目
        'pretrained': True,               # 使用预训练权重
        'optimizer': 'auto',              # 自动选择最佳优化器
        'lr0': 0.01,                     # 初始学习率
        'cos_lr': True,                  # 使用余弦学习率调度
        'patience': 10,                   # 早停耐心值
        'box': 7.5,                       # box损失增益
        'cls': 0.5,                      # 分类损失增益
        'dfl': 1.5,                       # dfl损失增益
        # 'fl_gamma': 0.0,                # 已移除的参数
        'close_mosaic': 10,               # 最后10个epoch关闭mosaic增强
    }
    
    # 开始训练
    results = model.train(**train_args)
    
    # 验证模型 (自动使用验证集)
    val_results = model.val()
    print('\nValidation results:')
    print(f"mAP50-95: {val_results.box.map:.4f}")  # COCO mAP
    print(f"mAP50: {val_results.box.map50:.4f}")   # VOC-style mAP
    print(f"Precision: {val_results.box.precision:.4f}")
    print(f"Recall: {val_results.box.recall:.4f}")
    
    # 导出模型 (可选)
    model.export(format='onnx')  # 导出为ONNX格式
    
    return model

if __name__ == '__main__':
    trained_model = train_yolo_voc()