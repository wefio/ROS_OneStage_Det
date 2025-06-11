import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
data = pd.read_csv(r'C:\Downloads\YOLO 实验结果\train\yolo12n_voc_fast\results.csv')

# 提取相关列
epochs = data['epoch']
box_loss = data['train/box_loss']
cls_loss = data['train/cls_loss']
dfl_loss = data['train/dfl_loss']
lr = data['lr/pg0']

# 创建图形和主轴
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制 Loss 曲线（左侧 y 轴）
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(epochs, box_loss, label='Box Loss', color=color, linestyle='--')
ax1.plot(epochs, cls_loss, label='Cls Loss', color=color, linestyle='-.')
ax1.plot(epochs, dfl_loss, label='DFL Loss', color=color, linestyle=':')
ax1.tick_params(axis='y', labelcolor=color)

# 创建右侧 y 轴
ax2 = ax1.twinx()  # 共享 x 轴
color = 'tab:blue'
ax2.set_ylabel('Learning Rate', color=color)  # 设置右侧 y 轴标签
ax2.plot(epochs, lr, label='Learning Rate', color=color, linestyle='-')
ax2.tick_params(axis='y', labelcolor=color)

# 添加图例
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles=handles1 + handles2, labels=labels1 + labels2, loc='upper right')

# 设置标题
plt.title('Train Loss and Learning Rate')

# 显示图形
plt.tight_layout()
plt.show()