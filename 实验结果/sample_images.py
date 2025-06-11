import os
import random
import shutil

def sample_images(input_folder, output_folder, sample_ratio=0.1, seed=42):
    # 设置随机种子
    random.seed(seed)

    # 确保输出目录存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有图片文件（支持常见图像格式）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    image_files = []

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file_path)

    # 计算需要抽样的数量
    num_samples = int(len(image_files) * sample_ratio)
    if num_samples == 0:
        print("Warning: No images to sample.")
        return

    # 随机选择样本
    sampled_files = random.sample(image_files, num_samples)

    # 复制选中的图片到输出文件夹
    for src_path in sampled_files:
        filename = os.path.basename(src_path)
        dst_path = os.path.join(output_folder, filename)
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {src_path} -> {dst_path}")

    print(f"Total {num_samples} images sampled and copied to {output_folder}.")

# 示例用法
if __name__ == "__main__":
    input_dir = r"YOLO 实验结果\yolo_raw11"  # 替换为你的输入文件夹路径
    output_dir = r"YOLO 实验结果\yolo_raw11_sample"  # 替换为你的输出文件夹路径
    sample_images(input_dir, output_dir, sample_ratio=0.1, seed=42)