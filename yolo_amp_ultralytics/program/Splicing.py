from PIL import Image
import os

def tile_images(image_paths, output_path, image_size=(512, 512)):
    """
    将四张图片按田字形（2x2）拼接成一张图片。

    :param image_paths: 四张图片的路径列表，顺序为 [左上, 右上, 左下, 右下]
    :param output_path: 输出图片的路径
    :param image_size: 每张图片的尺寸（默认 512x512）
    """
    # 确保有四张图片
    if len(image_paths) != 4:
        raise ValueError("必须提供四张图片！")

    # 加载图片
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = img.resize(image_size)
        images.append(img)

    # 创建新图像（2x2 布局）
    new_image = Image.new('RGB', (image_size[0] * 2, image_size[1] * 2))

    # 拼接图片
    new_image.paste(images[0], (0, 0))      # 左上
    new_image.paste(images[1], (image_size[0], 0))  # 右上
    new_image.paste(images[2], (0, image_size[1]))  # 左下
    new_image.paste(images[3], (image_size[0], image_size[1]))  # 右下

    # 保存结果
    new_image.save(output_path)
    print(f"已保存到 {output_path}")

# 示例用法
if __name__ == "__main__":
    # 替换为你的图片路径
    image_paths = [
        r'C:\Downloads\YOLO 实验结果\train\yolov3_voc_fast\Figure_1.png',
        r'C:\Downloads\YOLO 实验结果\train\yolov5n_voc_fast\Figure_1.png',
        r'C:\Downloads\YOLO 实验结果\train\yolov8n_voc_fast\Figure_1.png',
        r'C:\Downloads\YOLO 实验结果\train\yolo11n_voc_fast\Figure_1.png'
    ]
    output_path = 'combined_tile.png'

    tile_images(image_paths, output_path)