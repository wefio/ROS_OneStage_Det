import cv2
import os
import numpy as np
from natsort import natsorted
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def load_image(image_path):
    """多线程加载单张图片"""
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img, os.path.basename(image_path)
    except Exception as e:
        print(f"\n错误: 加载图片失败 {image_path} - {str(e)}")
        return None, None

def create_video_from_images(input_folder, output_file, fps=10, workers=4):
    """
    多线程将图片合成为视频
    
    参数:
        input_folder: 图片文件夹路径
        output_file: 输出视频路径
        fps: 帧率 (默认10)
        workers: 线程数 (默认4)
    """
    # 获取所有jpg文件并自然排序
    images = [img for img in os.listdir(input_folder) if img.lower().endswith(".jpg")]
    if not images:
        print("错误: 未找到任何jpg图片")
        return

    images = natsorted(images)
    image_paths = [os.path.join(input_folder, img) for img in images]
    
    # 预加载第一张图片获取尺寸
    first_img, _ = load_image(image_paths[0])
    if first_img is None:
        print("错误: 无法读取首张图片，请检查路径和文件格式")
        return

    height, width, _ = first_img.shape
    
    # 创建视频写入器 (H.264编码)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    if not video_writer.isOpened():
        print("错误: 无法创建视频文件，检查编码器或路径权限")
        return

    print(f"▶ 开始合成视频 (尺寸: {width}x{height}, 帧率: {fps}, 线程数: {workers})")
    
    # 多线程加载图片
    loaded_images = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(load_image, path) for path in image_paths]
        
        # 使用tqdm显示进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="加载图片"):
            img, name = future.result()
            if img is not None:
                loaded_images.append((name, img))

    # 按原始文件名顺序写入视频
    loaded_images.sort(key=lambda x: images.index(x[0]))
    for name, img in tqdm(loaded_images, desc="写入视频"):
        video_writer.write(img)

    video_writer.release()
    print(f"✔ 视频合成完成: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    # 配置参数
    input_folder = r"SSD 实验结果\ssd_results"  # 使用原始字符串避免转义
    output_file = "output_video.mp4"
    
    # 创建视频 (8线程加速)
    create_video_from_images(input_folder, output_file, fps=10, workers=8)