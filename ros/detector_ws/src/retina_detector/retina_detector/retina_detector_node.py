#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import json
import time
import torch
import numpy as np
from PIL import Image as PILImage
import csv
from datetime import datetime
from torchvision import transforms

from .network_files import RetinaNet
from .backbone import resnet50_fpn_backbone, LastLevelP6P7
from .draw_box_utils import draw_objs

class RetinaDetectorNode(Node):
    def __init__(self):
        super().__init__('retina_detector_node')
        
        # 参数初始化
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_weights', 'src/resNetFpn-model-29.pth'),
                ('class_dict', 'src/pascal_voc_classes.json'),
                ('conf_threshold', 0.5),
                ('save_dir', 'retina_results'),
                ('image_topic', '/camera/image_raw'),
                ('result_topic', '/retina/detection_result'),
                ('fps_window_size', 10),
            ]
        )
        
        # 获取参数
        self.weights_path = self.get_parameter('model_weights').value
        self.class_dict_path = self.get_parameter('class_dict').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.save_dir = self.get_parameter('save_dir').value
        self.image_topic = self.get_parameter('image_topic').value
        self.result_topic = self.get_parameter('result_topic').value
        self.fps_window_size = self.get_parameter('fps_window_size').value
        
        # 初始化设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")
        
        # 初始化模型
        self.num_classes = 20  # VOC 20类，不包含背景
        self.model = self._init_model()
        
        # 加载类别字典
        self.category_index = self._load_class_dict()
        
        # 初始化数据变换
        self.data_transform = transforms.Compose([transforms.ToTensor()])
        
        # 初始化CV Bridge
        self.bridge = CvBridge()
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化统计变量
        self.frame_count = 0
        self.detection_times = []
        self.frame_times = []
        self.fps_values = []
        self.first_detection_time = None
        
        # 创建CSV文件记录结果
        self.csv_file = os.path.join(self.save_dir, 'detection_results.csv')
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'image_path', 'detection_time_ms', 
                'fps', 'num_detections', 'class_names', 'confidences'
            ])
        
        # 创建订阅者和发布者
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10)
        self.result_pub = self.create_publisher(Image, self.result_topic, 10)
        
        # 创建定时器用于计算和发布FPS
        self.timer = self.create_timer(1.0, self.update_fps)
        
        self.get_logger().info("RetinaNet Detector Node initialized")

    def _init_model(self):
        """初始化RetinaNet模型"""
        backbone = resnet50_fpn_backbone(
            norm_layer=torch.nn.BatchNorm2d,
            returned_layers=[2, 3, 4],
            extra_blocks=LastLevelP6P7(256, 256)
        )
        model = RetinaNet(backbone, self.num_classes)
        
        # 加载权重
        assert os.path.exists(self.weights_path), f"Weights file {self.weights_path} not found"
        weights_dict = torch.load(self.weights_path, map_location='cpu')
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        model.load_state_dict(weights_dict)
        model.to(self.device)
        
        # 初始化模型
        model.eval()
        init_img = torch.zeros((1, 3, 800, 800), device=self.device)  # 假设输入尺寸为800x800
        model(init_img)
        
        self.get_logger().info("RetinaNet model initialized")
        return model
    
    def _load_class_dict(self):
        """加载类别字典"""
        assert os.path.exists(self.class_dict_path), f"Class dict file {self.class_dict_path} not found"
        with open(self.class_dict_path, 'r') as f:
            class_dict = json.load(f)
        return {str(v): str(k) for k, v in class_dict.items()}
    
    def _time_synchronized(self):
        """同步时间"""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return time.time()
    
    def update_fps(self):
        """计算并更新FPS"""
        if len(self.frame_times) >= 2:
            time_diff = np.diff(self.frame_times[-self.fps_window_size:])
            if len(time_diff) > 0:
                current_fps = 1.0 / np.mean(time_diff)
                self.fps_values.append(current_fps)
                self.get_logger().info(f"Current FPS: {current_fps:.2f}")
    
    def image_callback(self, msg):
        try:
            current_time = time.time()
            self.frame_times.append(current_time)
            
            # 确保模型在eval模式
            self.model.eval()
            
            # 转换图像格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # 预处理图像
            img = self.data_transform(pil_image)
            img = torch.unsqueeze(img, dim=0)
            
            # 执行检测
            start_time = self._time_synchronized()
            with torch.no_grad():
                predictions = self.model(img.to(self.device))[0]
            detection_time = (self._time_synchronized() - start_time) * 1000  # 毫秒
            
            # 记录第一次检测时间
            if self.first_detection_time is None:
                self.first_detection_time = start_time
            
            # 计算当前FPS
            current_fps = 0.0
            if len(self.frame_times) >= 2:
                time_diff = current_time - self.frame_times[-2] if len(self.frame_times) >= 2 else 0.1
                current_fps = 1.0 / time_diff if time_diff > 0 else 0.0
            
            # 处理检测结果
            boxes = predictions["boxes"].to("cpu").numpy()
            classes = predictions["labels"].to("cpu").numpy()
            scores = predictions["scores"].to("cpu").numpy()
            
            # 过滤低置信度检测
            mask = scores >= self.conf_threshold
            boxes = boxes[mask]
            classes = classes[mask]
            scores = scores[mask]
            
            # 生成时间戳和文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_filename = f"detection_{timestamp}.jpg"
            img_path = os.path.join(self.save_dir, img_filename)
            
            # 绘制并保存结果
            if len(boxes) > 0:
                result_img = draw_objs(
                    pil_image,
                    boxes,
                    classes,
                    scores,
                    category_index=self.category_index,
                    box_thresh=self.conf_threshold,
                    line_thickness=3,
                    font='arial.ttf',
                    font_size=20
                )
                
                # 保存图像和结果
                result_img.save(img_path)
                
                # 记录到CSV
                with open(self.csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp,
                        img_path,
                        detection_time,
                        current_fps,
                        len(boxes),
                        [self.category_index[str(c)] for c in classes],
                        scores.tolist()
                    ])
                
                # 发布结果图像
                result_cv = cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)
                result_msg = self.bridge.cv2_to_imgmsg(result_cv, encoding="bgr8")
                result_msg.header = msg.header
                self.result_pub.publish(result_msg)
            
            self.frame_count += 1
            self.detection_times.append(detection_time)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = RetinaDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    # 输出统计信息
    if node.detection_times:
        avg_time = np.mean(node.detection_times)
        avg_fps = np.mean(node.fps_values) if node.fps_values else 0.0
        total_time = (time.time() - node.first_detection_time) if node.first_detection_time else 0
        node.get_logger().info("\n=== Detection Statistics ===")
        node.get_logger().info(f"Total frames processed: {node.frame_count}")
        node.get_logger().info(f"Total processing time: {total_time:.2f}s")
        node.get_logger().info(f"Average detection time: {avg_time:.2f}ms")
        node.get_logger().info(f"Average FPS: {avg_fps:.2f}")
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()