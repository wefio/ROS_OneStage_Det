#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import csv
import time
from datetime import datetime
from ultralytics import YOLO
from typing import List, Dict, Tuple
import numpy as np

class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        
        # 声明参数（使用推荐的新方式）
        self.declare_parameter('model_path', rclpy.Parameter.Type.STRING)
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.5)
        self.declare_parameter('save_dir', 'yolo_raw')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('publish_topic', '/yolo/detection_result')
        self.declare_parameter('publish_freq', 10.0)
        
        # 获取参数
        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.save_dir = self.get_parameter('save_dir').value
        self.image_topic = self.get_parameter('image_topic').value
        self.publish_topic = self.get_parameter('publish_topic').value
        self.publish_freq = self.get_parameter('publish_freq').value
        
        # 初始化YOLO模型 - 支持自定义路径
        try:
            if os.path.isabs(model_path) or os.path.exists(model_path):
                # 使用绝对路径或相对路径指定的模型
                self.model = YOLO(model_path)
                self.get_logger().info(f"Loaded YOLO model from: {model_path}")
            else:
                # 尝试加载预训练模型
                self.model = YOLO(model_path)
                self.get_logger().info(f"Loaded pretrained YOLO model: {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {str(e)}")
            raise
        
        # 初始化CV Bridge
        self.bridge = CvBridge()
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化变量
        self.last_time = None
        self.frame_count = 0
        self.fps = 0.0
        self.detection_times = []
        self.first_detection_time = None
        
        # 创建CSV文件并写入头
        self.csv_file = os.path.join(self.save_dir, 'detection_results.csv')
        with open(self.csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'image_filename', 'detection_time_ms', 
                'fps', 'num_detections', 'class_ids', 'confidences'
            ])
        
        # 创建订阅者
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )
        self.get_logger().info(f"Subscribed to {self.image_topic}")
        
        # 创建发布者
        self.result_pub = self.create_publisher(
            Image,
            self.publish_topic,
            10
        )
        self.get_logger().info(f"Publishing results to {self.publish_topic}")
        
        # 创建定时器用于计算FPS
        self.timer = self.create_timer(1.0, self.update_fps)
    
    def update_fps(self):
        if self.frame_count > 0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.get_logger().info(f"Current FPS: {self.fps:.2f}")
    
    def image_callback(self, msg: Image):
        try:
            # 转换图像消息为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 记录开始检测时间
            start_time = time.time()
            
            # 使用YOLO进行检测
            results = self.model.predict(
                cv_image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # 计算检测时间
            detection_time = (time.time() - start_time) * 1000  # 转换为毫秒
            self.detection_times.append(detection_time)
            
            # 记录第一次检测时间
            if self.first_detection_time is None:
                self.first_detection_time = start_time
                self.get_logger().info("First detection completed")
            
            # 处理检测结果
            if len(results) > 0:
                result = results[0]
                
                # 绘制检测框并获取标注后的图像
                annotated_image = result.plot()  # 这是带标注的图像
                
                # 获取检测信息
                num_detections = len(result.boxes)
                class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()
                confidences = result.boxes.conf.cpu().numpy().tolist()
                
                # 生成时间戳和文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                
                # 保存带标注的图像
                annotated_filename = f"annotated_{timestamp}.jpg"
                annotated_path = os.path.join(self.save_dir, annotated_filename)
                cv2.imwrite(annotated_path, annotated_image)
                
                # 可选：保存原始图像（如果需要）
                # original_filename = f"original_{timestamp}.jpg"
                # original_path = os.path.join(self.save_dir, original_filename)
                # cv2.imwrite(original_path, cv_image)
                
                # 保存结果到CSV
                with open(self.csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp, annotated_filename, detection_time,
                        self.fps, num_detections, class_ids, confidences
                    ])
                
                # 发布结果图像
                result_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
                result_msg.header = msg.header
                self.result_pub.publish(result_msg)
            
            # 更新帧计数
            self.frame_count += 1
            
        except Exception as e:
            self.get_logger().error(f"Error in image processing: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    # 节点关闭时输出统计信息
    if node.detection_times:
        avg_time = np.mean(node.detection_times)
        max_time = np.max(node.detection_times)
        min_time = np.min(node.detection_times)
        total_time = (time.time() - node.first_detection_time) if node.first_detection_time else 0
        
        node.get_logger().info("\n=== Detection Statistics ===")
        node.get_logger().info(f"Total detections: {len(node.detection_times)}")
        node.get_logger().info(f"Total processing time: {total_time:.2f} seconds")
        node.get_logger().info(f"Average detection time: {avg_time:.2f} ms")
        node.get_logger().info(f"Max detection time: {max_time:.2f} ms")
        node.get_logger().info(f"Min detection time: {min_time:.2f} ms")
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()