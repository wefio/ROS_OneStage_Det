#!/usr/bin/env python3
import sys
import argparse
import subprocess
import time
import re
import threading
import psutil
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

'''
Example:
python3 rosbag2video.py   --topic /camera/image_raw   --output output.mp4   --fps 30   --crf 21
'''

class Rosbag2Video(Node):
    def __init__(self, topic, output, fps, crf=21, timeout=5.0):
        super().__init__('rosbag2video')
        self.bridge = CvBridge()
        self.active = False               # 录制状态标志
        self.monitor_running = True       # 监控线程运行标志
        self.last_msg_time = 0.0          # 最后消息时间戳
        self.timeout = timeout            # 无消息超时时间
        self.frame_count = 0              # 已处理帧数
        self.resolution = None            # 动态分辨率存储
        self.start_time = time.time()     # 程序启动时间
        self.last_progress_log = 0.0      # 最后进度日志时间
        self.last_progress_data = {}      # 进度数据缓存

        # FFmpeg参数模板（动态分辨率填充）
        self.ffmpeg_template = [
            'ffmpeg',
            '-y',                        # 覆盖输出文件
            '-f', 'rawvideo',            # 输入格式
            '-pix_fmt', 'bgr24',         # OpenCV默认格式
            '-s', '{}x{}',               # 动态分辨率占位符
            '-r', str(fps),              # 输入帧率
            '-i', '-',                   # 从标准输入读取
            '-vf', 'format=yuv420p',     # 转换到兼容格式
            '-c:v', 'libx264',           # 编码器选择
            '-preset', 'medium',         # 平衡编码速度和质量
            '-crf', str(crf),            # 质量参数
            '-movflags', '+faststart',   # 流式播放优化
            '-progress', 'pipe:1',       # 进度输出通道
            '-nostats',                  # 禁用冗余统计
            '-loglevel', 'info',         # 日志级别
            output                       # 输出文件路径
        ]

        # 配置高可靠性QoS
        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        
        # 创建话题订阅
        self.sub = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            qos_profile=qos_profile
        )
        self.get_logger().info(f"🕙 等待话题数据: {topic}...")

        # 初始化定时器和线程
        self.timer = self.create_timer(1.0, self.check_timeout)
        self.log_thread = None
        self.monitor_thread = None

    def image_callback(self, msg):
        """ 图像消息回调函数 """
        try:
            if not self.active:
                # 初始化阶段：验证第一条消息
                self.resolution = (msg.width, msg.height)
                if self.resolution[0] % 2 != 0 or self.resolution[1] % 2 != 0:
                    raise ValueError(f"⚠️ 分辨率需为偶数，当前：{self.resolution[0]}x{self.resolution[1]}")
                
                # 构建FFmpeg命令
                ffmpeg_args = [arg.format(*self.resolution) if '{}' in arg else arg 
                              for arg in self.ffmpeg_template]
                self.ffmpeg = subprocess.Popen(
                    ffmpeg_args,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )
                self.active = True
                self.get_logger().info(f"开始录制 | 分辨率: {self.resolution[0]}x{self.resolution[1]}")

                # 启动日志处理线程
                self.log_thread = threading.Thread(target=self.process_ffmpeg_output)
                self.log_thread.daemon = True
                self.log_thread.start()

                # 启动系统监控线程
                self.monitor_thread = threading.Thread(target=self.system_monitor)
                self.monitor_thread.daemon = True
                self.monitor_thread.start()

            # 处理图像数据
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 验证分辨率一致性
            if (cv_image.shape[1], cv_image.shape[0]) != self.resolution:
                raise ValueError(f"⚠️ 分辨率变化！初始 {self.resolution}，当前 {cv_image.shape[1]}x{cv_image.shape[0]}")

            if self.ffmpeg.stdin:
                self.ffmpeg.stdin.write(cv_image.tobytes())
                self.frame_count += 1
                self.last_msg_time = time.time()

        except Exception as e:
            self.get_logger().error(f"❌ 处理失败: {str(e)}")
            self.cleanup()

    def process_ffmpeg_output(self):
        """ 处理FFmpeg进度输出 """
        bitrate_pattern = re.compile(r'(bitrate|total_bitrate)=(\d+\.?\d*)(k|m)?bits?/?s?')
        while self.ffmpeg and self.ffmpeg.poll() is None:
            try:
                raw_line = self.ffmpeg.stdout.readline()
                line = raw_line.decode(errors='replace').strip()
                if not line:
                    continue

                # 解析进度参数
                params = dict(re.findall(r'(\w+)=([^\s/]+)', line))
                
                # 更新缓存数据
                self.last_progress_data.update({
                    'frame': params.get('frame', self.last_progress_data.get('frame', 'N/A')),
                    'speed': params.get('speed', self.last_progress_data.get('speed', 'N/A')).rstrip('x'),
                })

                # 解析比特率
                bitrate_match = bitrate_pattern.search(line)
                if bitrate_match:
                    value, unit = float(bitrate_match.group(2)), bitrate_match.group(3)
                    self.last_progress_data['bitrate'] = f"{value*1000 if unit == 'm' else value:.0f}kbps"
                elif 'bitrate' in params:
                    self.last_progress_data['bitrate'] = params['bitrate']

                # 5秒节流输出
                current_time = time.time()
                if current_time - self.last_progress_log >= 5:
                    cpu_percent = psutil.cpu_percent()
                    mem = psutil.virtual_memory()
                    
                    self.get_logger().info(
                        f"📊 编码进度 | "
                        f"帧数: {self.last_progress_data.get('frame', 'N/A')} | "
                        f"速度: {self.last_progress_data.get('speed', 'N/A')}x | "
                        f"比特率: {self.last_progress_data.get('bitrate', 'N/A')} | "
                        f"CPU: {cpu_percent}% | "
                        f"内存: {mem.percent}%"
                    )
                    self.last_progress_log = current_time
                    self.last_progress_data.clear()

            except Exception as e:
                self.get_logger().warning(f"⚠️ 进度解析异常: {str(e)}")

    def system_monitor(self):
        """ 系统资源监控 """
        while self.monitor_running:
            try:
                # 分段检测退出标志
                for _ in range(5):
                    if not self.monitor_running:
                        return
                    time.sleep(1)

                runtime = time.time() - self.start_time
                self.get_logger().info(
                    f"系统监控 | "
                    f"运行时间: {runtime:.1f}s | "
                    f"CPU: {psutil.cpu_percent()}% | "
                    f"内存: {psutil.virtual_memory().percent}%",
                    throttle_duration_sec=5
                )
            except Exception as e:
                self.get_logger().warning(f"⚠️ 资源监控异常: {str(e)}")

    def check_timeout(self):
        """ 超时检测 """
        if not self.active:
            # 初始等待超时
            if time.time() - self.start_time > 120:
                self.get_logger().error("⏳ 等待初始消息超时（120秒）")
                self.cleanup()
            return

        # 录制中无消息超时
        if time.time() - self.last_msg_time > self.timeout:
            self.get_logger().warning(f"⏳ 无新消息超过 {self.timeout} 秒，自动停止")
            self.cleanup()

    def cleanup(self, exit_code=0):
        """ 资源清理 """
        if hasattr(self, '_cleaning'):
            return
        self._cleaning = True

        # 设置停止标志
        self.active = False
        self.monitor_running = False

        # 关闭FFmpeg进程
        if self.ffmpeg:
            if self.ffmpeg.poll() is None:
                try:
                    self.ffmpeg.stdin.close()
                    self.ffmpeg.wait(timeout=5)
                except Exception as e:
                    self.get_logger().error(f"❌ 关闭FFmpeg失败: {str(e)}")
                finally:
                    self.get_logger().info(f"✅ 视频已保存 | 总帧数: {self.frame_count} | ctrl+C退出")

        # 等待线程退出
        threads = [t for t in [self.monitor_thread, self.log_thread] if t]
        for t in threads:
            if t.is_alive():
                t.join(timeout=2)

        # 关闭ROS节点
        if rclpy.ok():
            self.destroy_node()
            rclpy.shutdown()

        sys.exit(exit_code)

def main():
    parser = argparse.ArgumentParser(
        description='ROS2 Bag视频转换工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--topic', required=True, help='图像话题名称（例如：/camera/image_raw）')
    parser.add_argument('--output', default='output.mp4', help='输出视频路径')
    parser.add_argument('--fps', type=int, default=30, help='目标帧率（需与原始数据匹配）')
    parser.add_argument('--crf', type=int, default=21, help='质量系数（0-51，值越小质量越高）')
    parser.add_argument('--timeout', type=float, default=5.0, help='无消息超时时间（秒）')

    args = parser.parse_args(rclpy.utilities.remove_ros_args(sys.argv[1:]))

    # 参数验证
    if args.crf < 0 or args.crf > 51:
        parser.error("❌ CRF值需在0-51范围内")
    if args.timeout < 1.0:
        parser.error("❌ 超时时间至少1秒")

    rclpy.init()
    node = Rosbag2Video(
        topic=args.topic,
        output=args.output,
        fps=args.fps,
        crf=args.crf,
        timeout=args.timeout
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 用户中断操作")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
