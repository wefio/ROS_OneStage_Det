from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取参数文件路径
    params_file = os.path.join(
        get_package_share_directory('yolo_detector'),
        'config',
        'params.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='yolo_detector',
            executable='yolo_detector_node',
            name='yolo_detector_node',
            output='screen',
            parameters=[params_file]
        )
    ])