from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ssd_detector',
            executable='ssd_detector_node',
            name='ssd_detector',
            output='screen',
            parameters=[
                {'model_weights': 'install/ssd_detector/share/ssd_detector/src/ssd300-29.pth'},
                {'class_dict': 'install/ssd_detector/share/ssd_detector/src/pascal_voc_classes.json'},
                {'conf_threshold': 0.5},
                {'save_dir': 'ssd_results'},
                {'image_topic': '/camera/image_raw'},
                {'result_topic': '/ssd/detection_result'},
            ]
        )
    ])