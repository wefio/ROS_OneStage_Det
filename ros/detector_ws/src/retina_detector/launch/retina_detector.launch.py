from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='retina_detector',
            executable='retina_detector_node',
            name='retina_detector_node',
            output='screen',
            parameters=[
                {'model_weights': 'install/retina_detector/share/retina_detector/src/resNetFpn-model-29.pth'},
                {'class_dict': 'install/retina_detector/share/retina_detector/src/pascal_voc_classes.json'},
                {'conf_threshold': 0.5},
                {'save_dir': 'retina_results'},
                {'image_topic': '/camera/image_raw'},
                {'result_topic': '/retina/detection_result'},
                {'fps_window_size': 10}
            ]
        )
    ])