from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'retina_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'src'), glob('src/*.*')),
    ],
    install_requires=['setuptools', 'torch', 'torchvision'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='RetinaNet Object Detection ROS2 Node',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'retina_detector_node = retina_detector.retina_detector_node:main',
        ],
    },
)