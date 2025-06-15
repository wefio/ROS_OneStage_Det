# 常用命令
## 安装软件
```
dpkg -i app_path
```

## 上一级目录
```
cd ..
```

## 小乌龟
```
ros2 run turtlesim turtlesim_node
```

## 小乌龟键盘控制
```
ros2 run turtlesim turtle_teleop_key
```

## 查看节点
```
ros2 node
ros2 node list
ros2 node info
```

## 查看话题
```
ros2 topic
ros2 topic list
ros2 topic bw /example  #查看带宽
ros2 topic echo /example    #查看话题内容
```

## 日志
```
ros2 bag
ros2 bag record /example
ros2 bag play file_path
```

# 功能包
- 代码丢到src文件夹

## 创建功能包
- 用于拆分功能
```
ros2 pkg create --build-type <build-type> <package_name>
ros2 pkg create --build-type  ament_cmake learning_pkg_c
ros2 pkg create --build-type  ament_python learning_pkg_python
```

## 编译功能包
```
colcon build #编译
source path/install/local_setup.bash #环境变量
```

# 运行节点
```
ros2 run <package_name> <executable_name> #包名 可执行文件名
```

## wafflepi gazebo
```
export TURTLEBOT3_MODEL=waffle_pi
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

## usb摄像头节点
```
sudo apt install ros-humble-usb-cam
ros2 run usb_cam usb_cam_node_exe
```

## 键盘控制节点
```
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

# 终止gazebo
```
killall -9 gzserver
```

# bag
```
ros2 bag record /camera/image_raw
ros2 bag play /home/ros/rosbag2_2025_04_26-23_56_14
```