# 基于深度学习的One-Stage机器人目标检测算法研究

[TOC]

## 前言

这是一个本科生的毕业设计，主要用于存档<br>

因为没找到能直接用的，只能手搓，能力有限<br>

受到了互联网上很多网友的帮助，希望能把分享的互联网精神传递下去<br>

只接受免费传播，不允许售卖<br>

## 摘要

主要将Ultralytics、SSD、RetinaNet分别训练、缝了个ROS节点、进行简单的性能测试。<del>结论为YOLO以外的那俩在算力比较差的环境下可以埋了，根本跑不动。</del><br>

环境使用Ubuntu22.04，ROS2 Humble，Ultralytics-8.3.110，PyTorch12.8。测试场景使用自己搭建的地图<br>
<video width="640" height="360" controls>
  <source src="https://github.com/wefio/ROS_OneStage_Det/blob/main/%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C/one-stage_1_2025.5.30-13.48.17.mp4" type="video/mp4">
  你的浏览器不支持视频标签。
</video>

## 模型训练

珍惜时间，珍爱生命，建议使用算力云平台进行训练<br>

使用PyTorch12.8，Ultralytics-8.3.110<br>

### YOLO

**You Only Look Once**

使用Ultralytics框架。

Ultralytics官网：https://docs.ultralytics.com/zh/

入门教程参考：https://www.bilibili.com/video/BV1A56bYtEFR/，其中的装饰器，可以参考：https://www.bilibili.com/video/BV1Gu411Q7JV/

Ultralytics仓库：https://github.com/ultralytics/ultralytics

YOLO发展：https://www.bilibili.com/video/BV1oN4113717/

YOLO理论：https://www.bilibili.com/video/BV1yi4y1g7ro/

#### 目录结构：

yolo_amp_ultralytics
	|--ultralytics-8.3.110   ultralytics仓库     
	|--datasets          训练数据存放的地方     
	|--program          使用的程序     
	|-- models          下载的模型     
	|-- runs            训练/测试/验证/预测…结果     
	|-- windows_v1.8.1     手动标注工具     
	|--其他不重要


yolo_amp_ultralytics是主文件夹，ultralytics-8.3.110是Ultralytics仓库。

终端cd切换到ultralytics-8.3.110目录，然后是用pip install -e . 进行本地安装。

然后进入program，运行detectv11.py，自带一个示例程序，可以检查安装有没有问题。

普通训练可以使用train.py，觉得慢可以使用train_turbo.py

predict.py是一个预测程序。

Validate.py是一个验证集程序，benchmark.py是性能评估程序。如果遇到报错，解决办法见https://zhuanlan.zhihu.com/p/1892947083977270162

其他的程序是一些玩具。

#### 超参数说明（看注释也行）：

model = YOLO(<u>'yolov8n.pt</u>')		*改成想要训练的模型*

'data': '<u>datasets/VOC/VOC.yaml</u>',	*训练集模板路径*

'epochs': <u>50</u>,						*训练轮数*

'batch': <u>32</u> if device == '0' else 4,     	*batch，3060 6G可以到32,4090可以到96或128（128无明显训练速度提示）*

'workers': <u>8</u>,                    			*填cpu核心数*

'project': '<u>runs/train</u>', 			*训练模型保存到的文件夹*

'name': '<u>yolov8n_voc</u>', 			*训练模型保存到的文件夹名*

### SSD和RetinaNet

**SSD**

**Single Shot MultiBox Detector**

理论：https://www.bilibili.com/video/BV1fT4y1L7Gi/

源码解析：https://www.bilibili.com/video/BV1vK411H771/

源码：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/ssd

 **RetinaNet**

**Focal Loss for Dense Object Detection**

理论：RetinaNet：https://www.bilibili.com/video/BV1vK411H771/

源码：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/retinaNet

#### 文件说明：

**ssd**：train_ssd300.py用于训练模型。

**retinanet**：train.py用于训练模型。

其余部分两个模型一样

validation.py用于验证，validation_tensorboard.py加入了tensorboard，并没有什么用。

predict.py用于预测。

#### 训练/验证超参数

因为训练和验证使用的超参数类似，这里用训练的超参数举例。

parser.add_argument('--data-path', default='<u>dataset</u>', help='dataset') 	 *dataset改成训练/测试集路径。*

parser.add_argument('--num-classes', default=<u>20</u>, type=int, help='num_classes')	 *根据训练/测试集种类进行修改*

parser.add_argument('--epochs', default=<u>5</u>, type=int, metavar='N',help='number of total epochs to run')	*修改epoch数（训练轮数），如果只是看一下程序能不能用建议改成1，训练模型建议30*

parser.add_argument('--batch_size', default=<u>16</u>, type=int, metavar='N', help='batch size when training.')	*修改batch数*，以下是GPU训练的参数，CPU不知道

ssd：3060 6G可以用16或32，4090可以到64，内存16G的建议4-8容易爆内存，训练速度较为正常。

retinanet：3060 6G用4,4090用32，改大了会爆显存，内存16G的不一定能训练，训练速度非常慢。

## ROS

### ROS2

ROS2教程： 

古月·ROS2入门21讲https://www.bilibili.com/video/BV16B4y1Q7jQ/

​           图文教程https://book.guyuehome.com/

​           教程源码：  Gitee：https://gitee.com/guyuehome/ros2_21_tutorials

Github：https://github.com/guyuehome/ros2_21_tutorials

### ROS及Linux常用命令

见detector_ws/笔记.md

 ### 程序使用说明

算力资源有限，所以实验思路是将Gazebo仿真和模型检测分开。

#### 目录结构

ros
	|--detector_ws			工作区
	|--src					代码包
		|--benchmark		地图 没错，这里是地图，名字是瞎起的
		|--retina_detector	retina
		|--ssd_detector		ssd
		|--yolo_detector		yolo


### world地图

```bash
cd detector_ws
colcon build #编译
source install/local_setup.bash #环境变量

ros2 launch benchmark city.launch.py #运行地图
```

#### 导入机器人模型、遥控和bag录制

```bash
export TURTLEBOT3_MODEL=waffle_pi #导入waffle_pi

ros2 run teleop_twist_keyboard teleop_twist_keyboard #使用键盘遥控

ros2 bag record /camera/image_raw #使用bag录制/camera/image_raw
```

新开一个终端，使用录制的bag
```bash
ros2 bag play /home/ros/rosbag2_2025_04_26-23_56_14 #使用录制的bag，文件名应该不同
```

#### rosbag topic转mp4

/其他/rosbag2video.py

先运行脚本再运行rosbag。基于ffmpeg。

```bash
python3 rosbag2video.py  --topic /camera/image_raw  --output output.mp4  --fps 30  --crf 21
```

### 运行ros节点

```bash
cd detector_ws
colcon build #编译
source path/install/local_setup.bash #环境变量
```

**摄像头话题**：新开一个终端，打开rviz，话题是/camera/image_raw的Image

```bash
rviz2
```

**运行YOLO**

```bash
ros2 launch yolo_detector yolo_detector.launch.py #yolo运行
```

<p style="color:red">会保存为yolo_raw，会覆盖，记得保存</p>

**修改yolo模型**

打开detector_ws\src\yolo_detector\config\params.yaml，修改model_path: "yolo11n.pt"



**SSD**

会保存为detector_ws/ssd_results

```bash

ros2 launch ssd_detector.launch.py    #SSD运行
```

**RetinaNet**

会保存为detector_ws/retina_results

```bash
ros2 launch retina_detector.launch.py  #RetinaNet运行
```

**修改ssd和retinanet模型**

打开detector_ws\src\\<u>retina_detector</u>\src，替换resNetFpn-model-29.pth

ssd的类似

## 其他工具 
/实验结果/video.py	*将文件夹中图片转为mp4*

/实验结果/sample_images.py	*文件夹中图片随机抽取其中10%*

使用ps“联系表”批量拼图https://www.bilibili.com/video/BV1VZ4y1r7hy/



## 补充

**值得一看的内容：**

卷积神经网络基础https://www.bilibili.com/video/BV1b7411T7DA/

基础看完可以看LeNet、AlexNet、ResNet（这三个是CNN）、 VGG。

论文精读：跟李沐学AI https://space.bilibili.com/1567748478

Python基础：https://space.bilibili.com/3546660493855256/lists/4227087

手搓一个简单的CNN：https://space.bilibili.com/3546660493855256/lists/3283204

**其他：**

晚上12点后使用deepseek，不会出现 服务器繁忙，请稍后再试

文献管理工具推荐：Zotero
