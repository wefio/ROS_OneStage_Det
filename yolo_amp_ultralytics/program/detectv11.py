# -*- coding: utf-8 -*-
from ultralytics import YOLO

class Detector(object):
    def __init__(self, model='yolo11n.pt', source='https://ultralytics.com/images/zidane.jpg', show=True, save=False):

        self.model = YOLO(model, task='detect')
        self.model.info()
        self.model.predict(stream=True)
        
        if save:
            self.result = self.model(source, save=True)
        else:
            self.result = self.model(source)
        
        if show and self.result:
            self.result[0].show()
    '''
    模型说明:
        ---------------此处YOLO后面带v------------------
        YOLOv3: yolov3u.pt      #u/tinyu/suppu
        YOLOv4: 不支持
        YOLOv5: yolov5n.pt/yolov5n6.pt or yolov5nu.pt/yolov5n6u.pt      #n/s/m/l/x
        YOLOv6: yolov6n.yaml    #n/s/m/l/x:没框，似乎需要自己训练模型
        YOLOv7: 不支持
        YOLOv8: yolov8n.pt      #n/s/m/l/x
        YOLOv9: yolov9c.pt      #t/s/m/c/e
        YOLOv10: yolov10n.pt    #n/s/m/b/l/x
        ---------------后面的YOLO去掉v------------------
        YOLOv11:yolo11n.pt      #n/s/m/l/x
        YOLOv12:yolo12n.pt      #n/s/m/l/x

    Examples:
            >>> from ultralytics import YOLO
            >>> if __name__ == "__main__":
                    Detector(
                    model='models/yolo11n.pt',
                    source='program/image.png',
                    show=True)

    '''

    def get_pos(self, target_name):
            """
            获取所有指定目标的坐标列表
            
            Args:
                target_name (str): 目标类别名称(如 'person')
            
            Returns:
                list[tuple]: [(x1,y1), (x2,y2), ...] 或空列表
            Examples:
                >>> detector.get_pos("person")
                [(948.2880859375, 376.4593505859375), (636.98046875, 459.01806640625)]
            """
            positions = []
            icon_id = None
            
            # 1. 查找类别ID
            if not hasattr(self.result[0], 'names'):
                return positions
            
            for class_id, name in self.result[0].names.items():
                if name == target_name:
                    icon_id = class_id
                    break
            
            if icon_id is None:
                return positions  # 类别不存在
            
            # 2. 遍历所有检测框，收集匹配的坐标
            boxes = self.result[0].boxes
            if not boxes:
                return positions
            
            for idx, cls in enumerate(boxes.cls):
                if cls == icon_id:
                    x = boxes.xywh[idx][0].item()
                    y = boxes.xywh[idx][1].item()
                    positions.append((x, y))
            
            return positions

    def print_target_position(self, target_name):
        """
        打印所有指定目标的坐标
        
        Args:
            target_name (str): 目标类别名称(如 'person')
        Examples:
            >>> detect = Detector()
            >>> detect.print_target_position("person")
            检测到 2 个 'person' 目标：
            目标 1 的坐标:x=948.29, y=376.46
            目标 2 的坐标:x=636.98, y=459.02
        """
        positions = self.get_pos(target_name)
        if positions:
            print(f"检测到 {len(positions)} 个 '{target_name}' 目标：")
            for i, (x, y) in enumerate(positions):
                print(f"目标 {i+1} 的坐标:x={x:.2f}, y={y:.2f}")
        else:
            print(f"未找到目标 '{target_name}'!")

if __name__ == "__main__":
    detector = Detector(
        model=r'runs\train\yolo11n_voc_fast\weights\best.pt',
        source='image',
        show=False,
        save=True
    )

