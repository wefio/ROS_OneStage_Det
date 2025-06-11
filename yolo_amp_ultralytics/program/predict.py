from ultralytics import YOLO
import pyautogui as pg

class GetPos:
    @classmethod
    def init(cls, source):
        yolo=YOLO(model='yolo11n.pt', task='detect')
        cls.result = yolo(source=source,save=True)
    @classmethod
    def __get_icon_id(cls, icon_name:str): #指定名字的id
        for i in cls.result[0].names:
            if cls.result[0].names[i] == icon_name:
                return i
    @classmethod
    def __get_cls_no(cls, icon_id):    #指定id第几位出现
        for idx,item in enumerate(cls.result[0].boxes.cls):
            if item == icon_id:
                return idx
    @classmethod
    def __get_xy(cls, cls_no):   #指定第几位的坐标
        x = cls.result[0].boxes.xywh[cls_no][0]
        y = cls.result[0].boxes.xywh[cls_no][1]
        return x,y
    @classmethod
    def get_pos(cls, icon_name:str): #获取坐标
        icon_id = cls.__get_icon_id(icon_name)
        cls_no = cls.__get_cls_no(icon_id)
        pos = cls.__get_xy(cls_no)
        return pos
    
if __name__ == "__main__":
    pg.screenshot('shot.png')
    GetPos.init('shot.png')
    person_pos = GetPos.get_pos('person')
    car_pos = GetPos.get_pos('car')
    print(person_pos, car_pos)
    



r'ultralytics-8.3.55\detect-footage.jpg'