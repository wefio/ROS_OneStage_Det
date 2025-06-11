import os
import time
import json

import torch
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import RetinaNet
from backbone import resnet50_fpn_backbone, LastLevelP6P7
from draw_box_utils import draw_objs


def create_model(num_classes):
    # resNet50+fpn+retinanet
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     returned_layers=[2, 3, 4],
                                     extra_blocks=LastLevelP6P7(256, 256))
    model = RetinaNet(backbone, num_classes)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    # 注意：不包含背景
    model = create_model(num_classes=20)

    # load train weights
    weights_path = "./save_weights/resNetFpn-model-29.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # 创建结果文件夹
    os.makedirs("result", exist_ok=True)

    # 获取所有待处理图片
    image_dir = "image"
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 数据转换
    data_transform = transforms.Compose([transforms.ToTensor()])

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # 初始化模型（使用第一张图片的尺寸）
        if len(image_files) > 0:
            first_img_path = os.path.join(image_dir, image_files[0])
            first_img = Image.open(first_img_path)
            first_img_tensor = data_transform(first_img)
            init_img = torch.zeros((1, 3, first_img_tensor.shape[1], first_img_tensor.shape[2]), device=device)
            model(init_img)

        for image_file in image_files:
            print(f"Processing {image_file}...")
            # load image
            image_path = os.path.join(image_dir, image_file)
            original_img = Image.open(image_path)

            # from pil image to tensor, do not normalize image
            img = data_transform(original_img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print(f"Inference+NMS time for {image_file}: {t_end - t_start}")

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print(f"没有在 {image_file} 中检测到任何目标!")
                continue

            plot_img = draw_objs(original_img,
                                 predict_boxes,
                                 predict_classes,
                                 predict_scores,
                                 category_index=category_index,
                                 box_thresh=0.5,
                                 line_thickness=3,
                                 font='arial.ttf',
                                 font_size=20)
            
            # 保存预测的图片结果
            result_path = os.path.join("result", image_file)
            plot_img.save(result_path)
            print(f"Saved result to {result_path}")

    print("All images processed!")


if __name__ == '__main__':
    main()