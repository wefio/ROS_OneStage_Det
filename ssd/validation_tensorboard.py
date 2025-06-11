"""
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
以及每个类别的mAP(IoU=0.5)
加入TensorBoard可视化功能
"""

import os
import json
from datetime import datetime
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

import transforms
from src import Backbone, SSD300
from my_dataset import VOCDataSet
from train_utils import get_coco_api_from_dataset, CocoEvaluator


def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 初始化TensorBoard
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", "eval", current_time)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # 初始化时间和FPS统计
    total_inference_time = 0
    total_images = 0

    data_transform = {
        "val": transforms.Compose([transforms.Resize(),
                                 transforms.ToTensor(),
                                 transforms.Normalization()])
    }

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {v: k for k, v in class_dict.items()}

    VOC_root = parser_data.data_path
    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # load validation data set
    val_dataset = VOCDataSet(VOC_root, "2012", transforms=data_transform["val"], train_set="val.txt")
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=nw,
                                                   pin_memory=True,
                                                   collate_fn=val_dataset.collate_fn)

    # create model
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=parser_data.num_classes + 1)

    # 载入你自己训练好的模型权重
    weights_path = parser_data.weights
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)
    model.eval()

    # warmup GPU
    print("Warming up GPU...")
    dummy_input = torch.rand(1, 3, 300, 300).to(device)
    for _ in range(10):
        _ = model(dummy_input)

    # evaluate on the test dataset
    coco = get_coco_api_from_dataset(val_dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    cpu_device = torch.device("cpu")

    print("Starting evaluation...")
    with torch.no_grad():
        for images, targets in tqdm(val_dataset_loader, desc="validation..."):
            # 将图片传入指定设备device
            images = torch.stack(images, dim=0).to(device)
            
            # 记录推理开始时间
            start_time = time.time()
            
            # inference
            results = model(images)
            
            # 记录推理时间
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            total_images += len(images)

            outputs = []
            for index, (bboxes_out, labels_out, scores_out) in enumerate(results):
                # 将box的相对坐标信息（0-1）转为绝对值坐标(xmin, ymin, xmax, ymax)
                height_width = targets[index]["height_width"]
                bboxes_out[:, [0, 2]] = bboxes_out[:, [0, 2]] * height_width[1]
                bboxes_out[:, [1, 3]] = bboxes_out[:, [1, 3]] * height_width[0]

                info = {"boxes": bboxes_out.to(cpu_device),
                        "labels": labels_out.to(cpu_device),
                        "scores": scores_out.to(cpu_device)}
                outputs.append(info)

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)

    # 计算FPS和时间统计
    avg_inference_time = total_inference_time / total_images
    fps = 1.0 / avg_inference_time
    
    print("\nInference Performance:")
    print(f"Total images processed: {total_images}")
    print(f"Total inference time: {total_inference_time:.4f} seconds")
    print(f"Average inference time per image: {avg_inference_time:.4f} seconds")
    print(f"FPS: {fps:.2f}")

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_eval = coco_evaluator.coco_eval["bbox"]
    coco_stats, print_coco = summarize(coco_eval)
    
    # 记录指标到TensorBoard
    writer.add_scalar('Evaluation/mAP_0.5', coco_stats[1], 0)
    writer.add_scalar('Evaluation/mAP_0.75', coco_stats[2], 0)
    writer.add_scalar('Evaluation/mAP_small', coco_stats[3], 0)
    writer.add_scalar('Evaluation/mAP_medium', coco_stats[4], 0)
    writer.add_scalar('Evaluation/mAP_large', coco_stats[5], 0)
    writer.add_scalar('Performance/FPS', fps, 0)
    writer.add_scalar('Performance/Inference_time', avg_inference_time, 0)

    # 计算每个类别的AP
    voc_map_info_list = []
    for i in range(len(category_index)):
        stats, _ = summarize(coco_eval, catId=i)
        voc_map_info_list.append(" {:15}: {}".format(category_index[i + 1], stats[1]))
        writer.add_scalar(f'Class_AP/{category_index[i + 1]}', stats[1], 0)

    print_voc = "\n".join(voc_map_info_list)
    print(print_voc)

    # 保存结果到文件
    record_path = os.path.join(log_dir, "evaluation_results.txt")
    with open(record_path, "w") as f:
        record_lines = [
            "COCO results:",
            print_coco,
            "",
            "Performance Metrics:",
            f"Total images processed: {total_images}",
            f"Total inference time: {total_inference_time:.4f} seconds",
            f"Average inference time per image: {avg_inference_time:.4f} seconds",
            f"FPS: {fps:.2f}",
            "",
            "mAP(IoU=0.5) for each category:",
            print_voc
        ]
        f.write("\n".join(record_lines))
    
    print(f"Evaluation results saved to: {record_path}")
    writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 使用设备类型
    parser.add_argument('--device', default='cuda', help='device')

    # 检测目标类别数
    parser.add_argument('--num-classes', type=int, default=20, help='number of classes')

    # 数据集的根目录(VOCdevkit根目录)
    parser.add_argument('--data-path', default='dataset', help='dataset root')

    # 训练好的权重文件
    parser.add_argument('--weights', default='./save_weights/ssd300-29.pth', 
                       type=str, help='training weights')

    # batch size
    parser.add_argument('--batch_size', default=64, type=int, 
                       metavar='N', help='batch size when validation.')

    args = parser.parse_args()
    main(args)