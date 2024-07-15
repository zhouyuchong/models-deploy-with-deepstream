import torch
import numpy as np
from tqdm import tqdm
from tool import *

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class CocoDetectionEvaluator():
    def __init__(self, names, device):
        self.device = device
        self.classes = []
        with open(names, 'r') as f:
            for line in f.readlines():
                self.classes.append(line.strip())
    
    def coco_evaluate(self, gts, preds):
        # Create Ground Truth
        coco_gt = COCO()
        coco_gt.dataset = {}
        coco_gt.dataset["images"] = []
        coco_gt.dataset["annotations"] = []
        k = 0
        for i, gt in enumerate(gts):
            for j in range(gt.shape[0]):
                k += 1
                coco_gt.dataset["images"].append({"id": i})
                coco_gt.dataset["annotations"].append({"image_id": i, "category_id": gt[j, 0],
                                                    "bbox": np.hstack([gt[j, 1:3], gt[j, 3:5] - gt[j, 1:3]]),
                                                    "area": np.prod(gt[j, 3:5] - gt[j, 1:3]),
                                                    "id": k, "iscrowd": 0})
                
        coco_gt.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(self.classes)]
        coco_gt.createIndex()

        # Create preadict 
        coco_pred = COCO()
        coco_pred.dataset = {}
        coco_pred.dataset["images"] = []
        coco_pred.dataset["annotations"] = []
        k = 0
        for i, pred in enumerate(preds):
            for j in range(pred.shape[0]):
                k += 1
                coco_pred.dataset["images"].append({"id": i})
                coco_pred.dataset["annotations"].append({"image_id": i, "category_id": np.int32(pred[j, 0]),
                                                        "score": pred[j, 1], "bbox": np.hstack([pred[j, 2:4], pred[j, 4:6] - pred[j, 2:4]]),
                                                        "area": np.prod(pred[j, 4:6] - pred[j, 2:4]),
                                                        "id": k})
                
        coco_pred.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(self.classes)]
        coco_pred.createIndex()

        coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAP05 = coco_eval.stats[1]
        return mAP05
    
    def postprocess(self, preds, net_shape, img_shape):
        x_scale = img_shape[1] / net_shape[1]
        y_scale = img_shape[0] / net_shape[0]
        preds = preds * torch.tensor([x_scale, y_scale, x_scale, y_scale, 1, 1]).to(preds.device)
        return preds

    def compute_map(self, val_dataloader, model, net_shape):
        gts, pts = [], []
        pbar = tqdm(val_dataloader)
        for i, (imgs, targets, ori_h, ori_w) in enumerate(pbar):
            # 数据预处理
            imgs = imgs.numpy()
            image = imgs / 255.0
            image = np.array(image, dtype=np.float32, order="C")
            # don't need since it is a tensorrt, but we keep it for the future
            # with torch.no_grad():
            # 模型预测
            preds = model.infer(image)[0]
            # sorted_indices = np.argsort(preds[:, :, 4])[0, -100:]  # 获取排序后的索引
            # preds = preds[:, sorted_indices, :]

            # 检测结果
            N, C, H, W = image.shape
            x_scale = ori_w / net_shape[1]
            y_scale = ori_h / net_shape[0]
            for p in preds:
                pbboxes = []
                for b in p:
                    score = b[4]
                    # use conf to filte
                    if score < 0.01:
                        continue
                    category = b[5]
                    x1, y1, x2, y2 = b[:4] * [x_scale, y_scale, x_scale, y_scale]
                    # print("pred", x1, y1, x2, y2, category)
                    pbboxes.append([category, score, x1, y1, x2, y2])
                pts.append(np.array(pbboxes))
            # 标注结果
            # for n in range(N):
            tbboxes = []
            for n in range(N):
                tbboxes = []
                for t in targets:
                    if t[0] == n:
                        t = t.cpu().numpy()
                        category = t[5]
                        x1, y1, x2, y2 = t[1:5]
                        # print("gt", x1, y1, x2, y2, t[5])
                        tbboxes.append([category, x1, y1, x2, y2])
                gts.append(np.array(tbboxes))
            # print("===")
            # for t in targets:
            #     # if t[0] == n:
            #     t = t.cpu().numpy()
            #     # print(t)
            #     category = t[5]
            #     x1, y1, x2, y2 = t[1:5]
            #     # print("gt", x1, y1, x2, y2)
                
            #     # bcx, bcy, bw, bh = t[2:] * [W, H, W, H]
            #     # x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
            #     # x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh
            #     tbboxes.append([category, x1, y1, x2, y2])
            # gts.append(np.array(tbboxes))
                
        mAP05 = self.coco_evaluate(gts, pts)

        return mAP05
