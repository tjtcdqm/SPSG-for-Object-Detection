import torch
import random
from datetime import datetime
from tqdm import tqdm
import os
import os.path as osp
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou
from collections import defaultdict as dd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pickle
import torch.optim as optim
from .SGP import *
from .loss import DetectionDistillLoss
def create_dir(dir_path):
    if not osp.exists(dir_path):
        print('Path {} does not exist. Creating it...'.format(dir_path))
        os.makedirs(dir_path)
def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max

import torch

def rand_in_range(low, high, size=None):
    """
    在指定区间 [low, high) 内生成服从均匀分布的 PyTorch 张量。
    
    参数:
        low (float): 区间下限（包含）
        high (float): 区间上限（不包含）
        size (tuple, optional): 输出张量的形状，默认为标量 (None)
        
    返回:
        torch.Tensor: 形状为 size 的张量，值在 [low, high) 范围内
    """
    if size is None:
        size = ()
    return (high - low) * torch.rand(size) + low
def recursive_clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone().detach()
    elif isinstance(x, list):
        return [recursive_clone(item) for item in x]
    elif isinstance(x, dict):
        return {k: recursive_clone(v) for k, v in x.items()}
    else:
        return x
def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h
def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.

    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.

    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor
# VOC-style mAP calculator without difficulty, supporting mAP@0.5 and mAP@[.5:.95]
def calculate_voc_mAP(det_boxes, det_labels, det_scores,
                      true_boxes, true_labels,
                      iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                      label_map=None, device='cpu'):
    assert label_map is not None
    rev_label_map = {v: k for k, v in label_map.items()}
    n_classes = len(label_map)

    true_images = [i for i, labels in enumerate(true_labels) for _ in range(labels.size(0))]
    det_images = [i for i, labels in enumerate(det_labels) for _ in range(labels.size(0))]
    true_images = torch.LongTensor(true_images).to(device)
    det_images = torch.LongTensor(det_images).to(device)

    true_boxes = torch.cat(true_boxes, dim=0)
    true_labels = torch.cat(true_labels, dim=0)

    det_boxes = torch.cat(det_boxes, dim=0)
    det_labels = torch.cat(det_labels, dim=0)
    det_scores = torch.cat(det_scores, dim=0)

    ap_results = {iou: torch.zeros((n_classes - 1), dtype=torch.float) for iou in iou_thresholds}

    for iou_thresh in iou_thresholds:
        for c in range(1, n_classes):  # skip background
            true_class_images = true_images[true_labels == c]
            true_class_boxes = true_boxes[true_labels == c].to(device)
            n_class_objects = true_class_boxes.size(0)
            true_class_boxes_detected = torch.zeros((n_class_objects), dtype=torch.uint8).to(device)

            det_class_images = det_images[det_labels == c]
            det_class_boxes = det_boxes[det_labels == c]
            det_class_scores = det_scores[det_labels == c]

            if det_class_boxes.size(0) == 0:
                continue

            det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)
            det_class_images = det_class_images[sort_ind]
            det_class_boxes = det_class_boxes[sort_ind]

            true_positives = torch.zeros(det_class_boxes.size(0), dtype=torch.float).to(device)
            false_positives = torch.zeros(det_class_boxes.size(0), dtype=torch.float).to(device)

            for d in range(det_class_boxes.size(0)):
                this_box = det_class_boxes[d].unsqueeze(0)
                this_image = det_class_images[d]
                object_boxes = true_class_boxes[true_class_images == this_image]

                if object_boxes.size(0) == 0:
                    false_positives[d] = 1
                    continue

                overlaps = box_iou(this_box, object_boxes).squeeze(0)
                max_overlap, ind = overlaps.max(dim=0)

                original_ind = torch.LongTensor(range(true_class_boxes.size(0))).to(device)[true_class_images == this_image][ind]

                if max_overlap.item() > iou_thresh:
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1
                    else:
                        false_positives[d] = 1
                else:
                    false_positives[d] = 1

            cumul_tp = torch.cumsum(true_positives, dim=0)
            cumul_fp = torch.cumsum(false_positives, dim=0)
            cumul_precision = cumul_tp / (cumul_tp + cumul_fp + 1e-10)
            cumul_recall = cumul_tp / (n_class_objects + 1e-10)

            precisions = torch.zeros(11, dtype=torch.float).to(device)
            for i, t in enumerate(torch.arange(0., 1.1, 0.1)):
                if (cumul_recall >= t).any():
                    precisions[i] = cumul_precision[cumul_recall >= t].max()
                else:
                    precisions[i] = 0.

            ap_results[iou_thresh][c - 1] = precisions.mean()

    mean_ap_50 = ap_results[0.5].mean().item()
    mean_ap_all = torch.stack([ap.mean() for ap in ap_results.values()]).mean().item()
    per_class_ap = {rev_label_map[c + 1]: ap_results[0.5][c].item() for c in range(n_classes - 1)}

    return {
        "mAP@0.5": mean_ap_50,
        "mAP@[.50:.95]": mean_ap_all,
        "per_class_AP@0.5": per_class_ap
    }

# VOC-style mAP calculator supporting mAP@0.5 and mAP@[.5:.95]
# def calculate_voc_mAP(det_boxes, det_labels, det_scores,
#                       true_boxes, true_labels, true_difficulties,
#                       iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
#                       label_map=None, device='cpu'):
#     assert label_map is not None
#     rev_label_map = {v: k for k, v in label_map.items()}
#     n_classes = len(label_map)

#     true_images = [i for i, labels in enumerate(true_labels) for _ in range(labels.size(0))]
#     det_images = [i for i, labels in enumerate(det_labels) for _ in range(labels.size(0))]
#     true_images = torch.LongTensor(true_images).to(device)
#     det_images = torch.LongTensor(det_images).to(device)

#     true_boxes = torch.cat(true_boxes, dim=0)
#     true_labels = torch.cat(true_labels, dim=0)
#     true_difficulties = torch.cat(true_difficulties, dim=0)

#     det_boxes = torch.cat(det_boxes, dim=0)
#     det_labels = torch.cat(det_labels, dim=0)
#     det_scores = torch.cat(det_scores, dim=0)

#     ap_results = {iou: torch.zeros((n_classes - 1), dtype=torch.float) for iou in iou_thresholds}

#     for iou_thresh in iou_thresholds:
#         for c in range(1, n_classes):
#             true_class_images = true_images[true_labels == c]
#             true_class_boxes = true_boxes[true_labels == c]
#             true_class_difficulties = true_difficulties[true_labels == c]
#             n_easy_class_objects = (1 - true_class_difficulties).sum().item()
#             true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(device)

#             det_class_images = det_images[det_labels == c]
#             det_class_boxes = det_boxes[det_labels == c]
#             det_class_scores = det_scores[det_labels == c]

#             if det_class_boxes.size(0) == 0:
#                 continue

#             det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)
#             det_class_images = det_class_images[sort_ind]
#             det_class_boxes = det_class_boxes[sort_ind]

#             true_positives = torch.zeros(det_class_boxes.size(0), dtype=torch.float).to(device)
#             false_positives = torch.zeros(det_class_boxes.size(0), dtype=torch.float).to(device)

#             for d in range(det_class_boxes.size(0)):
#                 this_box = det_class_boxes[d].unsqueeze(0)
#                 this_image = det_class_images[d]
#                 object_boxes = true_class_boxes[true_class_images == this_image]
#                 object_difficulties = true_class_difficulties[true_class_images == this_image]

#                 if object_boxes.size(0) == 0:
#                     false_positives[d] = 1
#                     continue

#                 overlaps = box_iou(this_box, object_boxes).squeeze(0)
#                 max_overlap, ind = overlaps.max(dim=0)

#                 original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]

#                 if max_overlap.item() > iou_thresh:
#                     if object_difficulties[ind] == 0:
#                         if true_class_boxes_detected[original_ind] == 0:
#                             true_positives[d] = 1
#                             true_class_boxes_detected[original_ind] = 1
#                         else:
#                             false_positives[d] = 1
#                 else:
#                     false_positives[d] = 1

#             cumul_tp = torch.cumsum(true_positives, dim=0)
#             cumul_fp = torch.cumsum(false_positives, dim=0)
#             cumul_precision = cumul_tp / (cumul_tp + cumul_fp + 1e-10)
#             cumul_recall = cumul_tp / (n_easy_class_objects + 1e-10)

#             precisions = torch.zeros(11, dtype=torch.float).to(device)
#             for i, t in enumerate(torch.arange(0., 1.1, 0.1)):
#                 if (cumul_recall >= t).any():
#                     precisions[i] = cumul_precision[cumul_recall >= t].max()
#                 else:
#                     precisions[i] = 0.

#             ap_results[iou_thresh][c - 1] = precisions.mean()

#     mean_ap_50 = ap_results[0.5].mean().item()
#     mean_ap_all = torch.stack([ap.mean() for ap in ap_results.values()]).mean().item()
#     per_class_ap = {rev_label_map[c + 1]: ap_results[0.5][c].item() for c in range(n_classes - 1)}

#     return {
#         "mAP@0.5": mean_ap_50,
#         "mAP@[.50:.95]": mean_ap_all,
#         "per_class_AP@0.5": per_class_ap
#     }
def test_step(model, blackbox, test_loader, criterion, device, label_map,
                        epoch=0., silent=False, writer=None):
    model.eval()
    blackbox.eval()
    test_loss = 0.
    total = 0

    # 用于 mAP 计算（proxy vs GT）
    all_proxy_boxes, all_proxy_labels, all_proxy_scores = [], [], []
    all_gt_boxes, all_gt_labels, all_gt_difficulties = [], [], []

    # 用于 mAP(proxy vs BB)
    all_bb_boxes, all_bb_labels, all_bb_scores = [], [], []

    with torch.no_grad():
        # for batch_idx, (images, gt_boxes, gt_labels, gt_difficulties) in enumerate(test_loader):
        # for batch_idx, (images, gt_boxes, gt_labels, gt_difficulties) in enumerate(test_loader):
        for batch_idx, (images, gt_boxes, gt_labels, gt_difficulties) in enumerate(tqdm(test_loader, desc=f"Testing Epoch {epoch}")):

            images = images.to(device)
            # GT
            # gt_boxes = [t['boxes'].to(device) for t in targets]
            # gt_labels = [t['labels'].to(device) for t in targets]
            # gt_difficulties = [t.get('difficult', torch.zeros_like(t['labels'])).to(device) for t in targets]

            # victim 模型输出（伪标签）
            bb_boxes, bb_labels, bb_scores, bb_logits = blackbox(images)
            all_bb_boxes.extend(bb_boxes)
            all_bb_labels.extend(bb_labels)
            all_bb_scores.extend(bb_scores)

            # 代理模型输出
            pred_locs, pred_scores = model(images)
            proxy_boxes, proxy_labels, proxy_scores,_ = model.detect_objects(pred_locs, pred_scores,min_score=0.1,max_overlap=0.5,top_k=200)

            all_proxy_boxes.extend(proxy_boxes)
            all_proxy_labels.extend(proxy_labels)
            all_proxy_scores.extend(proxy_scores)

            all_gt_boxes.extend(gt_boxes)
            all_gt_labels.extend(gt_labels)
            all_gt_difficulties.extend(gt_difficulties)

            # 计算蒸馏损失（基于 proxy vs victim 输出）
            total += len(images)
            # for i in range(len(images)):
            test_loss = criterion(
                proxy_boxes , proxy_labels , proxy_scores , pred_scores ,
                bb_boxes , bb_labels , bb_scores , bb_logits 
            )
                # test_loss += loss_i.item()

    test_loss /= total

    # ===== 评估 mAP@0.5 =====
    # proxy_eval = calculate_voc_mAP(
    #     all_proxy_boxes, all_proxy_labels, all_proxy_scores,
    #     all_gt_boxes, all_gt_labels, all_gt_difficulties,
    #     label_map=label_map, device=device
    # )
    proxy_eval = calculate_voc_mAP(
        all_proxy_boxes, all_proxy_labels, all_proxy_scores,
        all_gt_boxes, all_gt_labels, 
        label_map=label_map, device=device
    )

    # imitation_eval = calculate_voc_mAP(
    #     all_proxy_boxes, all_proxy_labels, all_proxy_scores,
    #     all_bb_boxes, all_bb_labels,
    #     [torch.zeros_like(l) for l in all_bb_labels],  # 黑盒没有 difficult
    #     label_map=label_map, device=device
    # )
    imitation_eval = calculate_voc_mAP(
        all_proxy_boxes, all_proxy_labels, all_proxy_scores,
        all_bb_boxes, all_bb_labels,
        # [torch.zeros_like(l) for l in all_bb_labels],  # 黑盒没有 difficult
        label_map=label_map, device=device
    )

    if not silent:
        print(f"[Test] Epoch: {epoch} | Loss: {test_loss.item():.4f} | "
              f"mAP@0.5 (GT): {proxy_eval['mAP@0.5']:.2f}% | "
              f"mAP@0.5 (vs Victim): {imitation_eval['mAP@0.5']:.2f}%")

    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('mAP@0.5/test', proxy_eval['mAP@0.5'], epoch)
        writer.add_scalar('mAP@0.5/imitation', imitation_eval['mAP@0.5'], epoch)

    return test_loss, proxy_eval['mAP@0.5'], imitation_eval['mAP@0.5'],proxy_eval['mAP@[.50:.95]'],imitation_eval['mAP@[.50:.95]']

def train_step(model,blackbox, train_loader, criterion, optimizer, epoch, device, log_interval=10, writer=None,log_path=None):
    model.train()
    train_loss = 0.
    train_losskl = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    # t_start = time.time()
    nans = 0
    # for batch_idx, (inputs, targets) in enumerate(train_loader):
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        sgs = []
        for idx in  range(inputs.shape[0]):
            with open(targets[idx], 'rb') as rf:
                real_targets = pickle.load(rf)
                # 一个sg是超像素划分图，其中具有相同超像素梯度的像素为一个超像素
                # 即图中每一个元素位置都不是像素而是超像素梯度
                sgs.append(real_targets)
        # 将一个包含多个张量的列表（或其他可迭代对象）logits，通过在新维度上堆叠这些张量来形成一个新的张量，并将其赋值给变量 logits
        sgs = torch.stack(sgs)
        inputs, sgs = inputs.to(device),sgs.to(device)
        # view()不会改变原始张量，而是返回新的张量
        # 相当于sgs.resize()
        sgs = sgs.view(sgs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3])
        if torch.isnan(sgs).any():
            sgs = sgs.reshape(sgs.shape[0],-1)
            logits = logits.reshape(sgs.shape[0],-1)
            mask = ~(torch.any(sgs.isnan(),dim=1))
            sgs = sgs[mask]
            sgs =  sgs.view(sgs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3])
            inputs =  inputs[mask]
            nans = nans+1
            # print(sgs.shape)


        optimizer.zero_grad()
        inputs.requires_grad_()
        # print(inputs.device)
        predicted_locs, predicted_scores = model(inputs)
        det_boxes, det_labels, det_scores,det_logits = model.detect_objects(predicted_locs, 
                                                            predicted_scores, min_score=0.01,
                                                            max_overlap=0.5, top_k=100)
        # proxy_outputs = model(inputs)
        # _, predict = outputs.max(1)
        with torch.no_grad():
            victim_boxes, victim_labels, victim_scores,victim_logits = blackbox(inputs)
            victim_boxes=recursive_clone(victim_boxes)
            victim_labels=recursive_clone(victim_labels)
            victim_scores=recursive_clone(victim_scores)
            victim_logits=recursive_clone(victim_logits)
            # victim_outputs = blackbox(inputs).clone().victimach()
            # _, box_predict = box_outputs.max(1)

        # equation 5
        # lossz = F.cross_entropy(outputs,box_predict)
        lossz = criterion(det_boxes,det_labels,det_scores,det_logits,
                            victim_boxes,victim_labels,victim_scores,victim_logits)
        # lossz这个变量对于inputs的每一个元素x_i求导的张量，所以torch.autograd.grad(lossz, inputs, create_graph=True)[0]的结果是和inputs一样的张量
        # 输入对loss的影响大小
        # 这里的newreg是 pixel gradient
        # newreg = torch.autograd.grad(lossz, inputs, create_graph=True)[0].view(inputs.shape)
        newreg = torch.autograd.grad(lossz, inputs, create_graph=True)[0].view(inputs.shape)

        # inputs_p = []
        # types = 10 # 这批图片中超像素分割种类最小的一个超像素数量
        # # sgs中特定的超像素梯度
        # # 超像素梯度净化模块，负责筛选极值梯度，并进行归一化
        # sgs = calculate(sgs).view(inputs.shape)
        # for j in range(sgs.shape[0]):
        #     a = sgs[j].unique()
        #     if types >= a.shape[0]:
        #         types = a.shape[0]
        #     # random.sample(population, k) 函数确保从 population 中抽取的每个元素都是唯一的，即每个元素只能被抽取一次。
        #     # 如果k>population的长度会出现异常
        #     Rsamples.append(random.sample(list(range(a.shape[0])), a.shape[0]))
        # for time in range(types):
        #     print(types)
        #     # inputs_p中每一个元素都是一个（N,C,H,W）的一批图片
        #     inputs_p.append(inputs.clone().detach())

        # SG 归一化后的超像素梯度图，形状: [B, C, H, W]
        sgs = calculate(sgs).view(inputs.shape)

        # 克隆输入用于扰动

        # losskk = []
        # targets_sgs = [-1.,-0.5,0.5,1.]
        # targets_sg = targets_sgs[random_sg]
        # for i in range(2):
            # 每张图片扰动一次：选取平均 SG 最接近 +1 的超像素
        input_p = inputs.clone().detach()
        for j in range(sgs.shape[0]):  # 遍历每张图片
            sg_avg = sgs[j].mean(dim=0)  # [H, W]，每个像素的平均 SG 值
            sps = sg_avg.unique()

            random_sg = torch.randint(low=0, high=2, size=(1,)).item()  
            targets_sg = None
            if random_sg == 0:
                max_sp ,_= sps.max(dim=0)
                # 在大于零的部分选出一个值
                targets_sg = rand_in_range(0,max_sp+1e-6)
            else :
                min_sp ,_ =sps.min(dim=0)
                targets_sg = rand_in_range(min_sp,0+1e-6)

            best_sp = None
            min_diff = float('inf')
            
            for sp in sps:
                diff = abs(sp- targets_sg)
                if diff < min_diff:
                    min_diff = diff
                    best_sp = sp

            # 在最接近 +1 的超像素区域添加扰动（3通道）
            if best_sp is not None:
                mask=sg_avg == best_sp
                for c in range(3):  # 每个通道都加扰动
                    input_p[j, c][mask] += 1e-4

        # 前向传播（扰动后）
        predicted_locs_p, predicted_scores_p = model(input_p)
        det_boxes_p, det_labels_p, det_scores_p, det_logits_p = model.detect_objects(
            predicted_locs_p, predicted_scores_p,
            min_score=0.01, max_overlap=0.50, top_k=200
        )

        # victim 模型前向
        with torch.no_grad():
            victim_boxes_p, victim_labels_p, victim_scores_p, victim_logits_p = blackbox(input_p)
            victim_boxes_p = recursive_clone(victim_boxes_p)
            victim_labels_p = recursive_clone(victim_labels_p)
            victim_scores_p = recursive_clone(victim_scores_p)
            victim_logits_p = recursive_clone(victim_logits_p)

        # 计算扰动后的 loss（代替 sum(losskk)）
        loss_p = criterion(det_boxes_p, det_labels_p, det_scores_p, det_logits_p,
                                victim_boxes_p, victim_labels_p, victim_scores_p, victim_logits_p)

            # losskk.append(loss_p)


        # losskk = []
        # for xiao in range(types):
        #     # 占用 1M
        #     input_p = inputs.clone().detach()
        # # for xiao,input_p in enumerate(inputs_p):
        #     #这个循环给每一张图片中的随机的一个超像素添加一个小扰动
        #     for j in range(sgs.shape[0]):
        #         a = sgs[j].unique()
        #         mask = sgs[j] == a[Rsamples[j][xiao]]
        #         input_p[j,mask] = input_p[j,mask] + 1e-4
        #         del mask

        #     # 占用100M 罪魁祸首
        #     predicted_locs, predicted_scores= model(input_p)
        #     det_boxes_p,det_labels_p,det_scores_p,det_logits_p =model.detect_objects(predicted_locs, predicted_scores,
        #                                                                             min_score=0.01, max_overlap=0.45,
        #                                                                             top_k=200)
        #     with torch.no_grad():
        #         victim_boxes_p,victim_labels_p,victime_scors_p,victim_logits_p = blackbox(input_p)
        #         victim_boxes=recursive_clone(victim_boxes)
        #         victim_labels=recursive_clone(victim_labels)
        #         victim_scores=recursive_clone(victim_scores)
        #         victim_logits=recursive_clone(victim_logits)
        #         # _, predict_p = box_outputs_p.max(1)
        #     if criterion==None:
        #         assert False
        #     else:
        #         # outputs是代理模型输出
        #         # box_outputs是受害者模型输出
        #         loss_p = criterion(det_boxes_p,det_labels_p,det_scores_p,det_logits_p,
        #                             victim_boxes_p,victim_labels_p,victime_scors_p,victim_logits_p)
                
        #         losskk.append(loss_p)
            # del inputs_p
            # del predicted_locs,predicted_scores



        for j in range(sgs.shape[0]):
            # 取出所有的超像素梯度
            a = sgs[j].unique()

            if a.shape[0] > 49 * 3:
                print("a", a.shape)
                # print("sgs[j]", torch.isnan(sgs[j]).any())
            for i in range(a.shape[0]):
                # newreg是梯度的平均值
                newreg[j, sgs[j] == a[i]] = newreg[j, sgs[j] == a[i]].view(-1).mean(dim=0)


        #这里的newreg是 simulated superpixel gradient
        newreg = calculate2(newreg,sgs)
        newreg = newreg.view(sgs.shape[0],-1)
        sgs = sgs.view(sgs.shape[0],-1)
        losskl = 1 - (F.cosine_similarity(newreg,
                                          sgs, dim=1).sum()) / (sgs.shape[0])


        with torch.no_grad():
            # lossz 和 losskl 都是标量
            delta = lossz.detach() - losskl.detach()
            
            # 控制范围 [min_val, max_val]
            min_val, max_val = 0.1, 2.0
            a = torch.sigmoid(delta) * (max_val - min_val) + min_val
        # with torch.no_grad():
        #     a = torch.exp(lossz.detach() - losskl.detach() - 3)
        #     a = torch.clamp(a, min=0.1, max=10.0)  # 稳定范围

        loss  = lossz +  loss_p + a *losskl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print("\nloss is {:.4f},lossz is {:.4f},loss_p is {:.4f},losskl is {:.4f} ,a is {:.4f}".format(
            loss.item(),lossz.item(),loss_p.item(),losskl.item(),a.item()
        ))
        # === Logging ===
        with open(log_path, 'a') as af:
            # Train 日志行：仅记录 loss
            train_cols = [batch_idx, epoch, 'train',blackbox.get_call_count(), f"{loss.item():.4f}", '', '', '']
            af.write('\t'.join(map(str, train_cols)) + '\n')

    return 

def train_model(model, blackbox,trainset, out_path,label_map, batch_size=64, 
                criterion_train=None, criterion_test=None, testset=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=100, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
                writer=None,eval_time=1, **kwargs):
    if device is None:
        device = torch.device('cuda')
    if not osp.exists(out_path):
        create_dir(out_path)
    run_id = str(datetime.now())

    # Data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,collate_fn=testset.collate_fn)
    else:
        test_loader = None

    # if weighted_loss:
    #     if not isinstance(trainset.samples[0][1], int):
    #         print('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]), int))

    #     class_to_count = dd(int)
    #     for _, y in trainset.samples:
    #         class_to_count[y] += 1
    #     class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
    #     print('=> counts per class: ', class_sample_count)
    #     weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
    #     weight = weight.to(device)
    #     print('=> using weights: ', weight)
    # else:
    #     weight = None

    weight = None
    criterion_train = DetectionDistillLoss()
    criterion_test = DetectionDistillLoss()
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    # best_train_acc, train_acc = -1., -1.
    # best_test_acc, test_acc, test_loss = -1., -1., -1.
    best_bb_mAP50 = -1.

    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            best_bb_mAP50 = checkpoint['best_bb_mAP50']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    log_path = osp.join(out_path, 'train{}.log.tsv'.format(checkpoint_suffix))
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            columns = ['run_id/batch_id', 'epoch', 'split', 'query','loss', 'gt_mAP50', 'bb_mAP50','bb_mAP5095','best_bb_mAP50']
            wf.write('\t'.join(columns) + '\n')

    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    for epoch in range(start_epoch, epochs + 1):
        train_step(model,blackbox, train_loader, criterion_train, optimizer, epoch, device,
                                           log_interval=log_interval,log_path=log_path)
        scheduler.step(epoch)

        if test_loader is not None and epoch % eval_time == 0:
            if epoch == epochs // 2:
                eval_time //= 2
            test_loss, gt_mAP50,bb_mAP50 ,gt_mAP5095,bb_mAP5095= test_step(model,blackbox, test_loader, criterion_test,label_map=label_map, device=device, epoch=epoch)

            # Checkpoint
            if bb_mAP50 > best_bb_mAP50:
                best_bb_mAP50 = bb_mAP50
                state = {
                    'epoch': epoch,
                    'arch': model.__class__,
                    'state_dict': model.state_dict(),
                    'best_bb_mAP50': best_bb_mAP50,
                    'bb_mAP5095':bb_mAP5095,
                    'optimizer': optimizer.state_dict(),
                    'created_on': str(datetime.now()),
                }
                torch.save(state, model_out_path)

            with open(log_path, 'a') as af:
                # Test 日志行：记录检测性能和模仿质量
                test_cols = [
                    run_id,
                    epoch,
                    'test',
                    blackbox.get_call_count(),
                    f"{test_loss.item():.4f}",     # procy vs BB
                    f"{gt_mAP50:.2f}",      # proxy vs GT
                    f"{bb_mAP50:.2f}",      # proxy vs BB
                    f"{gt_mAP5095:.2f}",  # proxy vs GT
                    f"{bb_mAP5095:.2f}",  # proxy vs BB
                    f"{best_bb_mAP50:.2f}"  # best proxy vs BB so far
                ]
                af.write('\t'.join(map(str, test_cols)) + '\n')

        # # Log
        # with open(log_path, 'a') as af:
        #     train_cols = [run_id, epoch, 'train', train_loss, train_acc, best_train_acc]
        #     af.write('\t'.join([str(c) for c in train_cols]) + '\n')
        #     test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc]
        #     af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    return model