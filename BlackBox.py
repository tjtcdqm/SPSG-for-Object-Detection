#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
import json
from matplotlib import cm
import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
torch.cuda.device_count()
torch.cuda.is_available()
torch.cuda.device_count()
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import lime
import random
from lime import lime_image
# from SPSG.utils.type_checks import TypeCheck
# import SPSG.utils.model as model_utils
# import SPSG.models.zoo as zoo
# from SPSG import datasets
from lime.wrappers.scikit_image import SegmentationAlgorithm
# from SPSG.utils.loss import MultiBoxLoss
from utils import find_jaccard_overlap


class Blackbox(object):
    def __init__(self, model, device=None, output_type='xyxy', topk=None, rounding=None,perturbation = None):
        self.device = torch.device('cuda') if device is None else device
        self.output_type = output_type
        self.topk = topk
        self.rounding = rounding

        self.__model = model.to(device)
        self.output_type = output_type
        self.__model.eval()

        self.__call_count = 0
        self.perturbation = 1e-4 if perturbation is None else perturbation

    def __call__(self, query_input):
        assert self.output_type in {'xyxy','xywh'}

        with torch.no_grad():
            query_input = query_input.to(self.device)
            predicted_locs, predicted_scores = self.__model(query_input)

            # 其中包含预测框位置，预测框标签，预测框置信度，以及预测框的logits
            det_boxes, det_labels, det_scores,det_logits = self.__model.detect_objects(predicted_locs, predicted_scores,
                                                                        min_score=0.3, max_overlap=0.45,
                                                                        top_k=200)
            # 计数器增加
            self.__call_count += query_input.shape[0]
            if self.output_type == 'xywh':
                # 该功能暂未支持
                assert False 
                batch_size = input.shape[0]
                # 转换坐标格式为xywh
                for i in range(batch_size):
                    det_boxes[i] = cxcy_to_xy(det_boxes[i])

        return det_boxes, det_labels, det_scores,det_logits


    def get_model(self):
        print('======================================================================================================')
        print('WARNING: USE get_model() *ONLY* FOR DEBUGGING')
        print('======================================================================================================')
        return self.__model

    def truncate_output(self, y_t_probs):
        if self.topk is not None:
            # Zero-out everything except the top-k predictions
            topk_vals, indices = torch.topk(y_t_probs, self.topk)
            newy = torch.zeros_like(y_t_probs)
            if self.rounding == 0:
                # argmax prediction
                newy = newy.scatter(1, indices, torch.ones_like(topk_vals))
            else:
                newy = newy.scatter(1, indices, topk_vals)
            y_t_probs = newy

        # Rounding of decimals
        if self.rounding is not None:
            y_t_probs = torch.Tensor(np.round(y_t_probs.numpy(), decimals=self.rounding))

        return y_t_probs

    def train(self):
        raise ValueError('Cannot run blackbox model in train mode')

    def eval(self):
        # Always in eval mode
        pass

    def get_call_count(self):
        return self.__call_count


    def getSuperGrandientLoss(self,gt_boxes,gt_labels,gt_scores,gt_logits,
                                det_boxes,det_labels,det_scores,det_logits):
        """
        计算超像素扰动前后目标检测输出之间的差异，用于SG估算。

        参数：
        - gt_logits: 原始图像下，每个检测框的 logits，形状 [num_boxes, num_classes]
        - det_logits: 扰动图像下，每个检测框的 logits
        - 其余参数用作匹配：gt_boxes, det_boxes, gt_labels, det_labels

        返回：
        - scalar loss，用作当前超像素梯度值（通过有限差分）
        """
        device = self.device

        if len(gt_boxes) == 0 and len(det_boxes) == 0:
            return torch.tensor(0.0).to(device)

        if len(gt_boxes) == 0 or len(det_boxes) == 0:
            # 如果一边为空，说明完全不匹配，返回最大扰动
            return torch.tensor(1.0).to(device)

        iou_matrix = find_jaccard_overlap(gt_boxes, det_boxes)# [N_gt, N_det]
        matched_gt_idx = iou_matrix.max(dim=0)[1]# 每个det框匹配的gt框索引
        matched_iou = iou_matrix.max(dim=0)[0]# 每个det框对应的最大IoU

        loss = 0.0
        count = 0
        matched_det_indices = []

        # 1. 匹配的框：正常计算交叉熵loss
        for i, iou in enumerate(matched_iou):
            if iou >= 0.8:
                matched_det_indices.append(i)
                gt_idx = matched_gt_idx[i]
                gt_logit = gt_logits[gt_idx]
                det_logit = det_logits[i]
                # Softmax + log_softmax + KL 散度
                P = F.softmax(gt_logit, dim=-1)
                Q_log = F.log_softmax(det_logit, dim=-1)
                kl_loss = F.kl_div(Q_log, P, reduction='batchmean') 
                # target_class = gt_logit.argmax().unsqueeze(0)
                # ce_loss = F.cross_entropy(det_logit.unsqueeze(0), target_class)
                loss += kl_loss
                count += 1

        # 2. 新增框（扰动后才出现，gt中无匹配）
        unmatched_det_indices = set(range(len(det_boxes))) - set(matched_det_indices)
        for idx in unmatched_det_indices:
            # det_logit = det_logits[idx]
            # 惩罚新增框的存在，可简单用其最大softmax得分作为“虚假置信度”
            # softmax_score = torch.softmax(det_logit, dim=0).max()
            loss += det_scores[idx] * 0.5 # 可以乘以权重
            count += 1

        # 3. 消失框（扰动前存在，扰动后不见了）
        matched_det_per_gt = iou_matrix.max(dim=1)[0]
        unmatched_gt_indices = (matched_det_per_gt < 0.8).nonzero(as_tuple=True)[0]
        for idx in unmatched_gt_indices:
            # gt_logit = gt_logits[idx]
            # 惩罚这个框的消失，可以用其原始分类置信度
            # softmax_score = torch.softmax(gt_logit, dim=0).max()
            loss += gt_scores[idx] * 0.5
            count += 1

        if count == 0:
            return torch.tensor(0.0).to(device)

        return loss / count



    def color2gray(self,color_img):
        size_h, size_w, channel = color_img.shape
        gray_img = np.zeros((size_h, size_w), dtype=np.uint8)
        for i in range(size_h):
            for j in range(size_w):
                gray_img[i, j] = round((color_img[i, j, 0] * 30 + color_img[i, j, 1] * 59 + \
                                        color_img[i, j, 2] * 11) / 100)
        return gray_img

    def get_SG_distribution(self,samples):
        '''
        生成图片的超像素梯度图
        params samples: 一批图片
        return : 对应的sg以及原始图片的输出
        '''
        SGs = []
        no_object_pic = [0] * len(samples) # 如果为1 表示被抛弃
        for i,origin_sample in enumerate(samples):
            sample = origin_sample.permute(1,2,0).cpu().numpy()
            # plt.imshow(sample)
            # plt.show()
            explainer = lime_image.LimeImageExplainer()
            xx = sample.astype(np.double)  # lime要求numpy array

            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=99)
            segments = segmentation_fn(xx)
            # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            # plt.show()

            sample_tensor = torch.tensor(copy.deepcopy(sample)).unsqueeze(dim=0).permute(0, 3, 1, 2)
            sample_tensor = sample_tensor.to(self.device)
            before_call_count =self.__call_count
            det_boxes, det_labels, det_scores,det_logits= self.__call__(sample_tensor)
            if len(det_boxes[0]) == 1 and det_labels[0][0] == 0:
                # 没有目标在图中
                no_object_pic[i] = 1
                SGs.append([])
                continue 
            # self.__call_count += sample_tensor.shape[0]

            SG = torch.zeros_like(sample_tensor)
            # print(np.unique(segments))
            # 对每一个超像素进行扰动
            for x in np.unique(segments):

                if np.any(segments==x):
                    for channel in range(3):
                        sample_permutation1 = copy.deepcopy(sample)
                        # sample_permutation2 = copy.deepcopy(sample)

                        with torch.no_grad():
                            sample_permutation1 = torch.tensor(sample_permutation1).unsqueeze(dim=0).permute(0, 3, 1, 2).to(self.device)

                            sample_permutation1[0, channel][(segments == x) ] += 1e-4
                            query_input1 = sample_permutation1.to(self.device)
                            gt_boxes_batch, gt_labels_batch, gt_scores_batch,gt_logits_batch  = self.__call__(query_input1)
                            # loss = self.getSuperGrandientLoss(gt_boxes[0],gt_labels[0],gt_scores[0],gt_logits[0],
                            #                             det_boxes_batch[0],det_labels_batch[0],det_scores_batch[0],det_logits_batch[0])
                            loss_original = self.getSuperGrandientLoss(
                                det_boxes[0], det_labels[0], det_scores[0],det_logits[0],
                                gt_boxes_batch[0], gt_labels_batch[0], gt_scores_batch[0],gt_logits_batch[0]
                            ).item()

                            loss_p = self.getSuperGrandientLoss(
                                gt_boxes_batch[0], gt_labels_batch[0], gt_scores_batch[0],gt_logits_batch[0],
                                gt_boxes_batch[0], gt_labels_batch[0], gt_scores_batch[0],gt_logits_batch[0]
                            ).item()

                            loss = loss_p - loss_original
                            a = loss / self.perturbation
                            SG[0, channel][torch.tensor((segments==x))]=a
            # print("call_count",self.__call_count)



            # reg1 = self.calculate(SG)
            # reg11 = reg1.view(3, sample_tensor.shape[2],  sample_tensor.shape[3]).permute(2, 1, 0).cpu().numpy()
            # reg2 = self.color2gray(reg11)
            #
            #
            #
            # fig = plt.figure()
            # ax = plt.axes(projection="3d")
            # X = np.arange(0, 224, 1)
            # Y = np.arange(0, 224, 1)
            # X, Y = np.meshgrid(X, Y)
            # surf = ax.plot_surface(X, Y, reg2, cmap=cm.coolwarm,
            #                        linewidth=0, antialiased=False)
            #
            # ax.set_zlim(0, 255)  # z轴的取值范围
            # # ax.zaxis.set_major_locator(LinearLocator(10))
            # # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            #
            # fig.colorbar(surf, shrink=0.5, aspect=5)
            # plt.show()
            # a = reg1.nonzero(as_tuple = False)
            # b = reg1[reg1.nonzero(as_tuple = True)]
            # print(a,b)
            # SGs.append([a,b])
            SGs.append(SG)
        return SGs,no_object_pic