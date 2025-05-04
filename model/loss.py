from torch import nn
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou
def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))
def match_boxes_greedy(victim_boxes, proxy_boxes, iou_threshold=0.5, device='cpu'):
    """
    模仿 SSD 的标签分配策略，每个 victim box 至少匹配一个 proxy box，
    每个 proxy box 匹配与其 IoU 最大的 victim（如果 IoU 超过阈值）
    
    返回匹配列表: [(proxy_idx, victim_idx)]
    """
    matches = []
    if victim_boxes.size(0) == 0 or proxy_boxes.size(0) == 0:
        return matches

    ious = box_iou(victim_boxes, proxy_boxes)  # [M, N]

    # 对每个 proxy box，找到 IoU 最大的 victim box
    iou_for_each_proxy, victim_for_each_proxy = ious.max(dim=0)  # (N,) ← 每列最大

    # 对每个 victim box，找到 IoU 最大的 proxy box
    iou_for_each_victim, proxy_for_each_victim = ious.max(dim=1)  # (M,) ← 每行最大

    # 第一步：正常匹配（IoU >= 阈值的 proxy -> victim）
    for proxy_idx in range(proxy_boxes.size(0)):
        if iou_for_each_proxy[proxy_idx] >= iou_threshold:
            victim_idx = victim_for_each_proxy[proxy_idx].item()
            matches.append((proxy_idx, victim_idx))

    # 第二步：确保每个 victim 至少有一个 proxy 匹配（即使 IoU < 阈值）
    for victim_idx in range(victim_boxes.size(0)):
        proxy_idx = proxy_for_each_victim[victim_idx].item()
        # 如果这个 proxy 已经在匹配里就不重复
        if not any(p == proxy_idx for p, v in matches):
            matches.append((proxy_idx, victim_idx))

    return matches
import torch
import torch.nn.functional as F

def soft_focal_loss(pred_logits, soft_targets, gamma=2.0, weights=None, reduction='mean'):
    """
    Soft-label version of Focal Loss for classification distillation.

    Args:
        pred_logits: (N, C) predicted logits from student/proxy model
        soft_targets: (N, C) soft targets (e.g. from blackbox/victim model softmax)
        gamma: focal factor
        weights: optional sample-wise weights (N,)
        reduction: 'mean' | 'sum' | 'none'
    Returns:
        scalar loss
    """
    log_probs = F.log_softmax(pred_logits, dim=1)         # (N, C)
    probs = torch.exp(log_probs)                          # (N, C)
    focal_weight = (1.0 - probs) ** gamma                 # (N, C)

    loss = - soft_targets * focal_weight * log_probs      # (N, C)
    loss = loss.sum(dim=1)                                # (N,)

    if weights is not None:
        loss = loss * weights

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

# def match_boxes_greedy(victim_boxes, proxy_boxes, iou_threshold=0.5):
#     """
#     victim-centric greedy matching: 每个 victim box 匹配一个 proxy box
#     返回匹配索引对: List of tuples (proxy_idx, victim_idx)
#     """
#     matches = []
#     if victim_boxes.size(0) == 0 or proxy_boxes.size(0) == 0:
#         return matches

#     ious = box_iou(victim_boxes, proxy_boxes)  # [M_v, N_p]
#     used_proxy_indices = set()

#     for v_idx in range(victim_boxes.size(0)):
#         iou_row = ious[v_idx]  # [N_p]
#         iou_row[list(used_proxy_indices)] = -1  # 屏蔽已匹配 proxy

#         best_iou, best_p_idx = iou_row.max(dim=0)
#         if best_iou >= iou_threshold:
#             matches.append((best_p_idx.item(), v_idx))
#             used_proxy_indices.add(best_p_idx.item())

#     return matches
def ciou_loss(pred_boxes, target_boxes, reduction='mean', eps=1e-7):
    """
    Complete IoU loss for bounding box regression.
    Input: pred_boxes and target_boxes: [N, 4] in format [x_min, y_min, x_max, y_max]
    """
    # Ensure correct shape
    if pred_boxes.size(0) == 0 or target_boxes.size(0) == 0:
        return torch.tensor(0.0, requires_grad=True, device=pred_boxes.device)

    # Get width and height
    pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
    target_w = target_boxes[:, 2] - target_boxes[:, 0]
    target_h = target_boxes[:, 3] - target_boxes[:, 1]

    # IoU
    inter = (torch.min(pred_boxes[:, 2], target_boxes[:, 2]) - torch.max(pred_boxes[:, 0], target_boxes[:, 0])).clamp(0) * \
            (torch.min(pred_boxes[:, 3], target_boxes[:, 3]) - torch.max(pred_boxes[:, 1], target_boxes[:, 1])).clamp(0)
    union = pred_w * pred_h + target_w * target_h - inter + eps
    iou = inter / union

    # Enclosing box diagonal
    cw = torch.max(pred_boxes[:, 2], target_boxes[:, 2]) - torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    ch = torch.max(pred_boxes[:, 3], target_boxes[:, 3]) - torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    c2 = cw ** 2 + ch ** 2 + eps

    # Center distance
    pred_ctr_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_ctr_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_ctr_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_ctr_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    rho2 = (pred_ctr_x - target_ctr_x) ** 2 + (pred_ctr_y - target_ctr_y) ** 2

    # Aspect ratio term
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(target_w / (target_h + eps)) - torch.atan(pred_w / (pred_h + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha * v
    loss = 1 - ciou

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
class DetectionDistillLoss(nn.Module):
    def __init__(self, location_loss_fn=ciou_loss, classification_loss_fn=soft_focal_loss,
                 iou_threshold=0.5, loc_weight=1.0, cls_weight=1.0):
        """
        location_loss_fn: 回归损失函数（例如 CIoU、GIoU）
        classification_loss_fn: 分类损失函数（例如 soft cross entropy）
        iou_threshold: IoU 匹配阈值
        loc_weight: 位置损失权重
        cls_weight: 分类损失权重
        """
        super().__init__()
        self.location_loss_fn = location_loss_fn
        self.classification_loss_fn = classification_loss_fn
        self.iou_threshold = iou_threshold
        self.loc_weight = loc_weight
        self.cls_weight = cls_weight

    def forward(self,
                proxy_boxes_batch, proxy_labels_batch, proxy_scores_batch, proxy_logits_batch,
                victim_boxes_batch, victim_labels_batch, victim_scores_batch, victim_logits_batch):

        batch_size = len(proxy_boxes_batch)
        total_loss = 0.0
        total_count = 0

        for i in range(batch_size):
            proxy_boxes = proxy_boxes_batch[i]  # [N_p, 4]
            proxy_logits = proxy_logits_batch[i]  # [N_p, C]
            victim_boxes = victim_boxes_batch[i]  # [N_v, 4]
            victim_logits = victim_logits_batch[i]  # [N_v, C]

            matches = match_boxes_greedy(victim_boxes, proxy_boxes, self.iou_threshold)

            for p_idx, v_idx in matches:
                # 分类 soft loss
                soft_target = F.softmax(victim_logits[v_idx], dim=0).unsqueeze(0)
                pred_logit = proxy_logits[p_idx].unsqueeze(0)
                cls_loss = self.classification_loss_fn(pred_logit, soft_target)

                # 位置回归 loss
                loc_loss = self.location_loss_fn(proxy_boxes[p_idx].unsqueeze(0),
                                                 victim_boxes[v_idx].unsqueeze(0))

                total_loss += self.loc_weight * loc_loss + self.cls_weight * cls_loss
                total_count += 1

        if total_count == 0:
            assert False ,'no matching error'
            # return torch.tensor(0.0, requires_grad=True).to(proxy_logits_batch[0].device)

        return total_loss / total_count

# def detection_distill_criterion(
#     proxy_boxes_batch, proxy_labels_batch, proxy_scores_batch, proxy_logits_batch,
#     victim_boxes_batch, victim_labels_batch, victim_scores_batch, victim_logits_batch,
#     iou_threshold=0.5,
#     location_loss = CIOU_loss,
#     classification_loss = soft_cross_entropy
# ):
#     """
#     支持 batch 输入的 detection distillation 损失
#     每张图像独立匹配，然后平均 batch loss

#     所有输入应为长度为 batch_size 的 list，每个元素为 [N_i, ...] 的 tensor
#     """

#     batch_size = len(proxy_boxes_batch)
#     total_loss = 0.0
#     total_count = 0

#     for i in range(batch_size):
#         proxy_boxes = proxy_boxes_batch[i]
#         proxy_logits = proxy_logits_batch[i]
#         victim_boxes = victim_boxes_batch[i]
#         victim_logits = victim_logits_batch[i]

#         if proxy_boxes.size(0) == 0 or victim_boxes.size(0) == 0:
#             continue

#         ious = box_iou(victim_boxes, proxy_boxes)  # [M_i, N_i]
#         matched_iou, matched_victim_idx = ious.max(dim=0)  # 每个 proxy box 匹配一个 victim box

#         for j, iou in enumerate(matched_iou):
#             if iou >= iou_threshold:
#                 v_idx = matched_victim_idx[j]

#                 # 分类损失（soft）
#                 soft_target = F.softmax(victim_logits[v_idx], dim=0).unsqueeze(0)
#                 pred_logit = proxy_logits[j].unsqueeze(0)
#                 ce_loss = F.kl_div(F.log_softmax(pred_logit, dim=1), soft_target, reduction='batchmean')

#                 # 位置损失（CIoU）
#                 box_loss = ciou_loss(proxy_boxes[j].unsqueeze(0), victim_boxes[v_idx].unsqueeze(0))

#                 total_loss += ce_loss + box_loss
#                 total_count += 1

#     if total_count == 0:
#         return torch.tensor(0.0).to(proxy_logits_batch[0].device)
    
#     return total_loss / total_count


# class MultiBoxLoss(nn.Module):
#     """
#     The MultiBox loss, a loss function for object detection.

#     This is a combination of:
#     (1) a localization loss for the predicted locations of the boxes, and
#     (2) a confidence loss for the predicted class scores.
#     """

#     def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
#         super(MultiBoxLoss, self).__init__()
#         self.priors_cxcy = priors_cxcy
#         self.priors_xy = cxcy_to_xy(priors_cxcy)
#         self.threshold = threshold
#         self.neg_pos_ratio = neg_pos_ratio
#         self.alpha = alpha

#         self.smooth_l1 = nn.L1Loss()  # *smooth* L1 loss in the paper; see Remarks section in the tutorial
#         self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

#     def forward(self, predicted_locs, predicted_scores, boxes, labels):
#         """
#         Forward propagation.

#         :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
#         :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
#         :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
#         :param labels: true object labels, a list of N tensors
#         :return: multibox loss, a scalar
#         """
#         batch_size = predicted_locs.size(0)
#         n_priors = self.priors_cxcy.size(0)
#         n_classes = predicted_scores.size(2)

#         assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

#         true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
#         true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

#         # For each image
#         for i in range(batch_size):
#             n_objects = boxes[i].size(0)

#             overlap = find_jaccard_overlap(boxes[i],
#                                            self.priors_xy)  # (n_objects, 8732)

#             # For each prior, find the object that has the maximum overlap
#             overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

#             # We don't want a situation where an object is not represented in our positive (non-background) priors -
#             # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
#             # 2. All priors with the object may be assigned as background based on the threshold (0.5).

#             # To remedy this -
#             # First, find the prior that has the maximum overlap for each object.
#             _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

#             # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
#             object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

#             # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
#             overlap_for_each_prior[prior_for_each_object] = 1.

#             # Labels for each prior
#             label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
#             # Set priors whose overlaps with objects are less than the threshold to be background (no object)
#             label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

#             # Store
#             true_classes[i] = label_for_each_prior

#             # Encode center-size object coordinates into the form we regressed predicted boxes to
#             true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

#         # Identify priors that are positive (object/non-background)
#         positive_priors = true_classes != 0  # (N, 8732)

#         # LOCALIZATION LOSS

#         # Localization loss is computed only over positive (non-background) priors
#         loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

#         # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
#         # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

#         # CONFIDENCE LOSS

#         # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
#         # That is, FOR EACH IMAGE,
#         # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
#         # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

#         # Number of positive and hard-negative priors per image
#         n_positives = positive_priors.sum(dim=1)  # (N)
#         n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

#         # First, find the loss for all priors
#         conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
#         conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

#         # We already know which priors are positive
#         conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

#         # Next, find which priors are hard-negative
#         # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
#         conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
#         conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
#         conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
#         hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
#         hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
#         conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

#         # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
#         conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

#         # TOTAL LOSS

#         return conf_loss + self.alpha * loc_loss