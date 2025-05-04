import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.patches as patches
import os
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from PIL import Image
import numpy as np

def visualize_detections_batch(images, gt_boxes, gt_labels,
                               bb_boxes, bb_labels, bb_scores,
                               proxy_boxes, proxy_labels, proxy_scores,
                               label_map,
                               save_dir=None,
                               score_thresh=0.2,
                               epoch=None,
                               max_images=8):
    """
    可视化一批图像上的 GT / victim / proxy 检测框
    """
    os.makedirs(save_dir, exist_ok=True)
    label_map = label_map if isinstance(label_map, dict) else {i: str(i) for i in range(100)}

    images = images.cpu()
    for i in range(min(len(images), max_images)):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        img_np = images[i].permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # normalize to [0,1]
        ax.imshow(img_np)

        # --- Ground Truth ---
        for j in range(gt_boxes[i].size(0)):
            box = gt_boxes[i][j].cpu().numpy()
            label = gt_labels[i][j].item()
            draw_box(ax, box, label_map[label], color='green', linewidth=2, text='GT')

        # --- Victim Model Output ---
        for j in range(bb_boxes[i].size(0)):
            score = bb_scores[i][j].item()
            if score < score_thresh:
                continue
            box = bb_boxes[i][j].cpu().numpy()
            label = bb_labels[i][j].item()
            draw_box(ax, box, label_map[label], color='orange', text=f"BB:{label_map[label]}:{score:.2f}")

        # --- Proxy Model Output ---
        for j in range(proxy_boxes[i].size(0)):
            score = proxy_scores[i][j].item()
            if score < score_thresh:
                continue
            box = proxy_boxes[i][j].cpu().numpy()
            label = proxy_labels[i][j].item()
            draw_box(ax, box, label_map[label], color='blue', text=f"PR:{label_map[label]}:{score:.2f}")

        ax.set_title(f"Sample {i} (Epoch {epoch})")
        ax.axis('off')
        fig.tight_layout()

        if save_dir is not None:
            save_path = os.path.join(save_dir, f"det_epoch{epoch}_img{i}.png")
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

def draw_box(ax, box, label, color='red', linewidth=2, text=None):
    """在图像上画一个框 + 标签"""
    x_min, y_min, x_max, y_max = box
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             linewidth=linewidth, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    if text:
        ax.text(x_min, y_min - 2, text, fontsize=8,
                color='white', bbox=dict(facecolor=color, edgecolor=color, pad=1, alpha=0.8))

def visualize_sg_map(sg_tensor, title='Superpixel Gradient', normalize=True):
    """
    可视化超像素梯度图（SG）。
    输入:
        - sg_tensor: shape [1, 3, H, W] 或 [3, H, W] 的 torch tensor
    """
    if sg_tensor.dim() == 4:
        sg_tensor = sg_tensor.squeeze(0)
    if isinstance(sg_tensor, torch.Tensor):
        sg_tensor = sg_tensor.detach().cpu().numpy()
    
    # 转换为灰度图（可选）
    sg_gray = np.mean(sg_tensor, axis=0)  # shape: [H, W]

    if normalize:
        sg_gray = (sg_gray - sg_gray.min()) / (sg_gray.max() - sg_gray.min() + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.imshow(sg_gray, cmap='hot')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    # plt.show()
    plt.savefig('sg.jpg')
