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

def calculate(image, beta=0.5):
    N, C, H, W = image.shape
    image = image.view(N, C, -1)  # flatten to (N, C, H*W)
    
    # Replace zeros to avoid division by zero
    image[image == 0] = 1e-14

    # Get extreme values for each channel
    max_vals, _ = image.max(dim=2)  # shape: (N, C)
    min_vals, _ = image.min(dim=2)

    # Expand to match image dimensions
    maxpool = max_vals.unsqueeze(2).expand(N, C, H*W)  # not divided by 2 here
    minpool = min_vals.unsqueeze(2).expand(N, C, H*W)

    # Thresholding using beta
    pos_mask = (image >= beta * maxpool)
    neg_mask = (image <= beta * minpool)
    significant_mask = pos_mask | neg_mask

    # Keep only significant gradients
    purified = torch.where(significant_mask, image, torch.zeros_like(image))

    # Normalize positive and negative parts separately
    purified_pos = purified / (maxpool + 1e-14)  # normalize positive with max_vals
    purified_neg = purified / (abs(minpool) + 1e-14)  # normalize negative with abs(min_vals)
    normalized = torch.where(pos_mask, purified_pos, purified_neg)
    normalized = torch.where(neg_mask, purified_neg, normalized)

    return normalized.view(N,C,H,W)
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
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_sg_map(sg_tensor, title_prefix='Superpixel Gradient', normalize=True, out_path='sg.jpg'):
    """
    可视化 SG 的三个通道与灰度图，并保存为一张图。
    输入:
        - sg_tensor: shape [1, 3, H, W] or [3, H, W] 的 torch tensor
    """
    sg_tensor = calculate(sg_tensor.unsqueeze(0))
    print(sg_tensor.shape)
    if sg_tensor.dim() == 4:
        sg_tensor = sg_tensor.squeeze(0)

    print(sg_tensor.shape)
    sg_np = sg_tensor.detach().cpu().numpy()  # [3, H, W]

    if normalize:
        for c in range(3):
            max_val = np.abs(sg_np[c]).max()
            if max_val > 1e-8:
                sg_np[c] = sg_np[c] / max_val

    sg_gray = np.mean(sg_np, axis=0)

    # 绘图
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    cmaps = ['seismic'] * 4
    titles = [f'{title_prefix} (Gray Avg)', 'Channel R', 'Channel G', 'Channel B']
    images = [sg_gray, sg_np[0], sg_np[1], sg_np[2]]

    for ax, img, title, cmap in zip(axs, images, titles, cmaps):
        im = ax.imshow(img, cmap=cmap, vmin=-1, vmax=1)
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def visualize_sg_map_four_pic(sg_tensor, title_prefix='SG', normalize=True, save_path='sg.jpg'):
    """
    显示四张图：三个通道图 + 平均灰度图。

    参数:
        - sg_tensor: [1, 3, H, W] 或 [3, H, W] 的张量
        - normalize: 是否将值归一化到 [-1, 1]
        - save_path: 若指定路径则保存图像，否则 plt.show()
    """
    if sg_tensor.dim() == 4:
        sg_tensor = sg_tensor.squeeze(0)  # [3, H, W]

    sg_np = sg_tensor.detach().cpu().numpy()  # shape: [3, H, W]

    if normalize:
        # 对每个通道分别归一化到 [-1, 1]
        for c in range(3):
            c_min, c_max = sg_np[c].min(), sg_np[c].max()
            max_abs = max(abs(c_min), abs(c_max), 1e-8)
            sg_np[c] = sg_np[c] / max_abs

    # 平均图
    sg_mean = sg_np.mean(axis=0)  # [H, W]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    titles = [f'{title_prefix} - Mean', f'{title_prefix} - Channel 0', 
              f'{title_prefix} - Channel 1', f'{title_prefix} - Channel 2']
    images = [sg_mean, sg_np[0], sg_np[1], sg_np[2]]

    for ax, img, title in zip(axes, images, titles):
        im = ax.imshow(img, cmap='bwr', vmin=-1, vmax=1)
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# def visualize_sg_map(SG, title='Signed Superpixel Gradient (2D)', save_path='sg.jpg', cmap='seismic'):
#     """
#     可视化 Signed SG 超像素梯度图（带正负方向）。
#     输入：
#         - SG: shape [1, 3, H, W] 或 [3, H, W] 的 torch.Tensor
#         - save_path: 若不为 None 则保存图像
#         - cmap: 默认使用 'seismic'，红为正，蓝为负
#     """
#     if SG.dim() == 4:
#         SG = SG.squeeze(0)  # [3, H, W]
#     assert SG.dim() == 3 and SG.shape[0] == 3, "Expected SG of shape [3, H, W]"

#     # 转灰度图（平均3通道）
#     sg_gray = SG.detach().cpu().numpy().mean(axis=0)  # [H, W]

#     # 归一化到 [-1, 1]，保留正负方向
#     max_val = np.abs(sg_gray).max()
#     if max_val > 0:
#         sg_gray = sg_gray / max_val  # now in [-1, 1]

#     # 可视化
#     plt.figure(figsize=(6, 6))
#     im = plt.imshow(sg_gray, cmap=cmap, vmin=-1, vmax=1)
#     plt.colorbar(im, fraction=0.046, pad=0.04)
#     plt.title(title)
#     plt.axis('off')
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight')
#     else:
#         plt.show()
