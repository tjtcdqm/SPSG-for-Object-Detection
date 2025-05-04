from scipy.spatial.distance import cosine
import numpy as np

def compare_sg_maps(sg1, sg2, normalize=True, title="Difference Map"):
    """
    比较两个 SG 显著图：
    - 可视化差异热图
    - 打印余弦相似度、L1差、L2差
    """

    if sg1.dim() == 4:
        sg1 = sg1.squeeze(0)
    if sg2.dim() == 4:
        sg2 = sg2.squeeze(0)

    sg1 = sg1.detach().cpu().numpy()
    sg2 = sg2.detach().cpu().numpy()

    sg1_gray = np.mean(sg1, axis=0)
    sg2_gray = np.mean(sg2, axis=0)

    if normalize:
        sg1_gray = (sg1_gray - sg1_gray.min()) / (sg1_gray.max() - sg1_gray.min() + 1e-8)
        sg2_gray = (sg2_gray - sg2_gray.min()) / (sg2_gray.max() - sg2_gray.min() + 1e-8)

    # 计算差异图
    diff_map = np.abs(sg1_gray - sg2_gray)

    # 可视化差异热图
    plt.figure(figsize=(6, 6))
    plt.imshow(diff_map, cmap='bwr')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()

    # 扁平化用于距离计算
    sg1_flat = sg1_gray.flatten()
    sg2_flat = sg2_gray.flatten()

    cos_sim = 1 - cosine(sg1_flat, sg2_flat)
    l1_diff = np.mean(np.abs(sg1_flat - sg2_flat))
    l2_diff = np.sqrt(np.mean((sg1_flat - sg2_flat) ** 2))

    print(f"余弦相似度（越接近1越好）: {cos_sim:.4f}")
    print(f"L1 平均绝对差: {l1_diff:.4f}")
    print(f"L2 欧氏距离: {l2_diff:.4f}")
