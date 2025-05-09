import torch
# def calculate(image):
#     # The SGP module

#     N, C, H, W = image.shape[:4]
#     # 每一个通道本来都是一个二维图片，经过这里的view之后，每个通道都是一个一维的数组，每个元素都是原来二维数组铺平的结果
#     image = image.view(N, C, -1)
#     # 保证sgs没有零值，在之后的calculate2中不会出现除零异常
#     image[image == 0] = 1e-14

#     # 在第二个维度，即通道上，选出每个通道的最大值，返回一个张量，每个样本都是结果张量中的一个三元素数组，因为每个样本都有三个通道
#     # 会消灭我们选择的哪个维度，会有降维的效果
#     maxpool,_ = image.max(2)
#     # unsqueeze(2) 在第2维（即第三个维度，因为维度索引是从0开始的）插入一个维度，使 maxpool 的形状变为 (N, C, 1)
#     # expand(N,C,H*W) 将这个新张量扩展到形状 (N, C, H*W)，即沿着最后一个维度进行扩展。
#     maxpool = maxpool.unsqueeze(2).expand(N,C,H*W)/2.0
#     minpool,_ = image.min(2)
#     minpool = minpool.unsqueeze(2).expand(N,C,H*W)/2.0
#     #     torch.where(condition, x, y)
#     # condition：一个布尔类型的张量，定义了每个位置的条件。
#     # x：当 condition 为真时使用的张量。
#     # y：当 condition 为假时使用的张量。
#     # 选出超过最大值的一半的值，和小于最小值一半的值
#     image = torch.where((image>=maxpool)|(image<=minpool),image,0)
#     # 进行最大值归一化到[-1，-0.5][0.5,1]
#     image = torch.where((image>=maxpool), image / (2*maxpool),image)
#     image = torch.where((image<=minpool), image / (2*maxpool), image)
#     return image
def calculate(image, beta=0.5):
    N, C, H, W = image.shape
    image = image.view(N, C, -1)  # flatten to (N, C, H*W)
    
    # Replace zeros to avoid division by zero
    image[image == 0] = 1e-14
    # image = torch.where(image == 0, torch.full_like(image, 1e-14), image)


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

    # print(normalized.requires_grad)
    return normalized
def calculate2(image, sgs):
    N, C, H, W = image.shape
    device = image.device

    image2 = image.clone() * (sgs != 0)
    image2 = image2.view(N, C, -1)

    min_val, _ = image2.min(dim=2, keepdim=True)
    max_val, _ = image2.max(dim=2, keepdim=True)

    min_val[min_val.abs() < 1e-6] = -1e-6
    max_val[max_val.abs() < 1e-6] = 1e-6

    pos_mask = image2 > 0
    neg_mask = image2 < 0

    normed = torch.zeros_like(image2)
    normed[pos_mask] = image2[pos_mask] / max_val.expand_as(image2)[pos_mask]
    normed[neg_mask] = image2[neg_mask] / min_val.expand_as(image2)[neg_mask]
    normed[neg_mask] *= -1  # 保证负方向归一化为 [-1, 0]

    normed = normed.view(N, C, H, W)
    normed = normed * (sgs != 0)

    assert not torch.isnan(normed).any(), "NaN in calculate2 output"
    # print(normed.requires_grad)
    return normed

# def calculate2(image,sgs):

#     # 归一化处理image 分母为sgs中绝对值最大的梯度
#     image2 = image.clone().detach()*(sgs!=0)
#     N, C, H, W = image.shape[:4]
#     image = image.view(N, C, -1)
#     image2 = image2.view(N, C, -1)
#     maxpool,_ = image2.abs().max(2)
#     maxpool = maxpool.unsqueeze(2).expand(N,C,H*W)

#     image =  image / (maxpool)
#     return image