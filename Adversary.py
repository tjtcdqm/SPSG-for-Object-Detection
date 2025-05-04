import numpy as np
import torch
import os.path as osp
import os
from tqdm import tqdm
import pickle
import json
from config import DEFLAUT_SEED
class RandomAdversary(object):
    def __init__(self, blackbox, queryset ,transfer_out_path ,batch_size=8):
        self.blackbox = blackbox
        self.queryset = queryset

        self.n_queryset = len(self.queryset)
        self.batch_size = batch_size
        self.idx_set = set()

        self.transferset = []  # List of tuples [(img_path, output_probs)]
        self.out_dir = transfer_out_path
        self._restart()

    def _restart(self):
        np.random.seed(DEFLAUT_SEED)
        torch.manual_seed(DEFLAUT_SEED)
        torch.cuda.manual_seed(DEFLAUT_SEED)

        self.idx_set = set(range(len(self.queryset)))
        self.transferset = []

    def superGradient(box):
        pass
    
    def get_transferset(self,budget):
        """
        构建转移数据集，返回[budget*(原始路径，最终路径)]数据作为转移数据集

        1. 获取每一个图片的超像素梯度图
        2. 将其保存到指定路径中
        3. 保存图片的原始路径和最终路径
        4. 返回
        :param budget: 转移数据集中图片的大小
        :return: 一个列表，其中每个元素都是原始图片路径和超像素梯度图片路径构成的元组
        """
        start_B = 0
        end_B = budget
        with tqdm(total=budget) as pbar:
            # for t, B in enumerate(range(start_B, end_B, self.batch_size)):
            while start_B < end_B:
                idxs = np.random.choice(list(self.idx_set), replace=False,
                                        size=min(self.batch_size, budget - len(self.transferset),len(self.idx_set)))
                self.idx_set = self.idx_set - set(idxs)

                if len(self.idx_set) == 0:
                    print('=> Query set exhausted. Now repeating input examples.')
                    self.idx_set = set(range(len(self.queryset)))

                # 取出图片
                # imageFolder[i]对象的返回值是(image,label)，所以再次取零就是取出图片
                x_t = torch.stack([self.queryset[i][0] for i in idxs]).to(self.blackbox.device)
                sg_t,no_object_pic = self.blackbox.get_SG_distribution(x_t)
                # 如果有图片被废弃，则延迟结束条件
                end_B += sum(no_object_pic)

                # 取出图片路径
                img_t = [self.queryset.get_image_path(i) for i in idxs]# Image paths

                for i in range(x_t.size(0)):
                    if no_object_pic[i] == 1:
                        # 该图片被废弃
                        continue
                    img_path = img_t[i]
                    sg = (sg_t[i].cpu().squeeze()) # 超像素梯度图

                    # img_t_i = img_t_i.replace("/root/SPSG_attack/data/256_ObjectCategories/","")
                    # img_t_i = img_t_i.replace(".JPEG", "")
                    # img_t_i1 = img_t_i.split("/",1)[0]
                    # img_t_i2 = img_t_i.split("/", 1)[1]
                    # 构造最终存储路径
                    filename_without_ext = os.path.splitext(os.path.basename(img_path))[0]
                    final_path  =os.path.join(self.out_dir,filename_without_ext+".pickle") 

                    # 存储sg图到路径中
                    if not os.path.exists(self.out_dir):
                        os.makedirs(self.out_dir)
                    with open(final_path, "wb") as wf:
                        pickle.dump(sg,wf)
                    self.transferset.append((img_path,final_path))
                
                print('=> transfer set ({} samples) written to: {}'.format(self.batch_size, self.out_dir))
                start_B += self.batch_size
                pbar.update(x_t.size(0)-sum(no_object_pic))
        return self.transferset

    def _load_checkpoint(self):
        if osp.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            self.transferset = checkpoint['transferset']
            self.idx_set = set(checkpoint['idx_set'])
            if 'rng_state' in checkpoint:
                np.random.set_state(checkpoint['rng_state'])
            return checkpoint['processed']
        return 0