import argparse
import os.path as osp
import os
import pickle
import json
from datetime import datetime
import copy
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from model import get_model
from utils import *
from datasets import PascalVOCDataset
from BlackBox import Blackbox
from Adversary import RandomAdversary

from config import DEFLAUT_SEED
def arg_parse():
    parser = argparse.ArgumentParser(description='Construct transfer set')

    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('--modelname', metavar='TYPE', type=str, help='Model name, ssd300_backbone}', default=None)
    parser.add_argument('--victim_dataset',type=str,help='victim_model was trained on this dataset')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set', required=True)
    parser.add_argument('--budget', metavar='N', type=int, help='Size of transfer set to construct',
                        required=True)
    parser.add_argument('--queryset', metavar='TYPE', type=str, help='Adversary\'s dataset (P_A(X)) must reqiure json file', required=True)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=8)
    parser.add_argument('--root', metavar='DIR', type=str, help='Root directory for ImageFolder', default=None)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=20)

    args = parser.parse_args()
    params = vars(args)
    return params

def main():
    params = arg_parse()
    
    torch.manual_seed(DEFLAUT_SEED)

    # 确定设备
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print("using ",device)

    # 数据集取出
    dataset_root = params['queryset']
    print('dataset_root is '+dataset_root)
    assert osp.exists(osp.join(dataset_root,'TRAIN_images.json'))
    assert osp.exists(osp.join(dataset_root,'TRAIN_objects.json'))
    assert osp.exists(osp.join(dataset_root,'label_map.json'))

    dataset = PascalVOCDataset(dataset_root,'test',True)


    # 获取种类数量
    with open(os.path.join(dataset_root,'label_map.json'), 'r') as file:
        catogeries = json.load(file)
    n_classes = len(catogeries)


    # 取出模型
    victim_dir = params['victim_model_dir']
    victim_name = params['modelname']
    victim_dataset = params['victim_dataset']
    checkpoint_path = osp.join('.',victim_dir,'checkpoint_'+victim_name+'_'+victim_dataset+'.pth.tar')
    
    assert osp.exists(checkpoint_path ),'{} is not exist'.format(checkpoint_path)
    model = get_model(checkpoint_path,victim_name,n_classes,device=device)
    # 封装为黑盒模型
    blackbox = Blackbox(model,device)

    # 指定输出路径
    out_path = params['out_dir']
    create_dir(out_path)

    # 初始化扰动攻击类
    batch_size = params['batch_size']
    adversary = RandomAdversary(blackbox,dataset,out_path,batch_size)


    # 对抗攻击
    budget = params['budget'] # 攻击样本样本数量
    transfer_set = adversary.get_transferset(budget) #会随机生成budget张sg图片并保存，返回结果为（图片路径，最终保存路径）的列表
    print(blackbox.get_call_count())

    # 保存转移数据集
    transfetset_path = os.path.join(out_path,'transferset.pickle')
    with open(transfetset_path,'wb') as f:
        pickle.dump(transfer_set,f)
    # 保存参数
    params_path = os.path.join(out_path,"parmas_transfer.json")
    with open(params_path, 'w') as jf:
        json.dump(params, jf, indent=True)

if __name__ == '__main__':
    main()