#!/usr/bin/python
import torchvision.transforms as transforms
import argparse
import json
import os
import os.path as osp
import pickle
from datetime import datetime
# from SPSG.victim.blackbox import Blackbox
from BlackBox import Blackbox
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch import optim
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader
from datasets import PascalVOCDataset
from config import DEFLAUT_SEED
from model import train_model
import config as cfg
from model import get_model
# from model import MultiBoxLoss
# ----------- set up  transform
transform = transforms.Compose([
    transforms.Resize((300, 300)),          # 调整图像尺寸为 300x300
    transforms.ToTensor(),                  # 将图像转换为 Tensor
    transforms.Normalize(                   # 归一化处理
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class TransferSetImagePaths(ImageFolder):
    """TransferSet Dataset, for when images are stored as *paths*"""

    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform


class TransferSetImages(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

        self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def samples_to_transferset(samples, budget=None, transform=None, target_transform=None):
    # Images are either stored as paths, or numpy arrays
    sample_x = samples[0][0]
    assert budget <= len(samples), 'Required {} samples > Found {} samples'.format(budget, len(samples))

    if isinstance(sample_x, str):
        return TransferSetImagePaths(samples[:budget], transform=transform, target_transform=target_transform)
    elif isinstance(sample_x, np.ndarray):
        return TransferSetImages(samples[:budget], transform=transform, target_transform=target_transform)
    else:
        raise ValueError('type(x_i) ({}) not recognized. Supported types = (str, np.ndarray)'.format(type(sample_x)))


def get_optimizer(parameters, optimizer_type, lr=0.01, momentum=0.5, **kwargs):
    assert optimizer_type in ['sgd', 'sgdm', 'adam', 'adagrad']
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr)
    elif optimizer_type == 'sgdm':
        optimizer = optim.SGD(parameters, lr, momentum=momentum)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(parameters)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(parameters)
    else:
        raise ValueError('Unrecognized optimizer type')
    return optimizer

def arg_parse():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('transferset_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle')
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('model_name', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test,containing test_image.json and test_object.json')
    parser.add_argument('--budgets', metavar='B', type=str,
                        help='Comma separated values of budgets. SPSGs will be trained for each budget.')
    # Optional arguments
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=40, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-step', type=int, default=60, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--pretrained_root', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm', choices=('sgd', 'sgdm', 'adam', 'adagrad'))
    parser.add_argument('--out_dir',type=str,help='Store well-train proxy model',default=None)
    # parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
    # Attacker's defense
    # parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    # parser.add_argument('--hard_label', type=bool, help='Used when only top-1 prediction is shown', default=False,)
    args = parser.parse_args()
    params = vars(args)
    return params

def main():

    params = arg_parse()
    torch.manual_seed(DEFLAUT_SEED)
    # 处理设备
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # model_dir = params['model_dir']
    # out_path = params['out_dir']

    # ----------- Set up transferset
    transferset_dir = params['transferset_dir']
    assert osp.exists(osp.join(transferset_dir,'transferset.pickle')) , 'transferset dir {} is not exists'.format(osp.join(transferset_dir,'transferset.pickle'))
    with open(osp.join(transferset_dir,'transferset.pickle'),'rb') as rf:
        transferset = pickle.load(rf)

    # ----------- Set up testset
    testset_root = params['testdataset']
    testset = PascalVOCDataset(testset_root,'val',True)
    print("length of testset is ",len(testset))

    # 获取种类数量
    with open(os.path.join(testset_root,'label_map.json'), 'r') as file:
        label_map = json.load(file)
    n_classes = len(label_map)

    # ----------- Set up proxy model
    model_name = params['model_name']
    pretrained_root = params['pretrained_root']
    proxy_model = get_model(pretrained_root,model_name,n_classes,device)

    # ----------- Set up victim model
    victim_dir = params['victim_model_dir']
    victim_name='ssd300_vgg'
    victim_checkpoint = osp.join(victim_dir,'checkpoint_{}_nwpu-vhr.pth.tar'.format(victim_name))
    assert osp.exists(victim_checkpoint), 'victim must load from checkponit'
    victim_model = get_model(victim_checkpoint,victim_name,n_classes,device=device)

    blackbox = Blackbox(victim_model,device)

    # ----------- Train
    budgets = [int(b) for b in params['budgets'].split(',')]


    for b in budgets:
        np.random.seed(DEFLAUT_SEED)
        torch.manual_seed(DEFLAUT_SEED)
        torch.cuda.manual_seed(DEFLAUT_SEED)

        # transferset_samples是通过transferset.pickle的来的
        # transferset.pickle是一个列表，每一个对象都是元组。每个元组由图片的原始路径(.jpg)和输出路径(.pickle文件)组成
        # transferset是一个imageFolder对象

        # 使用索引将获取原始图片和其对应的sg图
        transferset = samples_to_transferset(transferset, budget=b, transform=transform)
        print('=> Training at budget = {}'.format(len(transferset)))

        optimizer = get_optimizer(proxy_model.parameters(), params['optimizer_choice'], **params)
        print(params)

        checkpoint_suffix = '.{}'.format(b)

        out_dir = params['out_dir']
        epochs = params['epochs']
        train_model(proxy_model, blackbox, transferset, out_dir,label_map = label_map, testset=testset, checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer,**params)

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_dir, 'params_train.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
