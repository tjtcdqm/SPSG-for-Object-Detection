from .ssd_vgg import SSD300_vgg 
import torch
import os

def get_model(checkpoint_path,model_name,n_classes,device):

    # valid_model = {'SSD300_vgg'}
    # assert model_name in valid_model, 'model name {} invalid please use valid model {}'.format(model_name,str(valid_model))
    # 如果存在断点文件，则直接load
    print(checkpoint_path)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        # model = eval(model_name + '({})'.format(n_classes))
        # model = SSD300(n_classes)
        checkpoint = torch.load(checkpoint_path)
        print('your checkoutpoint is ',list(checkpoint.keys()))
        # checkpoint = torch.load(os.path.join(model_dir,'checkpoint_ssd300_NWPU-VHR_best.pth.tar'))
        model = SSD300_vgg(n_classes)
        best_mAP = checkpoint.get('best_mAP50',-1)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        print('loading {} from {} checkpoint, best_mAP is {}'.format(model_name,checkpoint_path,best_mAP))
        return model
    # TODO 否则。。
    else:
        print('using plain model {} not pretrain version'.format(model_name))
        if model_name == 'ssd300_vgg':
            return SSD300_vgg(n_classes).to(device)
        
        assert False
