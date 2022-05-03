import torch
import torchvision

import torchvision.transforms as transforms

import os
import time
import logging
from tqdm import tqdm
import yaml
import sys
import pprint
import numpy as np

# 导入数据集
from data.CIFAR10_Stacking import CIFAR10_Stacking
# 导入网络
from modules.NN_BS import *



start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_size = 32


testset = CIFAR10_Stacking(root='./data', train=False)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


NN_DICT = {
    "Stacking":NN_Stacking,
}

class NN_Config:
    def __init__(self, **data_dict):
        self.__dict__.update(data_dict)


def nn_builder(cfg):
    model = NN_DICT[cfg.nn_type](input_size=len(CIFAR10_Stacking.txt_fileL)*10)
    # 断点训练
    if os.path.exists(cfg.pth_test):
        model.load_state_dict(torch.load(cfg.pth_test, map_location=device))
        print(f"[{time.time()-start_time:.2f}] Load state_dict done")
    return model

def predict(cfg):
    model = nn_builder(cfg)
    model = model.to(device)
    predicted_all = np.array([],dtype=np.int64)

    correct = 0
    total = 0
    for images, labels in test_loader:
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        predicted_np = predicted.cpu().numpy()
        predicted_all = np.append(predicted_all,predicted_np)
    np.savetxt(f"log/{cfg.nn_type}_predict.txt",predicted_all,fmt="%d")
    print(f"[{time.time()-start_time:.2f}] All done,accuracy={accuracy:.4f}")


def main():
    args = sys.argv
    if len(args) == 1:
        yaml_path = "config/Stacking.yaml"
    elif len(args) == 2:
        yaml_path = args[1]
    else:
        print("arg num err")
        return
    
    if not os.path.exists(yaml_path):
        print(f"{yaml_path} not exists!")
        return
    with open(yaml_path,"r",encoding="utf-8") as f:
        data_dict = yaml.load(f,Loader=yaml.FullLoader)
        nn_cfg = NN_Config(**data_dict)

    print(f"Start Predict: {nn_cfg.nn_type} pth_path: {nn_cfg.pth_test}")

    predict(nn_cfg)


if __name__ == '__main__':
    main()