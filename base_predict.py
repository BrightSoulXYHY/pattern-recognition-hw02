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
from data.CIFAR10_BS import CIFAR10_BS
# 导入网络
from modules.NN_BS import *


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_size = 32

testset = CIFAR10_BS(root='./data', train=True, download=False, transform=transform_test,train_batchL=[4])
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


NN_DICT = {
    "FCx1":NN_FCx1,
    "CNN":NN_CNN,
    "CNN_AvgPool":NN_CNN_AvgPool,
    "LeNet":LeNet,
}

class NN_Config:
    def __init__(self, **data_dict):
        self.__dict__.update(data_dict)


def nn_builder(cfg):
    model = NN_DICT[cfg.nn_type]()
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
        if cfg.nn_type == "FCx1":
            images,labels = images.reshape(-1, img_size*img_size*3),labels
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        predicted_np = predicted.cpu().numpy()
        predicted_all = np.append(predicted_all,predicted_np)
    # predicted_all = np.array(predicted_all,dtype=np.int64)
    # predicted_all = predicted_all.reshape(-1,1)
    np.savetxt(f"log/{cfg.nn_type}_predict.txt",predicted_all,fmt="%d")
    print(f"[{time.time()-start_time:.2f}] All done,accuracy={accuracy:.4f}")


def main():
    args = sys.argv
    if len(args) == 1:
        yaml_path = "config/FCx1.yaml"
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