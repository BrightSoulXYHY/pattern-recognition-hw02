import os
import numpy as np

import torch
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, List, Optional, Tuple

# 继承自父类torch.utils.data.Dataset
class CIFAR10_Stacking(Dataset):   
    txt_fileL = [
        "CNN_AvgPool_predict.txt",
        "CNN_predict.txt",
        "FCx1_predict.txt",
        "LeNet_predict.txt",
    ]
    
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train

        if self.train:
            data_path = "batch-5"
        else:
            data_path = "batch-test"
        
        base_path = os.path.join(self.raw_folder,data_path)
        self.data_np = np.vstack([
            np.loadtxt(f"{base_path}/{txt_file}",dtype=np.int)
            for txt_file in  self.txt_fileL
        ]).T

        self.label_np = np.loadtxt(f"{base_path}/data_gt.txt",dtype=np.int)

        self.transform = transform
        self.target_transform = target_transform




    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = self.data_np[index]
        label = self.label_np[index]
        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self) -> int:
        return len(self.label_np)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root,"cifar-10-stacking")


if __name__ == '__main__':
    import torchvision
    test_dataset = CIFAR10_Stacking(root='.',train=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    for data, label in test_loader:
        # print(type(labels),)
        print(data.shape)
        break