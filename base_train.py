import torch
import torchvision

import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

import os
import time
import logging
from tqdm import tqdm
import yaml
import sys
import pprint


# 导入数据集
from data.CIFAR10_BS import CIFAR10_BS
# 导入网络
from modules.NN_BS import *


start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_batchL = [0,1,2,3]

# tb_writer = SummaryWriter("tb_log", flush_secs=10)


img_size = 32


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# 加载数据
trainset = CIFAR10_BS(root='./data', train=True, download=False, transform=transform_train,train_batchL=train_batchL)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = CIFAR10_BS(root='./data', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

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
    if cfg.continue_train and os.path.exists(cfg.pth_path):
        model.load_state_dict(torch.load(cfg.pth_path, map_location=device))
        print(f"[{time.time()-start_time:.2f}] Load state_dict done")
    return model

def train(cfg):
    time_str = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(
        level=logging.DEBUG,
        filename=f'log/{cfg.nn_type}_{time_str}.log',
        format='%(asctime)s - %(levelname)s: %(message)s'
    )
    num_epochs = cfg.num_epochs
    save_text = f'weights_log/{cfg.nn_type}_time={time_str}'+'_device={}_epoch={:02d}_acc={:.4f}.pth'


    model = nn_builder(cfg)
    # 设置损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    model = model.to(device)



    logging.info("start train")
    print(f"[{time.time()-start_time:.2f}] start train")
    # 训练模型
    for epoch in range(num_epochs):
        total_loss = 0
        val_loss = 0

        # 训练的进度条
        with tqdm(total=len(train_loader),mininterval=0.3) as pbar:
            for iteration, batch in enumerate(train_loader):
                model = model.train()
                images,labels = batch
                if cfg.nn_type == "FCx1":
                    images,labels = images.reshape(-1, img_size*img_size*3),labels
                images,labels = images.to(device),labels.to(device)

                # 前向传播获得模型的预测值
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 反向传播算出Loss对各参数的梯度
                optimizer.zero_grad()
                loss.backward()

                # 更新参数
                optimizer.step()

                total_loss += loss.item()
                val_loss = total_loss / (iteration + 1)
                desc_str = f"{'Train':8s} [{epoch + 1}/{num_epochs}] loss:{val_loss:.3f}"
                pbar.desc = f"{desc_str:40s}"
                # pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1)})
                pbar.update(1)
        logging.info(f"epoch={epoch} total_loss={val_loss:.6f}")
        # tb_writer.add_scalar("loss",val_loss,epoch)
        # print(f"[{time.time()-start_time:.2f}] epoch={epoch} total_loss={val_loss:.6f}")
        
        # 检验模型在测试集上的准确性
        with tqdm(total=len(test_loader),mininterval=0.3) as pbar:
            
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
                desc_str = f"{'Test':8s} [{epoch + 1}/{num_epochs}] accuracy:{accuracy:.3f}"
                pbar.desc = f"{desc_str:40s}"
                # pbar.set_postfix(**{'accuracy': accuracy})
                pbar.update(1)

            if accuracy >= 98:
                torch.save(model.state_dict(), save_text.format(device, epoch + 1 ,accuracy) )
                # print(f"[{time.time()-start_time:.2f}] epoch={epoch} accuracy={accuracy:.4f} train done")
                logging.warning(f"epoch={epoch} accuracy={accuracy:.4f} train done")
                break
        logging.info(f"epoch={epoch} accuracy={accuracy:.4f}")
        # tb_writer.add_scalar("accuracy",accuracy,epoch)
        # print(f"[{time.time()-start_time:.2f}] epoch={epoch} accuracy={accuracy:.4f}")
        # 每10个epoch保存一次
        if not (epoch + 1) % 10:
            torch.save(model.state_dict(), save_text.format(device, epoch + 1 ,accuracy) )
            logging.warning(f"data saver")


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
    # pprint.pprint(nn_cfg.__dict__)
    print(f"Start Train: {nn_cfg.nn_type} with learning_rate={nn_cfg.learning_rate} num_epochs={nn_cfg.num_epochs}")
    print(f"continue_train is {nn_cfg.continue_train}")
    if nn_cfg.continue_train:
        print(f"pth_path: {nn_cfg.pth_path}")



    train(nn_cfg)

if __name__ == '__main__':
    main()