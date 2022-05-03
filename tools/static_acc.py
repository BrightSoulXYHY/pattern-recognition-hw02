import numpy as np

txtfile_pathL = [
    # "CNN_AvgPool_predict.txt",
    # "CNN_predict.txt",
    # "FCx1_predict.txt",
    # "LeNet_predict.txt",
    # "AlexNet_predict.txt",
    "Stacking_predict.txt",
]

data_path = "../data/cifar-10-stacking/batch-test"

gt_data = np.loadtxt(f"{data_path}/data_gt.txt",dtype=np.int)
# gt_data = np.loadtxt("../log/data/data_batch_5_gt.txt",dtype=np.int)
# fcx1_data = np.loadtxt("../log/data/FCx1_predict.txt",dtype=np.int)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def static_acc(pre_data,gt_data):
    predict_trueL = [0]*len(classes)
    class_totalL = [0]*len(classes)
    for pre,gt in zip(pre_data,gt_data):
        predict_trueL[gt] += 1 if pre == gt else 0
        class_totalL[gt] += 1
    for pre,gt,name in zip(predict_trueL,class_totalL,classes):
        print(f"{name} {100*pre/gt:.4f}")
        # print(f"{name} {pre} {gt} {100*pre/gt:.4f}")
    
    print(f"toatal {100*np.sum(predict_trueL)/np.sum(class_totalL):.4f}")
    
for i in txtfile_pathL:
    print(i)
    pre_data = np.loadtxt(f"{data_path}/{i}",dtype=np.int)
    static_acc(pre_data,gt_data)