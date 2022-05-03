import os
import pickle
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='latin1')
    return data_dict

def batch2png(file_name):
    data_dict = unpickle(f"data/cifar-10-batches-py/{file_name}")
    labelL = data_dict["labels"]
    labelL_np = np.array(labelL)
    np.savetxt(f"log/{file_name}_gt.txt",labelL_np,fmt="%d")


batch2png("data_batch_5")