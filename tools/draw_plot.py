import matplotlib.pyplot as plt
import numpy as np

import os
import glob


txt_fileL = glob.glob("../log/plot/*.txt")

# print(txt_fileL)
for txt_file in txt_fileL:
# txt_file = txt_fileL[0]
    basename_no_ext = os.path.splitext(os.path.basename(txt_file))[0]
    print(basename_no_ext)
    data = np.loadtxt(txt_file)


    fig, ax1 = plt.subplots()
    # plt.xticks(rotation=45)

    ax1.plot(data[:50,1], color="blue", alpha=0.5, label="accuracy")
    ax1.set_xlabel("iter_nums")
    ax1.set_ylabel("accuracy")


    ax2 = ax1.twinx()
    ax2.plot(data[:50,0], color="red", label="loss")
    ax2.set_ylabel("loss")

    fig.legend()
    plt.title(f"{basename_no_ext}")
    # plt.show()
    plt.savefig(f"../img/{basename_no_ext}.png",dpi=300)