import re
from tkinter.messagebox import NO
loss_re = re.compile(r"total_loss=(\d+)\.(\d+)")
accuracy_re = re.compile(r"accuracy=(\d+)\.(\d+)")


log_fileL = [
    "log/archive/NN_FCx1_20220502-172008.log"
    # "log/archive/LeNet_20220502-231836.log"
    # "log/archive/CNN_20220502-230001.log",
    # "log/archive/CNN_20220502-231128.log",
    # "log/archive/CNN_AvgPool_20220502-231618.log",
    # "log/archive/CNN_AvgPool_20220503-101431.log",
]
data_file = "log/plot/FCx1.txt"

textL = []
for log_file in log_fileL:
    with open(log_file,"r",encoding="utf-8") as fp:
        textL += fp.readlines()
total_loss = None
accuracy = None
lossL = []
accuracyL = []
for text in textL:

    loss_str = loss_re.search(text)
    if loss_str is not None:
        total_loss = (loss_str.group()).split("=")[-1]
        lossL.append(total_loss)

    accuracy_str = accuracy_re.search(text)
    if accuracy_str is not None:
        accuracy = (accuracy_str.group()).split("=")[-1]
        accuracyL.append(accuracy)
# print(lossL)
# print(accuracyL)
with open(data_file,"a",encoding="utf-8") as fp:
    for loss,accuracy in zip(lossL,accuracyL):
        fp.write(f"{loss} {accuracy}\n")