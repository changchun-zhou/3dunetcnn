# -*- coding: UTF-8 -*-

#可视化
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
#读取csv中指定列的数据
import argparse

from fnmatch import fnmatch
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--path", type=str, default="")
# parser.add_argument("--excel", type=str, default="")

args = parser.parse_args()
x =[]
y1 = []
y2 = []
y3 = []
sparsity = []
lr0 = []
lr1 = []
lr2 = []
with open(args.path+'test_acc_loss.csv','r') as f:
    reader = csv.reader(f)
    for row in reader:
        x.append(int(str(row[1]).lstrip('Step[').rstrip(']')))
        y1.append(float(row[2])*100)
        y2.append(float(row[3])*100)

        # column = str(row[4])
        # t = column[:column.index(',')]
        # t = t[t.index('(')+1:]
        # y3.append(float(t))
        y3.append(float(row[4]))

        lr0.append(float(row[5])*500000)  # weight learning rate
        lr1.append(float(row[6])*50000)  # threshold learning rate
        lr2.append(float(row[7])*10000)  # threshold learning rate


for line in open(args.path+'get_logger.log'):
    if fnmatch(line, "*] Total sparsity:*"):
        sparsity.append(float(line[-6:].rstrip(" |\n").lstrip(" ")))  # sparsity


#绘图
highest_accuracy = max(y1[20:-1])
index_highest_accuracy = y1[20:-1].index(highest_accuracy) + 20
print(' highest_accuracy: ',highest_accuracy/100., 'step: ', index_highest_accuracy,
    ' threshold: ', y3[index_highest_accuracy])
plt.plot(x,y1,label="accuracy")
plt.plot(x,y2,label="loss")
plt.plot(x,y3,label="threshold")
plt.plot(x,sparsity,label="sparsity")
plt.plot(x,lr0,label="lr0")
plt.plot(x,lr1,label="lr1")
plt.plot(x,lr2,label="lr2")
plt.title("accuracy & loss & threshold") 
plt.xlabel('step')
plt.ylabel('Value')
plt.legend()   #显示标签
plt.savefig(args.path+'test_acc_loss'+'.png')
# plt.show()
