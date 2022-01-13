import random
import pandas as pd
from datetime import datetime
import numpy as np
#创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
# df = pd.DataFrame(columns=['time','step','train Loss','training accuracy'])#列名

#初始化train数据
def to_csv(CSV_PATH, t_step, t_loss, t_acc, lr0=0, lr1=0, lr2=0):
    time = "%s"%datetime.now()#获取当前时间
    step = "Step[%d]"%t_step
    train_loss = "%f"%t_loss
    train_acc = "%g"%t_acc
    #将数据保存在一维列表
    list = [time,step,train_loss,train_acc, lr0, lr1, lr2]
    #由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
    data = pd.DataFrame([list])
    data.to_csv(CSV_PATH,mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了

