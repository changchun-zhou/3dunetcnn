# *_*coding:utf-8 *_*
import os
import sys
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def cnt_zero(array):
    return np.sum(np.where(array,0,1))
def voxel_filter(point_cloud, leaf_size, random=False):

    point_cloud_xyz = np.array( point_cloud[0, :, 0:3] ) # batch 0
    filtered_points = []
    # 计算边界点
    x_min, y_min, z_min = np.amin(point_cloud_xyz, axis=0) #计算x y z 三个维度的最值
    x_max, y_max, z_max = np.amax(point_cloud_xyz, axis=0)
    print("x_min, y_min, z_min: {}, {}, {}".format(x_min, y_min, z_min))
    print("x_max, y_max, z_max: {}, {}, {}".format(x_max, y_max, z_max))
    # 计算 voxel grid维度
    Dx = ( (x_max - x_min)//leaf_size + 1 ).astype(np.int32)
    Dy = ( (y_max - y_min)//leaf_size + 1 ).astype(np.int32)
    Dz = ( (z_max - z_min)//leaf_size + 1 ).astype(np.int32)
    print("Dx x Dy x Dz is {} x {} x {}: {}".format(Dx, Dy, Dz, Dx*Dy*Dz*3))
 
    # 计算每个点的voxel索引
    h = list()  #h 为保存索引的列表
    voxel = np.zeros( (Dx, Dy, Dz, 3),dtype=np.int )
    voxel_diff = np.zeros( (Dx, Dy, Dz, 3),dtype=np.int )
    for i in range(len(point_cloud_xyz)):
        hx = ( (point_cloud_xyz[i][0] - x_min)//leaf_size ).astype(np.int32)
        hy = ( (point_cloud_xyz[i][1] - y_min)//leaf_size ).astype(np.int32)
        hz = ( (point_cloud_xyz[i][2] - z_min)//leaf_size ).astype(np.int32)
        h.append(hx + hy*Dx + hz*Dx*Dy)
        voxel[hx][hy][hz] = point_cloud[0][i][3:6]
    h = np.array(h)
    
    for x in range(1, Dx):
        voxel_diff[x] = voxel[x] - voxel[x-1]

    g_max = 0
    for x in range(Dx):
        for y in range(Dy):
            for z in range(Dz):
                if g_max < voxel[x][y][z][1]:
                    g_max = voxel[x][y][z][1]
    print("g_max: ", g_max)
    voxel_quant = np.round(voxel * 255)
    voxel_diff_quant = np.round(voxel_diff * 255)

    for x in range(Dx):
        for y in range(Dy):
            for z in range(Dz):
                # print(voxel_diff_quant[x][y][z])
                for c in range(3):
                    if voxel_diff_quant[x][y][z][c] <= 100:
                        voxel_diff_quant[x][y][z][c] = 0
    print("voxel valid: ", Dx*Dy*Dz*3 - cnt_zero(voxel))
    print("voxel_quant valid: ", Dx*Dy*Dz*3 - cnt_zero(voxel_quant))
    print("voxel_diff_quant valid: ", Dx*Dy*Dz*3 - cnt_zero(voxel_diff_quant))
    sys.exit()
    # length = np.round(1.0/leaf_size)
    # for i in range( length**3 ):
    #     voxel[i//(length)**2][i//length][i%length] = h[i]
    # return torch.from_numpy(voxel)

    # # 筛选点
    # h_indice = np.argsort(h) # 返回h里面的元素按从小到大排序的索引
    # h_sorted = h[h_indice]
    # begin = 0
    # for i in range(len(h_sorted)-1):   # 0~9999
    #     if h_sorted[i] == h_sorted[i + 1]: # delete repeated point
    #         continue
    #     else:
    #         point_idx = h_indice[begin: i + 1]
    #         filtered_points.append(np.mean(point_cloud[point_idx], axis=0)) # mean filter
    #         begin = i
    
    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

class PartNormalDataset(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints # 采样点数
        self.root = root # 文件根路径
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt') # 类别和文件夹名字对应的路径
        self.cat = {}
        self.normal_channel = normal_channel # 是否使用rgb信息


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()} #{'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', 'Car': '02958343', 'Chair': '03001627', 'Earphone': '03261776', 'Guitar': '03467517', 'Knife': '03624134', 'Lamp': '03636649', 'Laptop': '03642806', 'Motorbike': '03790512', 'Mug': '03797390', 'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987', 'Table': '04379243'}
        self.classes_original = dict(zip(self.cat, range(len(self.cat)))) #{'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}

        if not class_choice is  None:  # 选择一些类别进行训练  好像没有使用这个功能
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {} # 读取分好类的文件夹jason文件 并将他们的名字放入列表中
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)]) # '928c86eabc0be624c2bf2dcc31ba1713' 这是第一个值
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item]) # # 拿到对应一个文件夹的路径 例如第一个文件夹02691156
            fns = sorted(os.listdir(dir_point))  # 根据路径拿到文件夹下的每个txt文件 放入列表中
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids] # 判断文件夹中的txt文件是否在 训练txt中，如果是，那么fns中拿到的txt文件就是这个类别中所有txt文件中需要训练的文件，放入fns中
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                "第i次循环  fns中拿到的是第i个文件夹中符合训练的txt文件夹的名字"
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))  # 生成一个字典，将类别名字和训练的路径组合起来  作为一个大类中符合训练的数据
                #上面的代码执行完之后，就实现了将所有需要训练或验证的数据放入了一个字典中，字典的键是该数据所属的类别，例如飞机。值是他对应数据的全部路径
                #{Airplane:[路径1，路径2........]}
        #####################################################################################################################################################
        self.datapath = []
        for item in self.cat: # self.cat 是类别名称和文件夹对应的字典
            for fn in self.meta[item]:
                self.datapath.append((item, fn)) # 生成标签和点云路径的元组， 将self.met 中的字典转换成了一个元组

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        ## self.classes  将类别的名称和索引对应起来  例如 飞机 <----> 0
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        """
        shapenet 有16 个大类，然后每个大类有一些部件 ，例如飞机 'Airplane': [0, 1, 2, 3] 其中标签为0 1  2 3 的四个小类都属于飞机这个大类
        self.seg_classes 就是将大类和小类对应起来
        """
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        if index in self.cache: # 初始slef.cache为一个空字典，这个的作用是用来存放取到的数据，并按照(point_set, cls, seg)放好 同时避免重复采样
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index] # 根据索引 拿到训练数据的路径self.datepath是一个元组（类名，路径）
            cat = self.datapath[index][0] # 拿到类名
            cls = self.classes[cat] # 将类名转换为索引
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32) # size 20488,7 读入这个txt文件，共20488个点，每个点xyz rgb +小类别的标签
            if not self.normal_channel:  # 判断是否使用rgb信息
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32) # 拿到小类别的标签
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3]) # 做一个归一化

        choice = np.random.choice(len(seg), self.npoints, replace=True) # 对一个类别中的数据进行随机采样 返回索引，允许重复采样
        # resample
        point_set = point_set[choice, :] # 根据索引采样
        seg = seg[choice]
        return point_set, cls, seg # pointset是点云数据，cls十六个大类别，seg是一个数据中，不同点对应的小类别

    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    import torch

    root = r'/workspace/Pytorch/Dataset/3D-Unet/shapenetcore_partanno_segmentation_benchmark_v0_normal'
    "测试一下sharpnet数据集"
    data =  PartNormalDataset(root=root, normal_channel=True)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    test_cnt = 0
    for point in DataLoader:
        if test_cnt == 10:
            print('point0.shape:\n', point[0].shape, type(point[0]) ) # ([2, 2500, 3/6])
            print('point1.shape:\n', point[1].shape) # [2, 1])  大部件的标签
            print('point2.shape:\n', point[2].shape)  # torch.Size([2, 2500])  部件类别标签，每个点的标签
            images = voxel_filter(point[0], leaf_size = 0.01)
        test_cnt += 1





