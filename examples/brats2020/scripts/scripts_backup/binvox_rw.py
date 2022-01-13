# coding=utf-8
import numpy as np
class Voxels(object):
    def __init__(self, data, dims, translate, scale, axis_order):
        self.data = data
        self.dims = dims
        self.translate = translate
        self.scale = scale
        # axis不是xyz坐标系的时候触发异常
        assert (axis_order in ('xzy', 'xyz'))
        self.axis_order = axis_order
def read_as_3d_array(fp, fix_coords=True):
    dims, translate, scale = read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    # if just using reshape() on the raw data:
    # indexing the array as array[i,j,k], the indices map into the
    # coords as:
    # i -> x
    # j -> z
    # k -> y
    # if fix_coords is true, then data is rearranged so that
    # mapping is
    # i -> x
    # j -> y
    # k -> z
    values, counts = raw_data[::2], raw_data[1::2]
    # data = np.repeat(values, counts).astype(np.bool)
    data = np.repeat(values, counts)
    data = data.reshape(dims)
    if fix_coords:
        # xzy to xyz TODO the right thing
        data = np.transpose(data, (0, 2, 1))
        axis_order = 'xyz'
    else:
        axis_order = 'xzy'
    return Voxels(data, dims, translate, scale, axis_order)
def read_header(fp):
    # 读取binvox头文件
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    line = fp.readline()
    return dims, translate, scale
if __name__ == '__main__':
    path = '/workspace/Pytorch/Dataset/3D-Unet/ShapeNetVox32/02691156/1a04e3eab45ca15dd86060f189eb133/model.binvox'
    with open(path, 'rb') as f:
        model = read_as_3d_array(f)
        # 尺寸(长宽高)，转化矩阵，放缩系数
        print(model.dims, model.translate, model.scale)
        for x in model.data:
            for y in x:
                print(y)