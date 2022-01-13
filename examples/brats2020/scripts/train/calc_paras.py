model = XXX_Net()#注：以下代码放在模型实例化之后，模型名用model

def my_hook(Module, input, output):
    outshapes.append(output.shape)
    modules.append(Module)

names,modules,outshapes = [],[],[]
for name,m in model.named_modules():
    if isinstance(m,nn.Conv3d): 
        m.register_forward_hook(my_hook)
        names.append(name)

# torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

def calc_paras_flops(modules, inshapes, outshapes):
    total_para_nums = 0
    total_flops = 0
    for i,m in enumerate(modules):

        Cin = m.in_channels
        Cout = m.out_channels
        k = m.kernel_size
        g = m.groups

        Dout = outshapes[i][2]
        Hout = outshapes[i][3]
        Wout = outshapes[i][4]
        # if m.bias is None:
        #     para_nums = k[0] * k[1] * Cin / g * Cout
        #     flops = (2 * k[0] * k[1] * Cin/g - 1) * Cout * Hout * Wout
        # else:
        para_nums = (k[0] * k[1] * k[2] * Cin / g +1) * Cout
        flops = 2 * k[0] * k[1] * k[2] * Cin/g * Dout* Cout * Hout * Wout
        para_nums = int(para_nums)
        flops = int(flops)
        print(names[i], 'para:', para_nums, 'flops:',flops)
        total_para_nums += para_nums
        total_flops += flops
    print('total conv parameters:',total_para_nums, 'total conv FLOPs:',total_flops)
    return total_para_nums, total_flops

input = torch.rand(32,3,224,224)#需要先提供一个输入张量
y = model(input)
total_para_nums, total_flops = calc_paras_flops(modules, outshapes)
