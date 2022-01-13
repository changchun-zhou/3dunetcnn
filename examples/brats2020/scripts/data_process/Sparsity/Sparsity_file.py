import cv2
import os
import torch 
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append("./scripts/data_process/Sparsity")
import tensor_to_file_act
import statistical_distribution

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--name", type=str, default="")
parser.add_argument("--batch", type=int, default=0)
args = parser.parse_args()

extract_src_dir = 'extract/extract_sp80/run/run_4/extract'
# extract_src_dir = '.'

dequant_dir = 'Extract/Extract/0_data_analysis'
file_epoch = 20
threshold = 2
batch =args.batch

act_dict = {}
act_dict = {
            'inputs_quant':{'shape':[batch,   4, 112, 112, 112], 'scale': 12.5, 'zero_point': 57}, # zp 57
            'module.encoder.layers.0.blocks.0.conv1.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64}, # zero_point 64
            'module.encoder.layers.0.blocks.0.conv2.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 9.8 , 'zero_point': 64}, # zero_point 64
            'module.encoder.layers.0.blocks.1.conv1.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 7.1 , 'zero_point': 64}, # 64
            'module.encoder.layers.0.blocks.1.conv2.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 8.3 , 'zero_point': 64},
            'module.encoder.layers.1.blocks.0.conv1.conv' :{'shape':[batch, 64,   56,   56,   56], 'scale': 5.4 , 'zero_point': 64},
            'module.encoder.layers.1.blocks.0.conv2.conv' :{'shape':[batch, 64,   56,   56,   56], 'scale': 6.4 , 'zero_point': 64},
            'module.encoder.layers.1.blocks.1.conv1.conv' :{'shape':[batch, 64,   56,   56,   56], 'scale': 5.0 , 'zero_point': 64},
            'module.encoder.layers.1.blocks.1.conv2.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.encoder.layers.2.blocks.0.conv1.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.encoder.layers.2.blocks.0.conv2.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.encoder.layers.2.blocks.1.conv1.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.encoder.layers.2.blocks.1.conv2.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.encoder.layers.3.blocks.0.conv1.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.encoder.layers.3.blocks.0.conv2.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.encoder.layers.3.blocks.1.conv1.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.encoder.layers.3.blocks.1.conv2.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.encoder.layers.4.blocks.0.conv1.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.encoder.layers.4.blocks.0.conv2.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.encoder.layers.4.blocks.1.conv1.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.encoder.layers.4.blocks.1.conv2.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.decoder.layers.0.blocks.0.conv1.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.decoder.layers.0.blocks.0.conv2.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.decoder.layers.1.blocks.0.conv1.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.decoder.layers.1.blocks.0.conv2.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.decoder.layers.2.blocks.0.conv1.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.decoder.layers.2.blocks.0.conv2.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.decoder.layers.3.blocks.0.conv1.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.decoder.layers.3.blocks.0.conv2.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.decoder.layers.4.blocks.0.conv1.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64},
            'module.decoder.layers.4.blocks.0.conv2.conv' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.3, 'zero_point': 64}

            }

            # 'inputs':{'shape':[8,   3, 16, 112, 112], 'scale': 0.3818}, # 0.3818
            # 'pool1' :{'shape':[8,  64, 16,  56,  56], 'scale': 0.0653},
            # 'pool2' :{'shape':[8, 128,  8,  28,  28], 'scale': 0.0351},
            # 'relu3' :{'shape':[8, 256,  8,  28,  28], 'scale': 0.0329},
            # 'pool3' :{'shape':[128, 256,  4,  14,  14], 'scale': 0.0331},
            # 'relu5' :{'shape':[128, 512,  4,  14,  14], 'scale': 0.0392},
            # 'pool4' :{'shape':[64, 512,  2,   7,   7], 'scale': 0.0551},
            # # 'relu7' :{'shape':[64, 512,  2,   7,   7], 'scale': 0.0870}
# print(type(act_dict) )
wei_dict = {}
wei_dict = {
            'conv1.float_weight' :{'shape':[64 ,  3, 3, 3, 3 ], 'scale': 152.29},
            'conv2.float_weight' :{'shape':[128, 64, 3, 3, 3 ], 'scale': 231.12},
            'conv3a.float_weight':{'shape':[256,128, 3, 3, 3 ], 'scale': 264.11},
            'conv3b.float_weight':{'shape':[256,256, 3, 3, 3 ], 'scale': 214.02},
            'conv4a.float_weight':{'shape':[512,256, 3, 3, 3 ], 'scale': 287.76},
            'conv4b.float_weight':{'shape':[512,512, 3, 3, 3 ], 'scale': 347.82},
            'conv5a.float_weight':{'shape':[512,512, 3, 3, 3 ], 'scale': 589.83},
            'conv5b.float_weight':{'shape':[512,512, 3, 3, 3 ], 'scale': 622.83}
            }

density_wei = np.array([10, 3, 5, 3, 5, 3, 5, 5])*10
# density_wei = np.array([50, 10, 10, 10,  10, 10, 10, 10])
factor_sparsity = np.array([32.0/3, 1, 1, 1,  1, 1, 4, 4 ])
# factor_sparsity = np.array([1, 1, 1, 1,  1, 1, 1, 1 ])
MACs = np.array([1040.449536*32/3,
11098.12838,
5549.064192,
11098.12838,
2774.532096,
5549.064192,
693.633024*4,
693.633024*4
])*2


print(extract_src_dir)
Number_Conv = 0
plt.figure(figsize=(45, 15))
plt.suptitle(args.name + ", batch = " + str(batch) + ', in ' + extract_src_dir)
import datetime   
date_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')

for net, shape_scale in act_dict.items():
    # if net == 'module.encoder.layers.0.blocks.0.conv1.conv' :
        tensor = torch.ones(shape_scale['shape'])
        scale = shape_scale['scale']
        zero_point = shape_scale['zero_point']
        print(' <<<< ', net,' >>>>')
        statistical_distribution.statistical_distribution(extract_dir=extract_src_dir, dequant_dir=dequant_dir, \
                name='Activation_'+str(file_epoch)+'_'+net,tensor=tensor,type='act',mode='dequant', scale=scale, theshold=threshold, \
                Number_Conv=Number_Conv, density_wei = density_wei[Number_Conv], MACs = MACs[Number_Conv], date_str= date_str, factor_sparsity=factor_sparsity[Number_Conv], file_name=args.name, zero_point=zero_point) # >= theshold
        Number_Conv += 1

# for net, shape_scale in wei_dict.items():
#     if net == 'conv2.float_weight':
#         tensor = torch.ones(shape_scale['shape'])
#         scale = shape_scale['scale']
#         print(' <<<< ', net,' >>>>')
#         statistical_distribution.statistical_distribution(extract_dir=extract_src_dir, dequant_dir=dequant_dir, \
#                 name='Weight_'+str(file_epoch)+'_'+net,tensor=tensor,type='wei',mode='dequant', scale=scale, theshold=threshold) # >= theshold

