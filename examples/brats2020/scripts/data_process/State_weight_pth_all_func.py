import cv2
import os
import torch 
import numpy as np
import C3D_model
from fnmatch import fnmatch 
from Function_self import Function_self
import tensor_to_file_act
import Verify_Conv
extract_src_dir = 'Extract/prune_quant_extract_proportion/run/run_0'

extract_save_dir = 'Extract/prune_quant_extract/prune_quant_extract_proportion'

Function_self = Function_self()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)
#C3D = C3D_model.C3D(num_classes=101, pretrained=False)

State_FileName = os.path.join(extract_save_dir, 'LeNet_Scale.dat')
Weight_FileName = os.path.join(extract_save_dir, 'Weight.dat')
# checkpoint = torch.load(os.path.join(extract_src_dir,'models/quantized_C3D-ucf101_epoch-45_checkpoint.pth.tar') \
#              ,map_location=lambda storage, loc: storage)
# checkpoint = torch.load("Extract/LeNet/LeNet_quant___2020.04.22-112340/LeNet_quant_checkpoint.pth.tar"
# ,map_location=lambda storage, loc: storage)
file_epoch = 45
#print(checkpoint.size())
# np.save('AlexNet_checkpoint.npy',checkpoint)
# with open (State_FileName,'w') as State_File:
# 	State_File.writelines(str(checkpoint))

# fp = open (State_FileName,'w')
# for key in checkpoint['state_dict']:
#     if fnmatch(key,'*scale'):
#         fp.write(key)
#         fp.write(str(checkpoint['state_dict'][key])+'\n')
#net_array = ['inputs', 'pool3b','relu5']
net_array = [
'conv1.weight',
'conv2.weight',
'conv3a.weight',
'conv3b.weight',
'conv4a.weight',
'conv4b.weight',
'conv5a.weight',
'conv5b.weight'
]
# test = [ 0 , 1]
# test.shape[0]
#for net in net_array:
# net = 'conv4a.float_weight'
#Function_self.tensor_to_file(extract_dir=extract_dir,name='Weight_'+str(file_epoch)+'_'+net,tensor=checkpoint['state_dict'][net],type='wei',mode='dequant', scale=289.3234)
net = 'pool1'
tensor = torch.ones(8,64,16,56,56) # ( batch, channel, frame, height, width )
# tensor_act = np.array([[[[[ 0 for x in range(56)] for y in range(56)] for z in range(16)] for m in range(64)] for n in range(8)])
tensor_act = np.zeros(shape=(8,64,16,56,56) )
tensor_act = tensor_to_file_act.tensor_to_file_act(extract_dir=extract_src_dir, \
            name='Activation_'+str(file_epoch)+'_'+net,tensor=tensor,type='act',mode='dequant', scale=0.0649)
net = 'conv2.float_weight'
tensor = torch.ones(128,64,3,3,3) # (channel_out, channel_in, frame, height, width)
# tensor_wei = [[[[[ 0 for x in range(3)] for y in range(3)] for z in range(3)] for m in range(64)] for n in range(128)]
tensor_wei = np.zeros(shape = (128,64,3,3,3))
tensor_wei = tensor_to_file_act.tensor_to_file_wei(extract_dir=extract_src_dir, \
            name='Weight_'+str(file_epoch)+'_'+net,tensor=tensor,type='wei',mode='dequant', scale=229.8959)
# bias = [ 0 for x in range(64)]

# test_out = Verify_Conv.test_conv(np.array(tensor_act),np.array(tensor_wei),np.array(bias),1,1,1,1)

'''
parm = {}

def parm_to_file(FileName,key_name,parm):
    with open (FileName,'w') as File:
        [output_num,input_num,frame,filter_size,_]=parm[key_name].size()
        parme = parm[key_name].numpy()
        for i in range(output_num):
            for j in range(input_num):
                for m in range(frame):
                    for n in range(filter_size):
                        for p in range(_):
                            #File.write(str(parm[key_name][i,j,m,n,p].detach().numpy()))
                            weight = int(parme[i,j,m,n,p])			    
                            File.write(str(weight))
                    
parm_to_file(Weight_FileName,'conv4a.wrapped_module.weight',checkpoint['state_dict'])
'''

'''print('-----------weight----------')
print(checkpoint['state_dict']['conv1.wrapped_module.weight'][0,0,:,:])
print('-----------scale---------')
print(checkpoint['state_dict']['conv1.w_scale'])
print('-----------num_forwards---------')
print(checkpoint['state_dict']['conv1.num_forwards'])
print('-----------zeropoint---------')

print(checkpoint['state_dict']['conv1.w_zero_point'])
print('-----------shifted---------')

print(checkpoint['state_dict']['conv1.is_simulated_quant_weight_shifted'])

model=C3D.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval
'''
