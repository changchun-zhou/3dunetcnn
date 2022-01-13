import torch 
import numpy as np 
#import Prune_quant
from torch.autograd import Variable
import os
class Function_self(object):
    def __init__(self):
        print("call Fuction_self")
    def test_conv(self,act,wei,bias,scale_act,scale_wei,scale_bias,scale_out):
        out = np.zeros(shape=(1,64,16,112,112))
        
        act = act.cpu().numpy()
        wei = wei.cpu().numpy()
        bias = bias.cpu().numpy()
        
        accum_channel = 0
        for batch in range(act.shape[0]): # all for 1
            for frame in range(1,2):# cut for no padding
                for channel in range(2):
                    for height in range(1,int(act.shape[3]/2-1)):
                        for width in range(1,int(act.shape[4]/2-1)):
                            center_x = width
                            center_y = height
                            for frame_i in range(wei.shape[2]):
                                accum_frame = 0
                                for channel_i in range(act.shape[1]):
                                    accum_channel = act[batch][channel_i][frame-1+frame_i][center_y-1][center_x-1]*wei[channel][channel_i][frame_i][0][0] + \
                                                    act[batch][channel_i][frame-1+frame_i][center_y-1][center_x  ]*wei[channel][channel_i][frame_i][0][1] + \
                                                    act[batch][channel_i][frame-1+frame_i][center_y-1][center_x+1]*wei[channel][channel_i][frame_i][0][2] +  \
                                                    act[batch][channel_i][frame-1+frame_i][center_y  ][center_x-1]*wei[channel][channel_i][frame_i][1][0] + \
                                                    act[batch][channel_i][frame-1+frame_i][center_y  ][center_x  ]*wei[channel][channel_i][frame_i][1][1] + \
                                                    act[batch][channel_i][frame-1+frame_i][center_y  ][center_x+1]*wei[channel][channel_i][frame_i][1][2] +  \
                                                    act[batch][channel_i][frame-1+frame_i][center_y+1][center_x-1]*wei[channel][channel_i][frame_i][2][0] + \
                                                    act[batch][channel_i][frame-1+frame_i][center_y+1][center_x  ]*wei[channel][channel_i][frame_i][2][1] + \
                                                    act[batch][channel_i][frame-1+frame_i][center_y+1][center_x+1]*wei[channel][channel_i][frame_i][2][2]
                                    accum_frame += accum_channel
                                out[batch][channel][frame][height][width] += accum_frame
                    out[batch][channel][frame] /= scale_act*scale_wei
                    bias_f = bias[channel]/scale_bias
                    out[batch][channel][frame] += [[bias_f]]
                    #out[batch][channel][frame] *= scale_out 
                    for height in range(1,int(act.shape[3]/2-1)): # relu
                        for width in range(1,int(act.shape[4]/2-1)):
                            #if out[batch][channel][frame][height][width] < 0:
                                #out[batch][channel][frame][height][width] = 0
                            # return to quantized output
                            out[batch][channel][frame][height][width] = out[batch][channel][frame][height][width] *scale_out
        out =torch.from_numpy(out)
        #print('out[0][1][1][1]')
        #print(out[0][1][1][1])
        #tensor_to_file('out',out)
        return out

    inputs = torch.Tensor(1,3,16,112,112)*10
    def Conv3d(self,inputs):
        model = torch.nn.Conv3d(3,64,kernel_size=(3,3,3),padding=(1,1,1))
        conv1 = model(inputs)
        weights = list(model.parameters())[0].data
        bias = list(model.parameters())[1].data
        test_out = test_conv(inputs,weights,bias,1,1,1,1)
        
        print('Conv3d.test_out[0][1][1][1][1]:',test_out[0][1][1][1][2])
        print('Conv3d.conv1[0][1][1][1][1]:',conv1[0][1][1][1][2])
        return
    def dec_to_hex(self, dec ):
        dec = int(dec)
        if dec < 0:
            dec = hex(dec & 0xff)
            hex_out = (str(dec).lstrip('0x')).rstrip('L');
        else:
            dec = hex(dec & 0xff)
            hex_out = (str(dec).lstrip('0x')).rstrip('L').zfill(2);
        return hex_out
    def dec_to_hex_32(self, dec ):
        dec = int(dec)
        if dec < 0:
            dec = hex(dec & 0xffffffff)
            hex_out = (str(dec).lstrip('0x')).rstrip('L');
        else:
            dec = hex(dec & 0xffffffff)
            hex_out = (str(dec).lstrip('0x')).rstrip('L').zfill(8);
        return hex_out
    ##############  print parameter ###############################################
    # Because activation and outputs is related to network
    # Run network as well as print or save inter parameter

    # tensor to file

    def tensor_to_file(self,extract_dir,name,tensor, type, mode,scale):
        # try:
        if mode == 'extract':
            print("<<<<<<<<< create extract write file >>>>>", extract_dir+name, "shape: ", tensor.shape)
            if type == "act":
                fp_float_wr = open(os.path.join(extract_dir)+'/'+name+'_float.dat','a+')
                print('Add write: ' + os.path.join(extract_dir)+'/'+name+'_float.dat')
            else:
                fp_float_wr = open(os.path.join(extract_dir)+'/'+name+'_float.dat','w')
        elif mode == 'dequant':
            dequant_dir = 'Extract/dequant/run/dequant_data'
            fp_data_wr = open(os.path.join(dequant_dir)+'/'+name+'_data.dat','w') # activation for delta
            #fp_data_delta_wr = open(name+'_data_delta.dat','w')
            fp_flag_wr = open(os.path.join(dequant_dir)+'/'+name+'_flag.dat','w')
            #fp_flag_delta_wr = open(name+'_flag_delta.dat','w')
            fp_flag_bin_wr = open(os.path.join(dequant_dir)+'/'+name+'_flag_bin.dat','w')
            fp_float_rd = open(os.path.join(extract_dir)+'/'+name+'_float.dat','r')
        
        array = (tensor.cpu()).numpy()
        shape = array.shape
        str_row = ''
        cnt_col = 0

        str_flgbyte = ''
        str_flgword = ''
        cnt_flgbit = 0
        cnt_flgbyte = 0
        cnt_sparsity = 0
        cnt_sparsity_delta = 0
        height_array = range(shape[3])
        if type == 'wei':
            height_array = height_array[::-1] # reverse for weight
        for batch in range(shape[0]):
            for frame in range(shape[2]):
                for height in height_array:
                    for width in range(shape[4]):
                        for channel in range(shape[1]): ### NOT SUPPORT CONV1: Channel = 3
                            if mode == 'extract': 
                                fp_float_wr.write(str(array[batch][channel][frame][height][width])+'\n')
                            elif mode == 'dequant':
                                float_rd = fp_float_rd.readline().rstrip('\n')

                                data = round(float(float_rd)*scale) #float-scale-int
                                if data != 0:
                                    cnt_sparsity += 1
                                array[batch][channel][frame][height][width] = data
                                if frame >= 1 and mode == 'dequant' and type=='act': # act delta 
                                    data -=  array[batch][channel][frame-1][height][width] #theshold=0

                                ###############################################################
                                ## Write real value
                                if data != 0 :# activation != 0 need to be wroten
                                    #******************************************            
                                    str_row = Function_self.dec_to_hex(self,data) + str_row; # >>
                                    if cnt_col == 11:
                                        fp_data_wr.write(str_row+'\n')
                                        cnt_col = 0;
                                        str_row = ''
                                    else:
                                        cnt_col += 1

                                    flag = 1;
                                    cnt_sparsity_delta += 1
                                else:
                                    flag = 0;

                                if type == 'wei':
                                    str_flgbyte = str_flgbyte + str(flag)
                                else:
                                    str_flgbyte = str(flag) + str_flgbyte
                                if cnt_flgbit == 7:
                                    str_flgword = Function_self.dec_to_hex(self,int(str_flgbyte,2))+ str_flgword
                                    if cnt_flgbyte == 11:
                                        fp_flag_wr.write(str_flgword+'\n')
                                        str_flgword = ''
                                        cnt_flgbyte = 0
                                    else:
                                        cnt_flgbyte += 1
                                    fp_flag_bin_wr.write(str_flgbyte+'\n')
                                    str_flgbyte = ''
                                    cnt_flgbit = 0
                                else:
                                    cnt_flgbit += 1
        # except:
        #     print('_'*45)
        #     print(name)
        #     print(batch,frame,height,width,channel)
        #     if mode == 'dequant':
        #         print(data)
        #     print(array[batch][channel][frame-1][height][width])
        # print(name)
        # if mode == 'dequant':
        #     print('sparsity:',(1-float(cnt_sparsity)/float(shape[0]*shape[1]*shape[2]*shape[3]*shape[4]))*100)
        #     print('sparsity_delta:',(1-float(cnt_sparsity_delta)/float(shape[0]*shape[1]*shape[2]*shape[3]*shape[4]))*100)
        return

    def narrow(self,inputs):
        inputs = inputs.numpy()
        inputs += np.array([[[[[90.0]]],[[[98.0]]],[[[102.0]]]]])#0-255
        inputs /= 2.0#0-127
        inputs -= np.array([[[[[64.0]]]]])#-64-63
        inputs = torch.from_numpy(inputs) 
        return inputs

    def tensor_vision(self,name,tensor):
        array = tensor.cpu().numpy()
        np.savetxt(name+'.txt',array)
        return

    # Only quant and extract Activations: split to Patches
    def quant_stats_split(self,extract_src_dir,extract_save_dir, name,tensor, type, mode,scale):
        # try:
        if mode == 'extract':
            fp_float_wr = open(os.path.join(extract_src_dir)+'/'+name+'_float.dat','w')
        elif mode == 'dequant':
            fp_data_wr = open(os.path.join(extract_save_dir)+'/'+name+'_data.dat','w') # activation for delta
            fp_flag_wr = open(os.path.join(extract_save_dir)+'/'+name+'_flag.dat','w')
            fp_flag_bin_wr = open(os.path.join(extract_save_dir)+'/'+name+'_flag_bin.dat','w')
            fp_float_rd = open(os.path.join(extract_src_dir)+'/'+name+'_float.dat','r')
            Patch_DDR_BASE_File = open(os.path.join(extract_save_dir)+'/'+"Patch_DDR_BASE_File.dat",'w')
            Patch_NUMBER_File = open(os.path.join(extract_save_dir)+'/'+"Patch_NUMBER_File.dat",'w')
        
        array = (tensor.cpu()).numpy()
        shape = array.shape
        str_row = ''
        cnt_col = 0

        str_flgbyte = ''
        str_flgword = ''
        cnt_flgbit = 0
        cnt_flgbyte = 0
        cnt_sparsity = 0
        cnt_sparsity_delta = 0
        cnt_sparsity_delta_all = 0
        cnt_sparsity_delta_th = 0

        height_array = range(shape[3])
        # if type == 'wei':
        #     height_array = height_array[::-1] # reverse for weight
        for batch in range(1):#Only 1 patch
            for frame in range(shape[2]):
                for height in height_array:
                    for width in range(shape[4]):
                        for channel in range(shape[1]): ### NOT SUPPORT CONV1: Channel = 3
                            if mode == 'extract': 
                                fp_float_wr.write(str(array[batch][channel][frame][height][width])+'\n')
                            elif mode == 'dequant':
                                float_rd = fp_float_rd.readline().rstrip('\n')

                                data = round(float(float_rd)*scale) #float-scale-int
                                if data != 0:
                                    cnt_sparsity += 1
                                array[batch][channel][frame][height][width] = data
                                if frame >= 1 and mode == 'dequant' and type=='act': # act delta 
                                    data -=  array[batch][channel][frame-1][height][width] #theshold=0

                                ###############################################################
                                ## Write real value
                                if data != 0 :# activation != 0 need to be wroten
                                    #******************************************            
                                    str_row = Function_self.dec_to_hex(self,data) + str_row; # >>
                                    if cnt_col == 11:
                                        #fp_data_wr.write(str_row+'\n')
                                        cnt_col = 0;
                                        str_row = ''
                                    else:
                                        cnt_col += 1

                                    flag = 1;
                                    
                                    cnt_sparsity_delta_all += 1 # all frames' sparsity
                                    if frame >= 1 and type == 'act':
                                        cnt_sparsity_delta += 1 # only delta frames' sparsity
                                        if data >=2 or data <= -2:
                                            cnt_sparsity_delta_th += 1 # with theshold(2) delta sparsity

                                else:
                                    flag = 0;

                                if type == 'wei':
                                    str_flgbyte = str_flgbyte + str(flag)
                                else:
                                    str_flgbyte = str(flag) + str_flgbyte
                                if cnt_flgbit == 7:
                                    str_flgword = Function_self.dec_to_hex(self,int(str_flgbyte,2))+ str_flgword
                                    if cnt_flgbyte == 11:
                                        #fp_flag_wr.write(str_flgword+'\n')
                                        str_flgword = ''
                                        cnt_flgbyte = 0
                                    else:
                                        cnt_flgbyte += 1
                                    #fp_flag_bin_wr.write(str_flgbyte+'\n')
                                    str_flgbyte = ''
                                    cnt_flgbit = 0
                                else:
                                    cnt_flgbit += 1

        patch_width = shape[4] / 16 + 1 # to integer
        patch_height = shape[3] / 16 + 1 # to integer
        NumBlk = shape[1]/32
        print(int(patch_width), int(patch_height), int(NumBlk))

        LAYER_ACT_DDR_BASE = int("080A5600", 16)
        LAYER_FLGACT_DDR_BASE = int("083B5600", 16)

        Cnt_ACT = 0
        Cnt_ACT_last = 0
        Cnt_FLGACT = 0
        Cnt_FLGACT_last = 0
        patch_width = int(patch_width)
        patch_height = int(patch_height)
        Patch_DDR_BASE = [[[0 for x in range(patch_height)] for y in range(patch_width)] for z in range(4)] 
        for batch in range(1):
            for patch_x in range( int(patch_width)):
                for patch_y in range( int(patch_height) ):
                    Patch_DDR_BASE[0][patch_x][patch_y] = LAYER_ACT_DDR_BASE + Cnt_ACT
                    
                    Patch_DDR_BASE[1][patch_x][patch_y] = LAYER_FLGACT_DDR_BASE + Cnt_FLGACT/8
                    
                    for frame in range( shape[2]):
                        for blk in range ( int(NumBlk) ):
                            for height in range (16 ):# patch size
                                for width in range( 16):
                                    for channel_i in range(32 ): # 32 channel of a block;
                                        Cnt_FLGACT += 1
                                        if (height-1)+14*patch_y < 0 or (height-1)+14*patch_y >= shape[3] \
                                            or (width-1)+14*patch_x < 0 or (width-1)+14*patch_x >= shape[4] :
                                            data = 0
                                        else:
                                            data  = array[batch][blk*32 + channel_i][frame] \
                                                     [(height-1)+14*patch_y][(width-1)+14*patch_x]

                                        ###############################################################
                                        ## Write real value
                                        if data != 0 :# activation != 0 need to be wroten
                                            # ******************************************            
                                            str_row = str_row + Function_self.dec_to_hex(self,data); # <<
                                            if cnt_col == 11:
                                                fp_data_wr.write(str_row+'\n')
                                                cnt_col = 0;
                                                str_row = ''
                                            else:
                                                cnt_col += 1

                                            flag = 1;
                                            Cnt_ACT += 1
                                            
                                        else:
                                            flag = 0;

                                        if type == 'wei':
                                            str_flgbyte = str_flgbyte + str(flag)
                                        else:
                                            str_flgbyte = str_flgbyte + str(flag) 
                                        if cnt_flgbit == 7:
                                            str_flgword = str_flgword + Function_self.dec_to_hex(self,int(str_flgbyte,2))
                                            if cnt_flgbyte == 11:
                                                fp_flag_wr.write(str_flgword+'\n')
                                                str_flgword = ''
                                                cnt_flgbyte = 0
                                            else:
                                                cnt_flgbyte += 1
                                            fp_flag_bin_wr.write(str_flgbyte+'\n')
                                            str_flgbyte = ''
                                            cnt_flgbit = 0
                                        else:
                                            cnt_flgbit += 1
                    Patch_NUMBER_File.write(str(Cnt_ACT-Cnt_ACT_last)+'\n')
                    Patch_NUMBER_File.write(str(Cnt_FLGACT-Cnt_FLGACT_last)+'\n')
                    Cnt_ACT_last = Cnt_ACT 
                    Cnt_FLGACT_last = Cnt_FLGACT
        # complete the last row
        if cnt_col != 0:
            str_row = str_row + "xx"*(12-cnt_col)
            fp_data_wr.write(str_row+'\n')
        if cnt_flgbyte != 0:
            str_flgword = str_flgword + "xx"*(12- cnt_flgbyte)
            fp_flag_wr.write(str_flgword+'\n')

        for i in range(2):
            for patch_x in range(patch_width):
                for patch_y in range(patch_height):
                   Patch_DDR_BASE_File.write(Function_self.dec_to_hex_32(self,Patch_DDR_BASE[i][patch_x][patch_y])+'\n')            
                                                   
        # except:
        #     print('_'*45)
        #     print(name)
        #     print(batch,channel, frame,height,width)
        #     if mode == 'dequant':
        #         print(data)
        #     print(array[batch][channel][frame-1][height][width])
        print(name)
        if mode == 'dequant':
            print('sparsity:',(1-float(cnt_sparsity)/float(shape[0]*shape[1]*shape[2]*shape[3]*shape[4]))*100)
            print('sparsity_delta_all:',(1-float(cnt_sparsity_delta_all)/float(shape[0]*shape[1]*shape[2]*shape[3]*shape[4]))*100)
            print('sparsity_delta:',    (1-float(cnt_sparsity_delta)    /float(shape[0]*shape[1]*(shape[2]-1)*shape[3]*shape[4]))*100)
            print('sparsity_delta_th:', (1-float(cnt_sparsity_delta_th)/float(shape[0]*shape[1]*(shape[2]-1)*shape[3]*shape[4]))*100)
        return



    def quant_stats(self,extract_src_dir,extract_save_dir, name,tensor, type, mode,scale):
        # try:
            if mode == 'extract':
                fp_float_wr = open(os.path.join(extract_src_dir)+'/'+name+'_float.dat','w')
            elif mode == 'dequant':
                #dequant_dir = 'Extract/dequant/run/dequant_data'
                fp_data_wr = open(os.path.join(extract_save_dir)+'/'+name+'_data.dat','w') # activation for delta
                #fp_data_delta_wr = open(name+'_data_delta.dat','w')
                fp_flag_wr = open(os.path.join(extract_save_dir)+'/'+name+'_flag.dat','w')
                #fp_flag_delta_wr = open(name+'_flag_delta.dat','w')
                fp_flag_bin_wr = open(os.path.join(extract_save_dir)+'/'+name+'_flag_bin.dat','w')
                fp_float_rd = open(os.path.join(extract_src_dir)+'/'+name+'_float.dat','r')
                Patch_DDR_BASE_File = open(os.path.join(extract_save_dir)+'/'+"Patch_DDR_BASE_File.dat",'a')
            
            array = (tensor.cpu()).numpy()
            shape = array.shape
            str_row = ''
            cnt_col = 0

            str_flgbyte = ''
            str_flgword = ''
            cnt_flgbit = 0
            cnt_flgbyte = 0
            cnt_sparsity = 0
            cnt_sparsity_delta = 0
            cnt_sparsity_delta_all = 0
            cnt_sparsity_delta_th = 0
            height_array = range(shape[3])
            Cnt_WEI = 0
            Cnt_WEI_last = 0
            Cnt_FLGWEI = 0
            Cnt_FLGWEI_last = 0
            LAYER_WEI_DDR_BASE = int("088016C8", 16)
            LAYER_FLGWEI_DDR_BASE = int("088376C8", 16)
            if type == 'wei':
                height_array = height_array[::-1] # reverse for weight
            #output_channel = int(shape[0] / 16) # Number FtrGrp
            Patch_DDR_BASE = [[0 for x in range(int(shape[0] / 16))] for y in range(2)]
            for batch in range(shape[0]):
                if batch % 16 ==0:#0 16 32 
                    Patch_DDR_BASE[0][int(batch /16)] = LAYER_WEI_DDR_BASE + Cnt_WEI
                    Patch_DDR_BASE[1][int(batch /16)] = LAYER_FLGWEI_DDR_BASE + Cnt_FLGWEI/8

                for frame in range(shape[2]):
                    for height in height_array:
                        for width in range(shape[4]):
                            for channel in range(shape[1]): ### NOT SUPPORT CONV1: Channel = 3

                                if mode == 'extract': 
                                    fp_float_wr.write(str(array[batch][channel][frame][height][width])+'\n')
                                elif mode == 'dequant':

                                    float_rd = fp_float_rd.readline().rstrip('\n')

                                    data = round(float(float_rd)*scale) #float-scale-int
                                    if data != 0:
                                        cnt_sparsity += 1
                                    array[batch][channel][frame][height][width] = data
                                    if frame >= 1 and mode == 'dequant' and type=='act': # act delta 
                                        data_delta -=  array[batch][channel][frame-1][height][width] #theshold=0

                                    ###############################################################
                                    ## Write real value
                                    Cnt_FLGWEI += 1
                                    if data_delta != 0 :# activation != 0 need to be wroten
                                        #******************************************            
                                        str_row = str_row + Function_self.dec_to_hex(self,data_delta); # >>
                                        if cnt_col == 11:
                                            fp_data_wr.write(str_row+'\n')
                                            cnt_col = 0;
                                            str_row = ''
                                        else:
                                            cnt_col += 1

                                        flag = 1;
                                        Cnt_WEI += 1
                                        cnt_sparsity_delta_all += 1 # all frames' sparsity
                                        data_delta_th = 0 #init
                                        if frame >= 1 and type == 'act':
                                            cnt_sparsity_delta += 1 # only delta frames' sparsity
                                            if data_delta >=2 or data_delta <= -2:
                                                cnt_sparsity_delta_th += 1 # with theshold(2) delta sparsity
                                                data_delta_th = data_delta
                                        



                                    else:
                                        flag = 0;

                                    if type == 'wei':
                                        str_flgbyte = str_flgbyte + str(flag)
                                    else:
                                        str_flgbyte = str_flgbyte + str(flag)
                                    if cnt_flgbit == 7:
                                        str_flgword = str_flgword + Function_self.dec_to_hex(self,int(str_flgbyte,2))
                                        if cnt_flgbyte == 11:
                                            fp_flag_wr.write(str_flgword+'\n')
                                            str_flgword = ''
                                            cnt_flgbyte = 0
                                        else:
                                            cnt_flgbyte += 1
                                        fp_flag_bin_wr.write(str_flgbyte+'\n')
                                        str_flgbyte = ''
                                        cnt_flgbit = 0
                                    else:
                                        cnt_flgbit += 1 
            # complete the last row
            if cnt_col != 0:
                str_row = str_row + "xx"*(12-cnt_col)
                fp_data_wr.write(str_row+'\n')
            if cnt_flgbyte != 0:
                str_flgword = str_flgword + "xx"*(12- cnt_flgbyte)
                fp_flag_wr.write(str_flgword+'\n')
                
            for i in range(2):
                for j in range(int(shape[0] / 16)):
                       Patch_DDR_BASE_File.write(Function_self.dec_to_hex_32(self,Patch_DDR_BASE[i][j])+'\n')            

        # except:
        #     print('_'*45)
        #     print(name)
        #     print(batch,frame,height,width,channel)
        #     if mode == 'dequant':
        #         print(data)
        #     print(array[batch][channel][frame-1][height][width])
            print(name)
            return

'''
size = (1,3,3,3)
input=torch.rand(size)

input=Variable(input)
x=torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=(3,3))
out=x(input)

f_p=list(x.parameters())[0]
f_p=f_p.data.numpy()
f_b=list(x.parameters())[1]
f_b=f_b.data.numpy()
print("parameters:",list(x.parameters()))
print("output result is:", out[0])
#print("the result of first channel in image:", f_p[0].sum()+f_b[0])
accum = 0
for z in range(3):
  for i in range(3):
    for j in range(3):
      accum += (input[0][z][i][j]).numpy()*f_p[0][z][i][j]
test_out = accum + f_b[0]
print("test_out:", test_out)
''' 
