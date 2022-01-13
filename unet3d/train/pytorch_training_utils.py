"""
Modified from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import shutil
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import sys
import os
sys.path.append("./scripts/train")
import to_csv
from torch.autograd import Variable
#try:
#    from torch.utils.data._utils.collate import default_collate
#except ModuleNotFoundError:
    # import from older versions of pytorch
from torch.utils.data.dataloader import default_collate

# -----extract
from fnmatch import fnmatch
from Function_self import Function_self
Function_self = Function_self()
activation = {}
def get_activation(name):
     def hook(model,input,output):
         activation[name] = output.detach()
     return hook

calc_names,calc_modules,calc_outshapes = [],[],[]
def my_hook(Module, input, output):
    calc_outshapes.append(output.shape)
    calc_modules.append(Module)
def calc_paras_flops(modules, outshapes):
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
        print('{}: paras: {:.4f} M, flops: {:.4f} M'.format(calc_names[i], para_nums/1000000, flops/1000000) )
        total_para_nums += para_nums
        total_flops += flops
    print('total conv parameters: {:.4f} M, total conv FLOPs: {:.4f} M'.format(total_para_nums/1000000,total_flops/1000000) )
    return total_para_nums, total_flops

def hook_scale_zp(model):
        model.module.encoder.layers[0].blocks[0].conv1.scale =  model.module.encoder.layers[0].blocks[0].conv1.relu.fake_q.scale.item()
        model.module.encoder.layers[0].blocks[0].conv2.scale =  model.module.encoder.layers[0].blocks[0].conv2.relu.fake_q.scale.item()
        model.module.encoder.layers[0].blocks[1].conv1.scale =  model.module.encoder.layers[0].blocks[1].conv1.relu.fake_q.scale.item()
        model.module.encoder.layers[0].blocks[1].conv2.scale =  model.module.encoder.layers[0].blocks[1].conv2.relu.fake_q.scale.item()
        model.module.encoder.layers[1].blocks[0].conv1.scale =  model.module.encoder.layers[1].blocks[0].conv1.relu.fake_q.scale.item()
        model.module.encoder.layers[1].blocks[0].conv2.scale =  model.module.encoder.layers[1].blocks[0].conv2.relu.fake_q.scale.item()
        model.module.encoder.layers[1].blocks[1].conv1.scale =  model.module.encoder.layers[1].blocks[1].conv1.relu.fake_q.scale.item()
        model.module.encoder.layers[1].blocks[1].conv2.scale =  model.module.encoder.layers[1].blocks[1].conv2.relu.fake_q.scale.item()
        model.module.encoder.layers[2].blocks[0].conv1.scale =  model.module.encoder.layers[2].blocks[0].conv1.relu.fake_q.scale.item()
        model.module.encoder.layers[2].blocks[0].conv2.scale =  model.module.encoder.layers[2].blocks[0].conv2.relu.fake_q.scale.item()
        model.module.encoder.layers[2].blocks[1].conv1.scale =  model.module.encoder.layers[2].blocks[1].conv1.relu.fake_q.scale.item()
        model.module.encoder.layers[2].blocks[1].conv2.scale =  model.module.encoder.layers[2].blocks[1].conv2.relu.fake_q.scale.item()
        model.module.encoder.layers[3].blocks[0].conv1.scale =  model.module.encoder.layers[3].blocks[0].conv1.relu.fake_q.scale.item()
        model.module.encoder.layers[3].blocks[0].conv2.scale =  model.module.encoder.layers[3].blocks[0].conv2.relu.fake_q.scale.item()
        model.module.encoder.layers[3].blocks[1].conv1.scale =  model.module.encoder.layers[3].blocks[1].conv1.relu.fake_q.scale.item()
        model.module.encoder.layers[3].blocks[1].conv2.scale =  model.module.encoder.layers[3].blocks[1].conv2.relu.fake_q.scale.item()
        model.module.encoder.layers[4].blocks[0].conv1.scale =  model.module.encoder.layers[4].blocks[0].conv1.relu.fake_q.scale.item()
        model.module.encoder.layers[4].blocks[0].conv2.scale =  model.module.encoder.layers[4].blocks[0].conv2.relu.fake_q.scale.item()
        model.module.encoder.layers[4].blocks[1].conv1.scale =  model.module.encoder.layers[4].blocks[1].conv1.relu.fake_q.scale.item()
        model.module.encoder.layers[4].blocks[1].conv2.scale =  model.module.encoder.layers[4].blocks[1].conv2.relu.fake_q.scale.item()
        model.module.decoder.layers[0].blocks[0].conv1.scale =  model.module.decoder.layers[0].blocks[0].conv1.relu.fake_q.scale.item()
        model.module.decoder.layers[0].blocks[0].conv2.scale =  model.module.decoder.layers[0].blocks[0].conv2.relu.fake_q.scale.item()
        model.module.decoder.layers[1].blocks[0].conv1.scale =  model.module.decoder.layers[1].blocks[0].conv1.relu.fake_q.scale.item()
        model.module.decoder.layers[1].blocks[0].conv2.scale =  model.module.decoder.layers[1].blocks[0].conv2.relu.fake_q.scale.item()
        model.module.decoder.layers[2].blocks[0].conv1.scale =  model.module.decoder.layers[2].blocks[0].conv1.relu.fake_q.scale.item()
        model.module.decoder.layers[2].blocks[0].conv2.scale =  model.module.decoder.layers[2].blocks[0].conv2.relu.fake_q.scale.item()
        model.module.decoder.layers[3].blocks[0].conv1.scale =  model.module.decoder.layers[3].blocks[0].conv1.relu.fake_q.scale.item()
        model.module.decoder.layers[3].blocks[0].conv2.scale =  model.module.decoder.layers[3].blocks[0].conv2.relu.fake_q.scale.item()
        model.module.decoder.layers[4].blocks[0].conv1.scale =  model.module.decoder.layers[4].blocks[0].conv1.relu.fake_q.scale.item()
        model.module.decoder.layers[4].blocks[0].conv2.scale =  model.module.decoder.layers[4].blocks[0].conv2.relu.fake_q.scale.item()


        model.module.encoder.layers[0].blocks[0].conv1.zero_point =  model.module.encoder.layers[0].blocks[0].conv1.relu.fake_q.zero_point.item()
        model.module.encoder.layers[0].blocks[0].conv2.zero_point =  model.module.encoder.layers[0].blocks[0].conv2.relu.fake_q.zero_point.item()
        model.module.encoder.layers[0].blocks[1].conv1.zero_point =  model.module.encoder.layers[0].blocks[1].conv1.relu.fake_q.zero_point.item()
        model.module.encoder.layers[0].blocks[1].conv2.zero_point =  model.module.encoder.layers[0].blocks[1].conv2.relu.fake_q.zero_point.item()
        model.module.encoder.layers[1].blocks[0].conv1.zero_point =  model.module.encoder.layers[1].blocks[0].conv1.relu.fake_q.zero_point.item()
        model.module.encoder.layers[1].blocks[0].conv2.zero_point =  model.module.encoder.layers[1].blocks[0].conv2.relu.fake_q.zero_point.item()
        model.module.encoder.layers[1].blocks[1].conv1.zero_point =  model.module.encoder.layers[1].blocks[1].conv1.relu.fake_q.zero_point.item()
        model.module.encoder.layers[1].blocks[1].conv2.zero_point =  model.module.encoder.layers[1].blocks[1].conv2.relu.fake_q.zero_point.item()
        model.module.encoder.layers[2].blocks[0].conv1.zero_point =  model.module.encoder.layers[2].blocks[0].conv1.relu.fake_q.zero_point.item()
        model.module.encoder.layers[2].blocks[0].conv2.zero_point =  model.module.encoder.layers[2].blocks[0].conv2.relu.fake_q.zero_point.item()
        model.module.encoder.layers[2].blocks[1].conv1.zero_point =  model.module.encoder.layers[2].blocks[1].conv1.relu.fake_q.zero_point.item()
        model.module.encoder.layers[2].blocks[1].conv2.zero_point =  model.module.encoder.layers[2].blocks[1].conv2.relu.fake_q.zero_point.item()
        model.module.encoder.layers[3].blocks[0].conv1.zero_point =  model.module.encoder.layers[3].blocks[0].conv1.relu.fake_q.zero_point.item()
        model.module.encoder.layers[3].blocks[0].conv2.zero_point =  model.module.encoder.layers[3].blocks[0].conv2.relu.fake_q.zero_point.item()
        model.module.encoder.layers[3].blocks[1].conv1.zero_point =  model.module.encoder.layers[3].blocks[1].conv1.relu.fake_q.zero_point.item()
        model.module.encoder.layers[3].blocks[1].conv2.zero_point =  model.module.encoder.layers[3].blocks[1].conv2.relu.fake_q.zero_point.item()
        model.module.encoder.layers[4].blocks[0].conv1.zero_point =  model.module.encoder.layers[4].blocks[0].conv1.relu.fake_q.zero_point.item()
        model.module.encoder.layers[4].blocks[0].conv2.zero_point =  model.module.encoder.layers[4].blocks[0].conv2.relu.fake_q.zero_point.item()
        model.module.encoder.layers[4].blocks[1].conv1.zero_point =  model.module.encoder.layers[4].blocks[1].conv1.relu.fake_q.zero_point.item()
        model.module.encoder.layers[4].blocks[1].conv2.zero_point =  model.module.encoder.layers[4].blocks[1].conv2.relu.fake_q.zero_point.item()
        model.module.decoder.layers[0].blocks[0].conv1.zero_point =  model.module.decoder.layers[0].blocks[0].conv1.relu.fake_q.zero_point.item()
        model.module.decoder.layers[0].blocks[0].conv2.zero_point =  model.module.decoder.layers[0].blocks[0].conv2.relu.fake_q.zero_point.item()
        model.module.decoder.layers[1].blocks[0].conv1.zero_point =  model.module.decoder.layers[1].blocks[0].conv1.relu.fake_q.zero_point.item()
        model.module.decoder.layers[1].blocks[0].conv2.zero_point =  model.module.decoder.layers[1].blocks[0].conv2.relu.fake_q.zero_point.item()
        model.module.decoder.layers[2].blocks[0].conv1.zero_point =  model.module.decoder.layers[2].blocks[0].conv1.relu.fake_q.zero_point.item()
        model.module.decoder.layers[2].blocks[0].conv2.zero_point =  model.module.decoder.layers[2].blocks[0].conv2.relu.fake_q.zero_point.item()
        model.module.decoder.layers[3].blocks[0].conv1.zero_point =  model.module.decoder.layers[3].blocks[0].conv1.relu.fake_q.zero_point.item()
        model.module.decoder.layers[3].blocks[0].conv2.zero_point =  model.module.decoder.layers[3].blocks[0].conv2.relu.fake_q.zero_point.item()
        model.module.decoder.layers[4].blocks[0].conv1.zero_point =  model.module.decoder.layers[4].blocks[0].conv1.relu.fake_q.zero_point.item()
        model.module.decoder.layers[4].blocks[0].conv2.zero_point =  model.module.decoder.layers[4].blocks[0].conv2.relu.fake_q.zero_point.item()
        scale_threshold = [\
        model.module.encoder.layers[0].blocks[0].conv1.scale,
        model.module.encoder.layers[0].blocks[0].conv2.scale,
        model.module.encoder.layers[0].blocks[1].conv1.scale,
        model.module.encoder.layers[0].blocks[1].conv2.scale,
        model.module.encoder.layers[1].blocks[0].conv1.scale,
        model.module.encoder.layers[1].blocks[0].conv2.scale,
        model.module.encoder.layers[1].blocks[1].conv1.scale,
        model.module.encoder.layers[1].blocks[1].conv2.scale,
        model.module.encoder.layers[2].blocks[0].conv1.scale,
        model.module.encoder.layers[2].blocks[0].conv2.scale,
        model.module.encoder.layers[2].blocks[1].conv1.scale,
        model.module.encoder.layers[2].blocks[1].conv2.scale,
        model.module.encoder.layers[3].blocks[0].conv1.scale,
        model.module.encoder.layers[3].blocks[0].conv2.scale,
        model.module.encoder.layers[3].blocks[1].conv1.scale,
        model.module.encoder.layers[3].blocks[1].conv2.scale,
        model.module.encoder.layers[4].blocks[0].conv1.scale,
        model.module.encoder.layers[4].blocks[0].conv2.scale,
        model.module.encoder.layers[4].blocks[1].conv1.scale,
        model.module.encoder.layers[4].blocks[1].conv2.scale,
        model.module.decoder.layers[0].blocks[0].conv1.scale,
        model.module.decoder.layers[0].blocks[0].conv2.scale,
        model.module.decoder.layers[1].blocks[0].conv1.scale,
        model.module.decoder.layers[1].blocks[0].conv2.scale,
        model.module.decoder.layers[2].blocks[0].conv1.scale,
        model.module.decoder.layers[2].blocks[0].conv2.scale,
        model.module.decoder.layers[3].blocks[0].conv1.scale,
        model.module.decoder.layers[3].blocks[0].conv2.scale,
        model.module.decoder.layers[4].blocks[0].conv1.scale,
        model.module.decoder.layers[4].blocks[0].conv2.scale]

        zero_point_threshold = [\
        model.module.encoder.layers[0].blocks[0].conv1.zero_point,
        model.module.encoder.layers[0].blocks[0].conv2.zero_point,
        model.module.encoder.layers[0].blocks[1].conv1.zero_point,
        model.module.encoder.layers[0].blocks[1].conv2.zero_point,
        model.module.encoder.layers[1].blocks[0].conv1.zero_point,
        model.module.encoder.layers[1].blocks[0].conv2.zero_point,
        model.module.encoder.layers[1].blocks[1].conv1.zero_point,
        model.module.encoder.layers[1].blocks[1].conv2.zero_point,
        model.module.encoder.layers[2].blocks[0].conv1.zero_point,
        model.module.encoder.layers[2].blocks[0].conv2.zero_point,
        model.module.encoder.layers[2].blocks[1].conv1.zero_point,
        model.module.encoder.layers[2].blocks[1].conv2.zero_point,
        model.module.encoder.layers[3].blocks[0].conv1.zero_point,
        model.module.encoder.layers[3].blocks[0].conv2.zero_point,
        model.module.encoder.layers[3].blocks[1].conv1.zero_point,
        model.module.encoder.layers[3].blocks[1].conv2.zero_point,
        model.module.encoder.layers[4].blocks[0].conv1.zero_point,
        model.module.encoder.layers[4].blocks[0].conv2.zero_point,
        model.module.encoder.layers[4].blocks[1].conv1.zero_point,
        model.module.encoder.layers[4].blocks[1].conv2.zero_point,
        model.module.decoder.layers[0].blocks[0].conv1.zero_point,
        model.module.decoder.layers[0].blocks[0].conv2.zero_point,
        model.module.decoder.layers[1].blocks[0].conv1.zero_point,
        model.module.decoder.layers[1].blocks[0].conv2.zero_point,
        model.module.decoder.layers[2].blocks[0].conv1.zero_point,
        model.module.decoder.layers[2].blocks[0].conv2.zero_point,
        model.module.decoder.layers[3].blocks[0].conv1.zero_point,
        model.module.decoder.layers[3].blocks[0].conv2.zero_point,
        model.module.decoder.layers[4].blocks[0].conv1.zero_point,
        model.module.decoder.layers[4].blocks[0].conv2.zero_point]
        return [scale_threshold, zero_point_threshold]  
def get_tdvd_proportion(model, nsamples):
        tdvd_proportion = []
        tdvd_proportion.extend(model.module.encoder.layers[0].blocks[0].conv1.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[0].blocks[0].conv2.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[0].blocks[1].conv1.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[0].blocks[1].conv2.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[1].blocks[0].conv1.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[1].blocks[0].conv2.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[1].blocks[1].conv1.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[1].blocks[1].conv2.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[2].blocks[0].conv1.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[2].blocks[0].conv2.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[2].blocks[1].conv1.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[2].blocks[1].conv2.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[3].blocks[0].conv1.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[3].blocks[0].conv2.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[3].blocks[1].conv1.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[3].blocks[1].conv2.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[4].blocks[0].conv1.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[4].blocks[0].conv2.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[4].blocks[1].conv1.tdvd_proportion)
        tdvd_proportion.extend(model.module.encoder.layers[4].blocks[1].conv2.tdvd_proportion)
        tdvd_proportion.extend(model.module.decoder.layers[0].blocks[0].conv1.tdvd_proportion)
        tdvd_proportion.extend(model.module.decoder.layers[0].blocks[0].conv2.tdvd_proportion)
        tdvd_proportion.extend(model.module.decoder.layers[1].blocks[0].conv1.tdvd_proportion)
        tdvd_proportion.extend(model.module.decoder.layers[1].blocks[0].conv2.tdvd_proportion)
        tdvd_proportion.extend(model.module.decoder.layers[2].blocks[0].conv1.tdvd_proportion)
        tdvd_proportion.extend(model.module.decoder.layers[2].blocks[0].conv2.tdvd_proportion)
        tdvd_proportion.extend(model.module.decoder.layers[3].blocks[0].conv1.tdvd_proportion)
        tdvd_proportion.extend(model.module.decoder.layers[3].blocks[0].conv2.tdvd_proportion)
        tdvd_proportion.extend(model.module.decoder.layers[4].blocks[0].conv1.tdvd_proportion)
        tdvd_proportion.extend(model.module.decoder.layers[4].blocks[0].conv2.tdvd_proportion)
        return [x/nsamples for x in tdvd_proportion]
import gc
def epoch_training(train_loader, model, criterion, optimizer, epoch, compression_scheduler, config, extract=None,n_gpus=None, print_frequency=1, regularized=False,
                   print_gpu_memory=True, vae=False, scaler=None, save_dir='./', device=None):
    extract_minibatch = 0
    dict_activation = {}
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    scores = []
    for i in range(6):
        scores.append(AverageMeter('Score', ':.4e') )
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    use_amp = scaler is not None

    # switch to train mode
    model.train()

    end = time.time()
    # minibatch_id = 0
    print('***** current epoch: ', epoch)
    print('***** scale_threshold: {}'.format(hook_scale_zp(model)[0]))
    print('***** zero_point_threshold: {}'.format(hook_scale_zp(model)[1]))
    for i, (images, target) in enumerate(train_loader):
        if config['fast_test']:
            if i*config['batch_size'] >= config['extract_batch']:
                break
        
         
        gc.collect()
        torch.cuda.empty_cache()
    # for i, (images, target) in get_while_running(processes[0], queue):
        # print("images.shape: ", i, images.shape)
        # print("target.shape: ", target.shape)
        # print("distiller---on_minibatch_begin")
        # print("batch_id: ", i, "minibatches_per_epoch: ", len(train_loader))
        # torch.save(images, "images.pt")
        # torch.save(target, "target.pt")
        # break
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(
                        epoch,minibatch_id=i,minibatches_per_epoch=
                        len(train_loader),optimizer=optimizer)
        # measure data loading time
        data_time.update(time.time() - end)

        # if n_gpus:
        #     # torch.cuda.empty_cache()
        #     if print_gpu_memory:
        #         for i_gpu in range(n_gpus):
        #             print("Memory allocated (device {}):".format(i_gpu),
        #                   human_readable_size(torch.cuda.memory_allocated(i_gpu)))
        #             print("Max memory allocated (device {}):".format(i_gpu),
        #                   human_readable_size(torch.cuda.max_memory_allocated(i_gpu)))
        #             print("Memory cached (device {}):".format(i_gpu),
        #                   human_readable_size(torch.cuda.memory_cached(i_gpu)))
        #             print("Max memory cached (device {}):".format(i_gpu),
        #                   human_readable_size(torch.cuda.max_memory_cached(i_gpu)))

        optimizer.zero_grad()

        # -----extract
        # if i==0:
        #     extract_act_tensor = images.unsqueeze(4)
            
        extract_condition = extract and epoch == config['epoch_start_valtest'] and i*config['batch_size'] < config['extract_batch']
        if extract_condition : # 256
            if n_gpus > 0:
                layers = list(model.module._modules.items())
            else:
                layers = list(model._modules.items())
            for encoder_decoder in layers:
                # print(encoder_decoder,type(encoder_decoder))
                if fnmatch(encoder_decoder[0], '*coder'):
                    for number in encoder_decoder[1].layers._modules.items():
                        # print(number,type(number))
                        for number_y in number[1].blocks._modules.items():
                            # print('number_y', number_y, type(number_y))
                            for conv12 in number_y[1]._modules.items(): # conv1 / conv2
                                # print('conv12',conv12)
                                for [name, layer] in conv12[1]._modules.items():
                                    # print("name: ",name)
                                    if fnmatch(name,'*relu*')==True:
                                        name = 'module.' + encoder_decoder[0] + '.' + 'layers'+'.' + str(number[0]) + '.'+ 'blocks' +  '.' + number_y[0]+  '.' + conv12[0] + '.'+ name
                                        # print(name)
                                        layer.register_forward_hook(get_activation(name))
            extract_dir = os.path.join(save_dir,'extract')
            if not os.path.exists(extract_dir):
                os.makedirs(extract_dir)
            # extract_act_tensor = torch.stack((extract_act_tensor, images), 0)
            if i == 0:
                dict_activation['inputs_quant'] = images
            else:
                dict_activation['inputs_quant'] = torch.cat((dict_activation['inputs_quant'].to(torch.device('cuda:1')),images.to(torch.device('cuda:1'))))
            dict_activation['inputs_quant'].to(torch.device('cuda:1'))
            # Function_self.tensor_to_file(extract_dir=extract_dir,name='Activation_'+str(epoch)+'_'+'inputs_quant',tensor=images,type='act', mode='extract', scale=1)
            # torch.save(images,)
            # for net,param in model.named_parameters():
                # print(net,param.shape)

        # *** calculate 
        
        # for name,m in model.named_modules():
        #     if isinstance(m,torch.nn.Conv3d): 
        #         m.register_forward_hook(my_hook)
        #         calc_names.append(name)

        loss_weight, batch_size, score = batch_loss(model, images, target, criterion, n_gpus=n_gpus, regularized=regularized,
                                      vae=vae, use_amp=use_amp, save_dir=save_dir, epoch=epoch, device=device)
        # *** calculate 
        # total_para_nums, total_flops = calc_paras_flops(calc_modules, calc_outshapes) 

        # exp_th = 10.0/12 #scale

        # min_loss = 0.08
        # a = 0.1
        # b = min_loss*torch.norm(torch.full([30], exp_th))*exp_th #*weight_optim_th # torch.norm(exp_th,)

        # threshold_tensor = torch.zeros([1]).to(device)
        threshold_list = []
        for net,param in model.named_parameters():
            if fnmatch(net, '*threshold*'):
                threshold_list.append(param)
        threshold = torch.cat(threshold_list)
        loss_th = float(config['lambda'])/(torch.norm(threshold) + config['bias'])
        loss = loss_weight + loss_th
        # loss = loss_weight
        # print('threshold_list: {}, loss_weight: {} loss_th: {}'.format(threshold, loss_weight, loss_th))
        log_threshold_training = i % (len(train_loader)//batch_size//10 + 1) ==0
        if log_threshold_training:
            # print("loss_weight: {}, loss_th: {}, loss: {}".format(loss_weight, loss_th, loss))
            print("epoch: {}, loss_weight: {}, loss_th: {}, loss: {} \n, threshold: {}".format(epoch, loss_weight, loss_th, loss, (threshold.cpu().detach().numpy()).tolist()))
        if extract_condition :
            # All activations of all layers
            for encoder_decoder in layers:
                # print(encoder_decoder,type(encoder_decoder))
                if fnmatch(encoder_decoder[0], '*coder'):
                    for number in encoder_decoder[1].layers._modules.items():
                        # print(number,type(number))
                        for number_y in number[1].blocks._modules.items():
                            # print('number_y', number_y, type(number_y))
                            for conv12 in number_y[1]._modules.items(): # conv1 / conv2
                                # print('conv12',conv12)
                                for [name, layer] in conv12[1]._modules.items():
                                    # print("name: ",name)
                                    if fnmatch(name,'*relu*')==True:
                                        name = 'module.' + encoder_decoder[0] + '.' + 'layers'+'.' + str(number[0]) + '.'+ 'blocks' +  '.' + number_y[0]+  '.' + conv12[0] + '.'+ name
                                        # print(name, activation[name].shape)
                                        # extract_act_tensor = torch.stack((extract_act_tensor, activation[name]), 0)
                                        if i == 0:
                                            dict_activation[name] = activation[name]
                                        else:
                                            dict_activation[name] = torch.cat((dict_activation[name].to(torch.device('cuda:1')),activation[name].to(torch.device('cuda:1'))))
                                        dict_activation[name].to(torch.device('cuda:1'))
                                        # Function_self.tensor_to_file(extract_dir=extract_dir,name='Activation_'+str(epoch)+'_'+name,tensor=activation[name],type='act', mode='extract', scale=1)

        # print("epoch_training batch_size: ", batch_size)
        # score = dice_coefficient(input, target):
        if n_gpus:
            torch.cuda.empty_cache()

        # measure accuracy and record loss
        losses.update(loss.item(), batch_size)

        
        for i_score in range(len(score)):
            scores[i_score].update(score[i_score], batch_size)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # compute gradient and do step

            # distiller
            # print("distiller---before_backward_pass")
            if compression_scheduler:
                agg_loss = compression_scheduler.before_backward_pass(
                        epoch,minibatch_id=i,minibatches_per_epoch=len(train_loader),
                        loss=loss,return_loss_components=True,optimizer=optimizer)
                loss = agg_loss.overall_loss
            loss.backward()

            # print("distiller---before_parameter_optimization")
            if compression_scheduler:
                compression_scheduler.before_parameter_optimization(epoch,minibatch_id=i,minibatches_per_epoch=len(train_loader),optimizer=optimizer)
            optimizer.step()

        del loss
        del score
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_frequency == 0:
            progress.display(i)

        # print("distiller---on_minibatch_end")
        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch,minibatch_id=i,
                        minibatches_per_epoch=len(train_loader),optimizer=optimizer)
        # minibatch_id += 1
        # print("****** TDVD model.module.encoder.layers[0].blocks[0].conv2: {}".format(model.module.encoder.layers[0].blocks[0].conv2.tdvd_proportion/( (i+1)*batch_size)) )
        # print("****** TDVD model.module.decoder.layers[0].blocks[0].conv2: {}".format(model.module.decoder.layers[0].blocks[0].conv2.tdvd_proportion/( (i+1)*batch_size)) )
        tdvd_nsamples = batch_size*(i+1)
        to_csv.to_csv(save_dir + '/tdvd_range_'+ str(model.module.encoder.layers[0].blocks[0].conv2.tdvd_range)+'_scale_factor_'+str(model.module.encoder.layers[0].blocks[0].conv2.scale_factor)+'.csv', \
            tdvd_nsamples, get_tdvd_proportion(model, tdvd_nsamples))
    if extract and epoch == config['epoch_start_valtest']:
        # dict_activation['scale_threshold'] = scale_threshold
        # dict_activation['zero_point_threshold'] = zero_point_threshold
        [dict_activation['scale_threshold'], dict_activation['zero_point_threshold']] = hook_scale_zp(model)
        torch.save(dict_activation, os.path.join(extract_dir,'Activation_epoch_'+str(epoch) + '_batch_'+str(config['extract_batch']) + '.pth.tar'))
        del dict_activation
        torch.cuda.empty_cache()
    if save_dir:
        save_list = []
        for i in range(6):
            save_list.append(scores[i].avg)
        save_list.extend( (threshold.cpu().detach().numpy()).tolist())
        save_list.append(loss_weight)
        save_list.append(loss_th)
        save_list.append(optimizer.param_groups[0]['lr'])
        save_list.append(optimizer.param_groups[1]['lr'])
        save_list.extend(hook_scale_zp(model)[0])
        save_list.extend(hook_scale_zp(model)[1])
        to_csv.to_csv(save_dir+'/train_acc_loss.csv', epoch, save_list)

    return losses.avg

def dice_score_func(output, target):
    iflat = output.view(-1).float()
    tflat = target.view(-1).float()
    intersection = (iflat * tflat).sum()
    smooth = 1e-5
    dice_score = (2. * intersection + smooth)/(iflat.sum() + tflat.sum() + smooth)
    
    return dice_score

def iou_score_func(output, target):
    smooth = 1e-5
 
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.1
    target_ = target > 0.1
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
 
    return (intersection + smooth) / (union + smooth)

def batch_loss(model, images, target, criterion, n_gpus=0, regularized=False, vae=False, use_amp=None, save_dir=None, epoch=None, device=None):
    if n_gpus is not None:
        if device is not None:
            images = Variable(images, requires_grad=False).to(device)
            target = Variable(target).to(device)
        else:
            images = images.cuda()
            target = target.cuda()
    # compute output
    # print('-------------device:', device)
    if use_amp:
        from torch.cuda.amp import autocast
        with autocast():
            return _batch_loss(model, images, target, criterion, regularized=regularized, vae=vae, save_dir=save_dir, epoch=epoch)
    else:
        return _batch_loss(model, images, target, criterion, regularized=regularized, vae=vae, save_dir=save_dir, epoch=epoch)


def _batch_loss(model, images, target, criterion, regularized=False, vae=False, save_dir=None, epoch=None):
    output = model(images)
    batch_size = images.size(0)
    if regularized:
        try:
            output, output_vae, mu, logvar = output
            loss = criterion(output, output_vae, mu, logvar, images, target)
        except ValueError:
            pred_y, pred_x = output
            loss = criterion(pred_y, pred_x, images, target)
    elif vae:
        pred_x, mu, logvar = output
        loss = criterion(pred_x, mu, logvar, target)
    else:
        loss = criterion(output, target)
    # score

    dice_score = dice_score_func(output, target).item()
    # print("-----score: ", dice_score)
    dice_core_regions = []
    for i in range(3):
        dice_core_regions.append( dice_score_func(output.permute(1,0,2,3,4)[i].contiguous(), target.permute(1,0,2,3,4)[i].contiguous()).item() )
    iou_score = iou_score_func(output, target).item()
    score = []
    score.append(dice_score)
    score.extend(dice_core_regions)
    score.append(iou_score)
    score.append(loss.item() )
    return loss, batch_size, score



def epoch_validatation(val_loader, model, criterion, config, optimizer,n_gpus, print_freq=1, regularized=False, vae=False, use_amp=False, epoch=0, save_dir='.'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    scores = []
    for i in range(6):
        scores.append(AverageMeter('Score', ':.4e') )
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if config['fast_test']:
                if i >0:
                    break
            loss_weight, batch_size, score = batch_loss(model, images, target, criterion, n_gpus=n_gpus, regularized=regularized,
                                          vae=vae, use_amp=use_amp)

            threshold_list = []
            for net,param in model.named_parameters():
                if fnmatch(net, '*threshold*'):
                    threshold_list.append(param)
            threshold = torch.cat(threshold_list)
            loss_th = float(config['lambda'])/(torch.norm(threshold) + config['bias'])
            loss = loss_weight + loss_th

            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)
            for i_score in range(len(score)):
                scores[i_score].update(score[i_score], batch_size)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)
    if save_dir:
        save_list = []
        for i in range(6):
            save_list.append(scores[i].avg)
        save_list.extend( (threshold.cpu().detach().numpy()).tolist())
        save_list.append(loss_weight)
        save_list.append(loss_th)
        save_list.append(optimizer.param_groups[0]['lr'])
        save_list.append(optimizer.param_groups[1]['lr'])
        save_list.extend(hook_scale_zp(model)[0])
        save_list.extend(hook_scale_zp(model)[1])
        for net,param in model.named_parameters():
            if fnmatch(net, '*conv1.scale*') or fnmatch(net, '*conv2.scale*'):
                save_list.append(param.item())
        to_csv.to_csv(save_dir+'/val_acc_loss.csv', epoch, save_list)

    return losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def human_readable_size(size, decimal_places=1):
    for unit in ['', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f}{unit}"


def collate_flatten(batch, x_dim_flatten=5, y_dim_flatten=2):
    x, y = default_collate(batch)
    if len(x.shape) > x_dim_flatten:
        x = x.flatten(start_dim=0, end_dim=len(x.shape) - x_dim_flatten)
    if len(y.shape) > y_dim_flatten:
        y = y.flatten(start_dim=0, end_dim=len(y.shape) - y_dim_flatten)
    return [x, y]


def collate_5d_flatten(batch, dim_flatten=5):
    return collate_flatten(batch, x_dim_flatten=dim_flatten, y_dim_flatten=dim_flatten)
