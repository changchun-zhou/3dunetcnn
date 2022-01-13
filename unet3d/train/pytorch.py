import os
import shutil
import warnings
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn

from ..models.pytorch.build import build_or_load_model
from ..utils.pytorch import WholeBrainCIFTI2DenseScalarDataset
from .pytorch_training_utils import epoch_training, epoch_validatation, collate_flatten, collate_5d_flatten
from ..utils.pytorch import functions
from ..utils.utils import in_config
from distutils.dir_util import copy_tree
import sys
sys.path.append("../../../distiller")
# sys.path.append("/workspace/Pytorch/distiller-master")
from distiller.data_loggers import *

# compress = "../../unet3d/quant_prune.yaml"
# compress = None
def build_optimizer(optimizer_name, model_parameters, learning_rate=1e-4):
    return getattr(torch.optim, optimizer_name)(model_parameters, lr=learning_rate)


def run_pytorch_training(config, model_filename, training_log_filename, verbose=1, use_multiprocessing=False,
                         n_workers=1, max_queue_size=5, model_name='resnet_34', n_gpus=1, regularized=False,
                         sequence_class=WholeBrainCIFTI2DenseScalarDataset, directory=None, test_input=1,
                         metric_to_monitor="loss", model_metrics=(), bias=None, pin_memory=True, amp=False,
                         **unused_args):
    """
    :param test_input: integer with the number of inputs from the generator to write to file. 0, False, or None will
    write no inputs to file.
    :param sequence_class: class to use for the generator sequence
    :param model_name:
    :param verbose:
    :param use_multiprocessing:
    :param n_workers:
    :param max_queue_size:
    :param config:
    :param model_filename:
    :param training_log_filename:
    :param metric_to_monitor:
    :param model_metrics:
    :return:

    Anything that directly affects the training results should go into the config file. Other specifications such as
    multiprocessing optimization should be arguments to this function, as these arguments affect the computation time,
    but the results should not vary based on whether multiprocessing is used or not.
    """
    n_workers = config['num_workers']
    n_gpus = config['n_gpus']
    pin_memory = config['pin_memory']
    print("def run_pytorch_training: n_workers: ", n_workers)
    print("def run_pytorch_training: n_gpus: ", n_gpus)
    print("def run_pytorch_training: pin_memory: ", pin_memory)
    config_device = 'device' in config
    # config_extract = 
    window = np.asarray(config['window'])
    spacing = np.asarray(config['spacing']) if 'spacing' in config else None
    if 'model_name' in config:
        model_name = config['model_name']

    if "n_outputs" in config:
        n_outputs = config['n_outputs']
    else:
        n_outputs = len(np.concatenate(config['metric_names']))

    if "model_kwargs" in config:
        model_kwargs = config["model_kwargs"]
        if "input_shape" not in config["model_kwargs"]:
            # assume that the model will take in the whole image window
            config["model_kwargs"]["input_shape"] = window
    else:
        model_kwargs = dict()
    # [self.sw, self.threshold, self.scale, self.zero_point]
    # if 'tdvd' in config:
    #     model_kwargs['threshold_parameters'] =  

    # coding:utf-8
    # import configparser
    # import os
    # cfgpath = os.path.join(config['init'])
    # conf = configparser.ConfigParser()
    # conf.read(cfgpath,encoding="utf-8")

    from fnmatch import fnmatch                                           
    model = build_or_load_model(model_name, model_filename, n_features=config["n_features"], n_outputs=n_outputs,
                                freeze_bias=in_config("freeze_bias", config, False),
                                bias=bias, n_gpus=n_gpus, device=config_device,**model_kwargs)
    for net,param in model.named_parameters():
        if fnmatch(net, '*conv.weight*'):
            print('{} parameter size: {} M'.format(net, param.view(-1).size()[0]/1000000))


    model.train()

    criterion = load_criterion(config['loss'], n_gpus=n_gpus)

    if "weights" in config and config["weights"] is not None:
        criterion = functions.WeightedLoss(torch.tensor(config["weights"]), criterion)

    optimizer_kwargs = dict()
    if "initial_learning_rate" in config:
        optimizer_kwargs["learning_rate"] = config["initial_learning_rate"]
    
    # for name, param in model.named_parameters(): #查看可优化的参数有哪些
    #     if param.requires_grad:
    #         if fnmatch(name, '*threshold*'):
    #             threshold_params = list(map(id, param))
    #             threshold_param = param
    threshold_params = \
        [
        id(model.module.encoder.layers[0].blocks[0].conv1.threshold),
        id(model.module.encoder.layers[0].blocks[0].conv2.threshold),
        id(model.module.encoder.layers[0].blocks[1].conv1.threshold),
        id(model.module.encoder.layers[0].blocks[1].conv2.threshold),
        id(model.module.encoder.layers[1].blocks[0].conv1.threshold),
        id(model.module.encoder.layers[1].blocks[0].conv2.threshold),
        id(model.module.encoder.layers[1].blocks[1].conv1.threshold),
        id(model.module.encoder.layers[1].blocks[1].conv2.threshold),
        id(model.module.encoder.layers[2].blocks[0].conv1.threshold),
        id(model.module.encoder.layers[2].blocks[0].conv2.threshold),
        id(model.module.encoder.layers[2].blocks[1].conv1.threshold),
        id(model.module.encoder.layers[2].blocks[1].conv2.threshold),
        id(model.module.encoder.layers[3].blocks[0].conv1.threshold),
        id(model.module.encoder.layers[3].blocks[0].conv2.threshold),
        id(model.module.encoder.layers[3].blocks[1].conv1.threshold),
        id(model.module.encoder.layers[3].blocks[1].conv2.threshold),
        id(model.module.encoder.layers[4].blocks[0].conv1.threshold),
        id(model.module.encoder.layers[4].blocks[0].conv2.threshold),
        id(model.module.encoder.layers[4].blocks[1].conv1.threshold),
        id(model.module.encoder.layers[4].blocks[1].conv2.threshold),
        id(model.module.decoder.layers[0].blocks[0].conv1.threshold),
        id(model.module.decoder.layers[0].blocks[0].conv2.threshold),
        id(model.module.decoder.layers[1].blocks[0].conv1.threshold),
        id(model.module.decoder.layers[1].blocks[0].conv2.threshold),
        id(model.module.decoder.layers[2].blocks[0].conv1.threshold),
        id(model.module.decoder.layers[2].blocks[0].conv2.threshold),
        id(model.module.decoder.layers[3].blocks[0].conv1.threshold),
        id(model.module.decoder.layers[3].blocks[0].conv2.threshold),
        id(model.module.decoder.layers[4].blocks[0].conv1.threshold),
        id(model.module.decoder.layers[4].blocks[0].conv2.threshold)
        ]
    threshold_param = \
        [
        model.module.encoder.layers[0].blocks[0].conv1.threshold,
        model.module.encoder.layers[0].blocks[0].conv2.threshold,
        model.module.encoder.layers[0].blocks[1].conv1.threshold,
        model.module.encoder.layers[0].blocks[1].conv2.threshold,
        model.module.encoder.layers[1].blocks[0].conv1.threshold,
        model.module.encoder.layers[1].blocks[0].conv2.threshold,
        model.module.encoder.layers[1].blocks[1].conv1.threshold,
        model.module.encoder.layers[1].blocks[1].conv2.threshold,
        model.module.encoder.layers[2].blocks[0].conv1.threshold,
        model.module.encoder.layers[2].blocks[0].conv2.threshold,
        model.module.encoder.layers[2].blocks[1].conv1.threshold,
        model.module.encoder.layers[2].blocks[1].conv2.threshold,
        model.module.encoder.layers[3].blocks[0].conv1.threshold,
        model.module.encoder.layers[3].blocks[0].conv2.threshold,
        model.module.encoder.layers[3].blocks[1].conv1.threshold,
        model.module.encoder.layers[3].blocks[1].conv2.threshold,
        model.module.encoder.layers[4].blocks[0].conv1.threshold,
        model.module.encoder.layers[4].blocks[0].conv2.threshold,
        model.module.encoder.layers[4].blocks[1].conv1.threshold,
        model.module.encoder.layers[4].blocks[1].conv2.threshold,
        model.module.decoder.layers[0].blocks[0].conv1.threshold,
        model.module.decoder.layers[0].blocks[0].conv2.threshold,
        model.module.decoder.layers[1].blocks[0].conv1.threshold,
        model.module.decoder.layers[1].blocks[0].conv2.threshold,
        model.module.decoder.layers[2].blocks[0].conv1.threshold,
        model.module.decoder.layers[2].blocks[0].conv2.threshold,
        model.module.decoder.layers[3].blocks[0].conv1.threshold,
        model.module.decoder.layers[3].blocks[0].conv2.threshold,
        model.module.decoder.layers[4].blocks[0].conv1.threshold,
        model.module.decoder.layers[4].blocks[0].conv2.threshold
        ]
    base_params = filter(lambda p: id(p) not in threshold_params, model.parameters())

    optimizer = torch.optim.Adam([
    {'params': base_params, 'lr': optimizer_kwargs["learning_rate"]},
    {'params': threshold_param, 'lr': optimizer_kwargs["learning_rate"]*config['threshold_lr_x']}])

    # optimizer = build_optimizer(optimizer_name=config["optimizer"],
    #                             model_parameters=model.parameters(),
    #                             **optimizer_kwargs)

    # sys.exit()
    # distiller code 
    import os
    if os.path.exists("brats_baseline_training_log.csv"):
        os.remove("brats_baseline_training_log.csv")
    print("------- distiller: compression_scheduler config")
    conf_name = (config['prefix_name'] + '_'+config['user'] + '_input_shape'+str(config['model_kwargs']['input_shape'][0]) + '_lambda'+str(config['lambda'])+'_bias'+\
        str(config['bias'])+ '_lr' + str(config['initial_learning_rate']) + '_threshold_lr_x' + str(config['threshold_lr_x']) + '_compress_' + config['compress'].replace('/','_') + '_extract_' + str(config['extract'])).replace('.','_').replace('[','_').replace(']','_').replace(',','_')
    save_dir_root = os.path.join('extract',conf_name)
    resume_epoch = 0
    import glob
    if resume_epoch != 0:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy(config['config_filename'],save_dir)
    copy_tree('../../unet3d',save_dir+'/unet3d')
    compress = config['compress']

    # from torch.utils.data import DataLoader
    # import distiller.apputils.image_classifier as ic
    import distiller
    # from distiller.quantization.range_linear import PostTrainLinearQuantizer
    # from distiller.quantization.range_linear import RangeLinearQuantWrapper
    import distiller.apputils as apputils
    # from distiller.apputils.image_classifier import test

    msglogger = apputils.config_pylogger('logging.conf',experiment_name='distiller',output_dir=save_dir)
    tflogger = TensorBoardLogger(msglogger.logdir)
    tflogger.log_gradients = True
    pylogger = PythonLogger(msglogger)

    compression_scheduler = None
    if config_device:
        device = torch.device(config['device'])
        print("Device being used:", device)
        model.to(device)
    else:
        device = None
        # criterion.to(device)
    if 'resumed_checkpoint_path' in config:
        checkpoint = torch.load(os.path.join(config['resumed_checkpoint_path']),map_location=lambda storage, loc: storage)
        resume_epoch = checkpoint['epoch'] + 1
        # optimizer.load_state_dict(checkpoint['opt_dict'])
    if compress:
        # print("------model: ", model)
        shutil.copy(compress,save_dir)
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
        # sys.exit()
        compression_scheduler=distiller.file_config(model,optimizer=optimizer,filename=compress,resumed_epoch=resume_epoch if 'resumed_checkpoint_path' in config else None)
        compression_scheduler.append_float_weight_after_quantizer()
    if 'resumed_checkpoint_path' in config:
        model.load_state_dict(checkpoint['state_dict'])
        if not config['resumed_reset_optim' ]:
            optimizer.load_state_dict(checkpoint['opt_dict'])
    print("optimizer: {}".format(optimizer))
    sequence_kwargs = in_config("sequence_kwargs", config, dict())

    if "flatten_y" in config and config["flatten_y"]:
        collate_fn = collate_flatten
    elif "collate_fn" in config and config["collate_fn"] == "collate_5d_flatten":
        collate_fn = collate_5d_flatten
    else:
        from torch.utils.data.dataloader import default_collate
        collate_fn = default_collate

    # 4. Create datasets
    training_dataset = sequence_class(filenames=config['training_filenames'],
                                      flip=in_config('flip', config, False),
                                      reorder=config['reorder'],
                                      window=window,
                                      spacing=spacing,
                                      points_per_subject=in_config('points_per_subject', config, 1),
                                      surface_names=in_config('surface_names', config, None),
                                      metric_names=in_config('metric_names', config, None),
                                      base_directory=directory,
                                      subject_ids=config["training"],
                                      iterations_per_epoch=in_config("iterations_per_epoch", config, 1),
                                      **in_config("additional_training_args", config, dict()),
                                      **sequence_kwargs)
    print("****training_loader config[batch_size]: ", config["batch_size"])
    training_loader = DataLoader(training_dataset,
                                 batch_size=config["batch_size"] // in_config('points_per_subject', config, 1),
                                 shuffle=True,
                                 num_workers=n_workers,
                                 collate_fn=collate_fn,
                                 pin_memory=pin_memory)
    with open('training_filenames.txt','w') as f:
        for i in range(len(config['training_filenames'])):
            f.writelines(str(config['training_filenames'][i]))
    # print('------------',len(config['training_filenames']))
    if test_input:
        print("test input: ", test_input)
        # for index in range(test_input):
        #     x, y = training_dataset[index]
        #     if not isinstance(x, np.ndarray):
        #         x = x.numpy()
        #         y = y.numpy()
        #     x = np.moveaxis(x, 0, -1)
        #     x_image = nib.Nifti1Image(x.squeeze(), affine=np.diag(np.ones(4)))
        #     x_image.to_filename(model_filename.replace(".h5",
        #                                                "_input_test_{}.nii.gz".format(index)))
        #     if len(y.shape) >= 3:
        #         y = np.moveaxis(y, 0, -1)
        #         y_image = nib.Nifti1Image(y.squeeze(), affine=np.diag(np.ones(4)))
        #         y_image.to_filename(model_filename.replace(".h5",
        #                                                    "_target_test_{}.nii.gz".format(index)))

    if 'skip_validation' in config and config['skip_validation']:
        validation_loader = None
        metric_to_monitor = "loss"
    else:

        with open('validation_filenames.txt','w') as f:
            for i in range(len(config['validation_filenames'])):
                f.writelines(str(config['validation_filenames'][i]))

        validation_dataset = sequence_class(filenames=config['validation_filenames'],
                                            flip=False,
                                            reorder=config['reorder'],
                                            window=window,
                                            spacing=spacing,
                                            points_per_subject=in_config('validation_points_per_subject', config, 1),
                                            surface_names=in_config('surface_names', config, None),
                                            metric_names=in_config('metric_names', config, None),
                                            **sequence_kwargs,
                                            **in_config("additional_validation_args", config, dict()))
        validation_loader = DataLoader(validation_dataset,
                                       batch_size=config["validation_batch_size"] // in_config("points_per_subject",
                                                                                               config, 1),
                                       shuffle=False,
                                       num_workers=n_workers,
                                       collate_fn=collate_fn,
                                       pin_memory=pin_memory)
    try:
        train(model=model, optimizer=optimizer, criterion=criterion, n_epochs=config["n_epochs"], verbose=bool(verbose),
          training_loader=training_loader, validation_loader=validation_loader, model_filename=model_filename,
          training_log_filename=training_log_filename,
          metric_to_monitor=metric_to_monitor,
          early_stopping_patience=in_config("early_stopping_patience", config),
          config = config,
          save_best=in_config("save_best", config, False),
          learning_rate_decay_patience=in_config("decay_patience", config),
          regularized=in_config("regularized", config, regularized),
          n_gpus=n_gpus,
          vae=in_config("vae", config, False),
          decay_factor=in_config("decay_factor", config),
          min_lr=in_config("min_learning_rate", config),
          learning_rate_decay_step_size=in_config("decay_step_size", config),
          save_every_n_epochs=in_config("save_every_n_epochs", config),
          save_last_n_models=in_config("save_last_n_models", config),
          amp=amp, compression_scheduler=compression_scheduler, tflogger=tflogger,pylogger=pylogger,save_dir=save_dir,extract=config['extract'], device=device, start_epoch=resume_epoch)
    except KeyboardInterrupt:
        msglogger.info('-' * 89)
        msglogger.info('KeyboardInterrupt -> Exiting from train_model early')#

def train(model, optimizer, criterion, n_epochs, training_loader, validation_loader, training_log_filename,
          model_filename, compression_scheduler,tflogger,pylogger,config,metric_to_monitor="val_loss", early_stopping_patience=None,
          learning_rate_decay_patience=None, save_best=False, n_gpus=1, verbose=True, regularized=False,
          vae=False, decay_factor=0.1, min_lr=0., learning_rate_decay_step_size=None, save_every_n_epochs=None,
          save_last_n_models=None, amp=False, save_dir='.', extract=None, device=None, start_epoch=0):
    training_log = list()
    if os.path.exists(training_log_filename):
        training_log.extend(pd.read_csv(training_log_filename).values)
        # start_epoch = int(training_log[-1][0]) + 1
    # else:
    #     start_epoch = 0
    training_log_header = ["epoch", "loss", "lr", "val_loss"]

    if learning_rate_decay_patience:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=learning_rate_decay_patience,
                                                               verbose=verbose, factor=decay_factor, min_lr=min_lr, threshold=0.001)
    elif learning_rate_decay_step_size:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=learning_rate_decay_step_size,
                                                    gamma=decay_factor, last_epoch=-1)
        # Setting the last epoch to anything other than -1 requires the optimizer that was previously used.
        # Since I don't save the optimizer, I have to manually step the scheduler the number of epochs that have already
        # been completed. Stepping the scheduler before the optimizer raises a warning, so I have added the below
        # code to step the scheduler and catch the UserWarning that would normally be thrown.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(start_epoch):
                scheduler.step()
    elif OneCycleLR:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 16e-3, epochs=n_epochs-start_epoch, steps_per_epoch=len(train_loader),\
            div_factor = 10, final_div_factor=100)
    else:
        scheduler = None

    if amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    val_loss_old = 1000000
    epoch_old = 0
    for epoch in range(start_epoch, n_epochs):
        # print("save_last_n_models", save_last_n_models)
        # early stopping
        # if (training_log and early_stopping_patience
        #     and np.asarray(training_log)[:, training_log_header.index(metric_to_monitor)].argmin()
        #         <= len(training_log) - early_stopping_patience):
        #     print("Early stopping patience {} has been reached.".format(early_stopping_patience))
        #     break
        # hook scale_threshold
        print('***** before assignment: value of conv1.scale: {}, conv1.relu.fake_q.scale: {}'\
            .format(model.module.encoder.layers[0].blocks[0].conv1.scale, model.module.encoder.layers[0].blocks[0].conv1.relu.fake_q.scale.item()))



        model.module.encoder.layers[0].blocks[0].conv1.conv_name =  "model.module.encoder.layers[0].blocks[0].conv1"
        model.module.encoder.layers[0].blocks[0].conv2.conv_name =  "model.module.encoder.layers[0].blocks[0].conv2"
        model.module.encoder.layers[0].blocks[1].conv1.conv_name =  "model.module.encoder.layers[0].blocks[1].conv1"
        model.module.encoder.layers[0].blocks[1].conv2.conv_name =  "model.module.encoder.layers[0].blocks[1].conv2"
        model.module.encoder.layers[1].blocks[0].conv1.conv_name =  "model.module.encoder.layers[1].blocks[0].conv1"
        model.module.encoder.layers[1].blocks[0].conv2.conv_name =  "model.module.encoder.layers[1].blocks[0].conv2"
        model.module.encoder.layers[1].blocks[1].conv1.conv_name =  "model.module.encoder.layers[1].blocks[1].conv1"
        model.module.encoder.layers[1].blocks[1].conv2.conv_name =  "model.module.encoder.layers[1].blocks[1].conv2"
        model.module.encoder.layers[2].blocks[0].conv1.conv_name =  "model.module.encoder.layers[2].blocks[0].conv1"
        model.module.encoder.layers[2].blocks[0].conv2.conv_name =  "model.module.encoder.layers[2].blocks[0].conv2"
        model.module.encoder.layers[2].blocks[1].conv1.conv_name =  "model.module.encoder.layers[2].blocks[1].conv1"
        model.module.encoder.layers[2].blocks[1].conv2.conv_name =  "model.module.encoder.layers[2].blocks[1].conv2"
        model.module.encoder.layers[3].blocks[0].conv1.conv_name =  "model.module.encoder.layers[3].blocks[0].conv1"
        model.module.encoder.layers[3].blocks[0].conv2.conv_name =  "model.module.encoder.layers[3].blocks[0].conv2"
        model.module.encoder.layers[3].blocks[1].conv1.conv_name =  "model.module.encoder.layers[3].blocks[1].conv1"
        model.module.encoder.layers[3].blocks[1].conv2.conv_name =  "model.module.encoder.layers[3].blocks[1].conv2"
        model.module.encoder.layers[4].blocks[0].conv1.conv_name =  "model.module.encoder.layers[4].blocks[0].conv1"
        model.module.encoder.layers[4].blocks[0].conv2.conv_name =  "model.module.encoder.layers[4].blocks[0].conv2"
        model.module.encoder.layers[4].blocks[1].conv1.conv_name =  "model.module.encoder.layers[4].blocks[1].conv1"
        model.module.encoder.layers[4].blocks[1].conv2.conv_name =  "model.module.encoder.layers[4].blocks[1].conv2"
        model.module.decoder.layers[0].blocks[0].conv1.conv_name =  "model.module.decoder.layers[0].blocks[0].conv1"
        model.module.decoder.layers[0].blocks[0].conv2.conv_name =  "model.module.decoder.layers[0].blocks[0].conv2"
        model.module.decoder.layers[1].blocks[0].conv1.conv_name =  "model.module.decoder.layers[1].blocks[0].conv1"
        model.module.decoder.layers[1].blocks[0].conv2.conv_name =  "model.module.decoder.layers[1].blocks[0].conv2"
        model.module.decoder.layers[2].blocks[0].conv1.conv_name =  "model.module.decoder.layers[2].blocks[0].conv1"
        model.module.decoder.layers[2].blocks[0].conv2.conv_name =  "model.module.decoder.layers[2].blocks[0].conv2"
        model.module.decoder.layers[3].blocks[0].conv1.conv_name =  "model.module.decoder.layers[3].blocks[0].conv1"
        model.module.decoder.layers[3].blocks[0].conv2.conv_name =  "model.module.decoder.layers[3].blocks[0].conv2"
        model.module.decoder.layers[4].blocks[0].conv1.conv_name =  "model.module.decoder.layers[4].blocks[0].conv1"
        model.module.decoder.layers[4].blocks[0].conv2.conv_name =  "model.module.decoder.layers[4].blocks[0].conv2"



        print('***** after assignment: value of conv1.scale: {}, conv1.relu.fake_q.scale: {}'\
            .format(model.module.encoder.layers[0].blocks[0].conv1.scale, model.module.encoder.layers[0].blocks[0].conv1.relu.fake_q.scale.item()))
        # print('***** scale_threshold: {}'.format(scale_threshold))
        # print('***** zero_point_threshold: {}'.format(zero_point_threshold))
        # print('model.module.inputs_quant.scale.item(): ', model.module.inputs_quant.scale.item())
        # print('model.module.encoder.layers[4].blocks[0].conv1.relu.fake_q.scale.item()', model.module.encoder.layers[4].blocks[0].conv1.relu.fake_q.scale.item())
        # train the model
        # print("def train: n gpus:", n_gpus)
        # print("------- distiller: on_epoch_begin")
        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch)
        loss = epoch_training(training_loader, model, criterion, config=config,optimizer=optimizer, epoch=epoch, n_gpus=n_gpus,
                              regularized=regularized, vae=vae, scaler=scaler, compression_scheduler=compression_scheduler, save_dir=save_dir,extract=extract, device=device)
        # print("------- distiller: log_weights_sparsity")
        if compression_scheduler:
            import distiller
            distiller.log_weights_sparsity(model,epoch,loggers=[tflogger,pylogger])
        # print("------- distiller: on_epoch_end")
        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch)
        try:
            training_loader.dataset.on_epoch_end()
        except AttributeError:
            warnings.warn("'on_epoch_end' method not implemented for the {} dataset.".format(
                type(training_loader.dataset)))
        # predict validation data
        validation_loader = validation_loader if epoch >= config['epoch_start_valtest'] else None
        if validation_loader:
            val_loss = epoch_validatation(validation_loader, model, criterion, config=config,optimizer=optimizer, n_gpus=n_gpus, regularized=regularized,
                                          vae=vae, use_amp=scaler is not None, epoch=epoch, save_dir=save_dir)
        else:
            val_loss = None


        # update the training log
        training_log.append([epoch, loss, get_lr(optimizer), val_loss])
        # pd.DataFrame(training_log, columns=training_log_header).set_index("epoch").to_csv(training_log_filename)
        # min_epoch = np.asarray(training_log)[:, training_log_header.index(metric_to_monitor)].argmin()

        # check loss and decay
        if scheduler:
            lr0_old = optimizer.state_dict()['param_groups'][0]['lr']
            lr1_old = optimizer.state_dict()['param_groups'][1]['lr']

            if validation_loader and scheduler.__class__ == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(val_loss)
            elif scheduler.__class__ == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(loss)
            else:
                scheduler.step()

            if optimizer.state_dict()['param_groups'][0]['lr'] != lr0_old:
                optimizer.state_dict()['param_groups'][1]['lr'] = lr1_old * config['threshold_lr_factor']

        # save model

        # if save_best and min_epoch == len(training_log) - 1:
        #     # best_filename = model_filename.replace(".h5", "_best.h5")
        #     forced_copy(model_filename, best_filename)

        # if save_every_n_epochs and (epoch % save_every_n_epochs) == 0:
        #     epoch_filename = model_filename.replace(".h5", "_{}.h5".format(epoch))
        #     forced_copy(model_filename, epoch_filename)

        # if save_last_n_models is not None and save_last_n_models > 1:
        #     if not save_every_n_epochs or not ((epoch - save_last_n_models) % save_every_n_epochs) == 0:
        #         to_delete = model_filename.replace(".h5", "_{}.h5".format(epoch - save_last_n_models))
        #         remove_file(to_delete)
        #     epoch_filename = model_filename.replace(".h5", "_{}.h5".format(epoch))
        #     forced_copy(model_filename, epoch_filename)
        if os.path.exists(os.path.join(save_dir, 'models'))==False:
            os.makedirs(os.path.join(save_dir, 'models')) 
        if epoch % save_every_n_epochs == 0: # 0, save_epoch

            # torch.save({
            #     'epoch': epoch + 1,
            #     'state_dict': model.state_dict(),
            #     'opt_dict': optimizer.state_dict(),
            # }, os.path.join(save_dir, 'models', '_epoch-' + str(epoch) + '.pth.tar'))

            # # write .txt
            with open (os.path.join(save_dir, 'models', '_epoch-' + str(epoch) + '.pth.tar.txt'), 'w') as state_file:
                for k in model.state_dict():
                    state_file.writelines(k)
                    state_file.writelines(str(model.state_dict()[k]))
        val_loss = val_loss if validation_loader else 0
        if val_loss  <= val_loss_old:
            remove_file(os.path.join(save_dir, 'models', '_epoch-' + str(epoch_old) + '_best.pth.tar'))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', '_epoch-' + str(epoch) + '_best.pth.tar'), _use_new_zipfile_serialization=False)            
            val_loss_old = val_loss
            epoch_old = epoch
        # save last epoch
        remove_file(os.path.join(save_dir, 'models', '_epoch-' + str(epoch-1) + '_last.pth.tar'))
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'opt_dict': optimizer.state_dict(),
        }, os.path.join(save_dir, 'models', '_epoch-' + str(epoch) + '_last.pth.tar'), _use_new_zipfile_serialization=False)

def forced_copy(source, target):
    remove_file(target)
    shutil.copy(source, target)


def remove_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def get_lr(optimizer):
    lrs = [params['lr'] for params in optimizer.param_groups]
    return np.squeeze(np.unique(lrs))


def load_criterion(criterion_name, n_gpus=0):
    try:
        criterion = getattr(functions, criterion_name)
    except AttributeError:
        criterion = getattr(torch.nn, criterion_name)()
        if n_gpus > 0:
            criterion.cuda()
    return criterion
