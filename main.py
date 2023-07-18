import os
import glob
import torch
import random
import sys
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from common.opt import opts
from common.utils import *
from common.load_data_hm36 import Fusion
from common.h36m_dataset import Human36mDataset
from model.model import Model

from torch.cuda import amp # auto mix precision
from torch.autograd import Variable # gradient checkpoint
from torch.utils.checkpoint import checkpoint # gradient checkpoint

opt = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)


def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model)


def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    loss_all = {'loss': AccumLoss()}
    action_error_sum = define_error_list(actions)
    loss_weight_base = 0.5
    loss_weight_refine = 0.5
    
    if split == 'train':
        model.train()
    else:
        model.eval()

    for i, data in enumerate(tqdm(dataLoader, 0)):
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box])
        
        if split == 'train':      
            output_3D, output_3D_refine = model(input_2D, opt)
        else:
            input_2D, output_3D, output_3D_refine = input_augmentation(input_2D, model, opt)

        out_target = gt_3D.clone() # when split == 'train' gt_3D.shape : [B, F, J, 3]; when split == 'test' gt_3D.shape : [B, 1, J, 3]
        out_target[:, :, 0] = 0 # out_target.shape : [B, F, J, 3]
            
        if out_target.size(1) > 1:
            out_target_single = out_target[:, opt.pad].unsqueeze(1) # out_target_single.shape : [B, 1, J, 3]
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1) 
        else:
            out_target_single = out_target
            gt_3D_single = gt_3D
        
        if split == 'train':
            loss = mpjpe_cal(output_3D, out_target) * loss_weight_base + mpjpe_cal(output_3D_refine, out_target_single) * loss_weight_refine
            
            N = input_2D.size(0)
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif split == 'test':
            output_3D = output_3D[:, opt.pad].unsqueeze(1)
            output_3D[:, :, 0, :] = 0 
            action_error_sum = test_calculation(output_3D, out_target, action, action_error_sum, opt.dataset, subject) 
            action_error_sum_refine = test_calculation(output_3D_refine, out_target, action, action_error_sum, opt.dataset, subject) 

    if split == 'train':
        return loss_all['loss'].avg
    elif split == 'test':
        p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)
        p1_refine, p2_refine = print_error(opt.dataset, action_error_sum_refine, opt.train)

        return p1, p2, p1_refine, p2_refine


def input_augmentation(input_2D, model, opt):
    if opt.dataset == 'h36m':
        joints_left = [8, 9, 10, 11, 12, 13] 
        joints_right = [14, 15, 16, 5, 6, 7] 

    input_2D_non_flip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]
    
    output_3D_non_flip_base, output_3D_non_flip_refine = model(input_2D_non_flip, opt)
    output_3D_flip_base, output_3D_flip_refine         = model(input_2D_flip, opt)

    output_3D_flip_base[:, :, :, 0] *= -1
    output_3D_flip_refine[:, :, :, 0] *= -1 

    output_3D_flip_base[:, :, joints_left + joints_right, :] = output_3D_flip_base[:, :, joints_right + joints_left, :] 
    output_3D_flip_refine[:, :, joints_left + joints_right, :] = output_3D_flip_refine[:, :, joints_right + joints_left, :]

    output_3D_base = (output_3D_non_flip_base + output_3D_flip_base) / 2
    output_3D_refine = (output_3D_non_flip_refine + output_3D_flip_refine) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_3D_base, output_3D_refine



if __name__ == '__main__': 
    
    # opt.train = 0
    # opt.test = True
    # opt.previous_dir = 'checkpoint/0422_1420_41_81'
    torch.autograd.set_detect_anomaly(True)
    print("opt : ", opt)

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.train:
        if opt.resume != '':
            logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO,
                               filemode='a')
        else:
            logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
    
    root_path = opt.root_path
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'

    dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(opt.actions)

    if opt.train:
        train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)

    test_data = Fusion(opt=opt, train=False, dataset=dataset, root_path =root_path)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=int(opt.workers), pin_memory=True)

    model = Model(opt).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4])
  

    epoch_start = 0 
    lr = opt.lr 
    all_param = []
    all_param += list(model.parameters())
    optimizer = optim.AdamW(all_param, lr=opt.lr, weight_decay=0.1) 

    
    if opt.resume != '':  # for resuming 
        model_paths = sorted(glob.glob(os.path.join('checkpoint', opt.resume, '_______.pth'))) # model_paths : checkpoint/0328_0252_48_243/_______.pth

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer']) 
        epoch_start = checkpoint['epoch'] + 1
        lr = checkpoint['lr']
        train_generator_random_state = checkpoint['random_state']
        opt.previous_best_threshold = checkpoint['previous_best_threshold']
        opt.previous_name = checkpoint['previous_name']
        
        lr *= opt.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= opt.lr_decay
        
        print('resuming : ')
        print('epoch_start : ', epoch_start)
        print('lr : ', lr)
        print('train_generator_random_state : ', train_generator_random_state)
        print('opt.previous_best_threshold : ', opt.previous_best_threshold)
        print('opt.previous_name : ', opt.previous_name)
        
        
    if opt.previous_dir != '':  # for testing 
        model_paths = sorted(glob.glob(os.path.join(opt.previous_dir, '*.pth')))
        
        for path in model_paths:
            if path.split('/')[-1].startswith('model'):
                model_path = path
                print('model_path : ', model_path)
        
        pre_dict = torch.load(model_path)
        pre_dict = pre_dict['model']
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

        
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('\n', 'INFO: Trainable parameter count:', model_params, '\n')
    logging.info('INFO: Trainable parameter count: %d' % (model_params))
    
    for epoch in range(epoch_start, opt.nepoch):
        if opt.train: 
            loss = train(opt, actions, train_dataloader, model, optimizer, epoch)
        
        p1, p2, p1_refine, p2_refine = val(opt, actions, test_dataloader, model)

        if opt.train == 0:
            print('p1: %.2f, p2: %.2f' % (p1, p2))
            print('p1_refine: %.2f, p2_refine: %.2f' % (p1_refine, p2_refine)) 
            break
        else:
            logging.info('epoch: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, p1, p2))
            logging.info('p1_refine: %.2f, p2_refine: %.2f' % (p1_refine, p2_refine))
            print('e: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, p1, p2))
            print('p1_refine: %.2f, p2_refine: %.2f' % (p1_refine, p2_refine))
        
        if opt.train: 
            generator_random_state = train_data.get_generator_random_state() 
            save_model_epoch(opt.checkpoint, #  opt.checkpoint : checkpoint/0328_0252_48_243
                             epoch, # this current epoch
                             model, 
                             lr=lr, 
                             train_generator_random_state=generator_random_state, 
                             optimizer=optimizer, # 我加的 optimizer of this epoch
                             previous_best_threshold = opt.previous_best_threshold, 
                             previous_name = opt.previous_name 
                            ) 
 
            if p1 < opt.previous_best_threshold:
                generator_random_state = train_data.get_generator_random_state() 
                opt.previous_name = save_model(opt.previous_name, 
                                               opt.checkpoint, 
                                               epoch, # this current epoch 
                                               p1, # best p1 
                                               model, 
                                               lr=lr,
                                               train_generator_random_state=generator_random_state, 
                                               optimizer=optimizer) 
                opt.previous_best_threshold = p1
                
        lr *= opt.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= opt.lr_decay








