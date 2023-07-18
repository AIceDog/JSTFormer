import os
import glob
import torch
import sys
import random
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from common.opt import opts
from common.utils import *
from common.load_data_3dhp import Fusion
from model.model import Model

import scipy.io as scio

opt = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)


def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model)


def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    if split == 'train':
        model.train()
    else:
        model.eval()

    loss_all = {'loss': AccumLoss()}
    error_sum = AccumLoss()
    error_sum_test = AccumLoss()
    loss_weight_base = 0.5
    loss_weight_refine = 0.5

    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]

    data_inference = {}

    for i, data in enumerate(tqdm(dataLoader, 0)):
        if split == "train":
            batch_cam, gt_3D, input_2D, seq, subject, scale, bb_box, cam_ind = data
        else:
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = data

        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box])

        N = input_2D.size(0)

        out_target = gt_3D.clone().view(N, -1, opt.out_joints, opt.out_channels)
        out_target[:, :, 14] = 0

        if out_target.size(1) > 1:
            out_target_single = out_target[:, opt.pad].unsqueeze(1)
        else:
            out_target_single = out_target

        if opt.test_augmentation and split == 'test':
            input_2D, output_3D_VTE, output_3D = input_augmentation(input_2D, model, joints_left, joints_right, opt)
        else:
            output_3D_VTE, output_3D = model(input_2D, opt)
        
        
        output_3D_single = output_3D
        
        if split == 'train':
            pred_out = output_3D_VTE
        elif split == 'test':
            pred_out = output_3D_single
        
        loss = mpjpe_cal(pred_out, out_target) * loss_weight_base + mpjpe_cal(output_3D_single, out_target_single) * loss_weight_refine

        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_out[:, :, 14, :] = 0
            joint_error = mpjpe_cal(pred_out, out_target).item()

            error_sum.update(joint_error * N, N)

        elif split == 'test':
            pred_out[:, :, 14, :] = 0
            joint_error_test = mpjpe_cal(pred_out, out_target).item()
            out = pred_out

            if opt.train == 0:
                for seq_cnt in range(len(seq)):
                    seq_name = seq[seq_cnt]
                    if seq_name in data_inference:
                        data_inference[seq_name] = np.concatenate((data_inference[seq_name], out[seq_cnt].permute(2, 1, 0).cpu().numpy()), axis=2)
                    else:
                        data_inference[seq_name] = out[seq_cnt].permute(2, 1, 0).cpu().numpy()

            error_sum_test.update(joint_error_test * N, N)

            
    if split == 'train':
        return loss_all['loss'].avg, error_sum.avg
    elif split == 'test':
        if opt.train == 0:
            for seq_name in data_inference.keys():
                data_inference[seq_name] = data_inference[seq_name][:, :, None, :]
            mat_path = os.path.join(opt.checkpoint, 'inference_data.mat')
            scio.savemat(mat_path, data_inference)

        return error_sum_test.avg


def input_augmentation(input_2D, model, joints_left, joints_right, opt):   
    input_2D_flip = input_2D[:, 1]
    input_2D_non_flip = input_2D[:, 0]

    output_3D_flip_VTE, output_3D_flip = model(input_2D_flip, opt)

    output_3D_flip_VTE[:, 0] *= -1
    output_3D_flip[:, 0] *= -1
    
    output_3D_flip_VTE[:, :, joints_left + joints_right, :] = output_3D_flip_VTE[:, :, joints_right + joints_left, :]
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D_non_flip_VTE, output_3D_non_flip = model(input_2D_non_flip, opt)

    output_3D_VTE = (output_3D_non_flip_VTE + output_3D_flip_VTE) / 2
    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_3D_VTE, output_3D


if __name__ == '__main__':
    print("opt : ", opt)

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    print('opt.checkpoint : ', opt.checkpoint)
    # opt.train = 0
    
    if opt.train == 1:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)

    root_path = opt.root_path
    actions = define_actions(opt.actions)

    if opt.train:
        train_data = Fusion(opt=opt, train=True, root_path=root_path)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)

    test_data = Fusion(opt=opt, train=False, root_path=root_path)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=int(opt.workers), pin_memory=True)

    opt.out_joints = 17

    model = Model(opt).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4])
    
    
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    epoch_start = 0  
    lr = opt.lr
    all_param = []
    all_param += list(model.parameters())
    optimizer = optim.AdamW(all_param, lr=opt.lr, weight_decay=0.1)  

    for epoch in range(0, opt.nepoch):
        if opt.train == 1:
            loss, mpjpe = train(opt, actions, train_dataloader, model, optimizer, epoch)

        p1 = val(opt, actions, test_dataloader, model)
        data_threshold = p1

        if opt.train and data_threshold < opt.previous_best_threshold:
            opt.previous_name = save_model_3dhp(opt.previous_name, opt.checkpoint, epoch, data_threshold, model, 'model')

            opt.previous_best_threshold = data_threshold

        if opt.train == 0:
            print('p1: %.2f' % (p1))
            break
        else:
            logging.info(
                'epoch: %d, lr: %.7f, loss: %.4f, MPJPE: %.2f, p1: %.2f' % (epoch, lr, loss, mpjpe, p1))
            print('e: %d, lr: %.7f, loss: %.4f, M: %.2f, p1: %.2f' % (epoch, lr, loss, mpjpe, p1))

        lr *= opt.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= opt.lr_decay







