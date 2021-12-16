"""Applied temporal transformation prediction.

For training in drift - python train.py --cl 24 --log drift_log/4_classes/24 --bs 2 --gpu 0 --img_hgt 224 --img_width 224 --dataset DriftDataset --epochs 1000 --train_root_dir drift_dataset/in/training --cal_root_dir drift_dataset/in/calibration/ --lr 0.00001
For testing - python train.py --bs 2 --mode test --ckpt cl16_mod_enc/r3d_cl16_11161201/model_300.pt --root_dir CARLA_dataset/Vanderbilt_data/testing
For testing in drift - python train.py --bs 2 --mode test --ckpt drift_log/r3d_cl24_11302229/model_400.pt --test_root_dir drift_dataset/out --dataset DriftDataset --cl 24


******************** FINAL Training ****************************
Drift - python train.py --cl 16 --log drift_log/4_classes/16/new_in_data/ --bs 2 --gpu 0 --img_hgt 224 --img_width 224 --dataset DriftDataset --epochs 2000 --train_root_dir drift_dataset/temp_in/training --cal_root_dir drift_dataset/temp_in/calibration/ --lr 0.00001

Crowd - python train.py --cl 16 --log crowd_log/16 --bs 2 --gpu 1 --img_hgt 224 --img_width 224 --dataset CrowdDataset --epochs 3000 --train_root_dir moving_crowd_dataset/in/training/ --cal_root_dir moving_crowd_dataset/in/calibration/ --lr 0.00001

"""
import os
import math
import itertools
import argparse
import time
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
from tensorboardX import SummaryWriter

#from datasets.ucf101 import UCF101VCOPDataset
# from models.c3d import C3D
# from models.r3d import R3DNet
# from models.r21d import R2Plus1DNet

from r3d import Regressor as r3d_regressor

from dataset.drift import DriftDataset
from dataset.crowd import CrowdDataset

import pdb


def train(args, model, criterion, optimizer, device, train_dataloader, writer, epoch):
    torch.set_grad_enabled(True)
    model.train()

    total_loss = 0.0
    correct = 0
    for i, data in enumerate(train_dataloader, 1):
        # get inputs
        orig_tuple_clips, transformed_tuple_clips, tranformation = data
        # print(torch.mean(torch.abs(orig_tuple_clips)), torch.mean(torch.abs(transformed_tuple_clips)))
        orig_tuple_clips = orig_tuple_clips.to(device)
        transformed_tuple_clips = transformed_tuple_clips.to(device)
        target_tranformations = torch.tensor(tranformation).to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward and backward
        outputs = model(orig_tuple_clips, transformed_tuple_clips) # return logits here
        # print("Outputs: {}, targets: {}".format(outputs, target_tranformations))
        loss = criterion(outputs, target_tranformations)
        loss.backward()
        optimizer.step()
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(target_tranformations == pts).item()
        # print statistics and write summary every N batch
        # if i % args.pf == 0:
    avg_loss = total_loss / len(train_dataloader)
    avg_acc = correct / len(train_dataloader.dataset)
    print('[TRAIN] epoch-{}, batch-{}, loss: {:.3f}, acc: {:.3f}'.format(epoch, i, avg_loss, avg_acc))
    step = (epoch-1)*len(train_dataloader) + i
    writer.add_scalar('train/CrossEntropyLoss', avg_loss, step)
    writer.add_scalar('train/Accuracy', avg_acc, step)
    # running_loss = 0.0
    # correct = 0
    # summary params and grads per epoch
    # for name, param in model.named_parameters():
    #     writer.add_histogram('params/{}'.format(name), param, epoch)
    #     writer.add_histogram('grads/{}'.format(name), param.grad, epoch)


def validate(args, model, criterion, device, val_dataloader, writer, epoch):
    torch.set_grad_enabled(False)
    model.eval()
    
    total_loss = 0.0
    correct = 0
    for i, data in enumerate(val_dataloader, 1):
        # get inputs
        orig_tuple_clips, transformed_tuple_clips, tranformation = data
        orig_tuple_clips = orig_tuple_clips.to(device)
        transformed_tuple_clips = transformed_tuple_clips.to(device)
        target_tranformations = torch.tensor(tranformation).to(device)
        # forward
        outputs = model(orig_tuple_clips, transformed_tuple_clips) # return logits here
        loss = criterion(outputs, target_tranformations)
        #print("Outputs: {}, targets: {}".format(outputs, target_tranformations))
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(target_tranformations == pts).item()
        # print('correct: {}, {}, {}'.format(correct, target_tranformations, pts))
    avg_loss = total_loss / len(val_dataloader)
    avg_acc = correct / len(val_dataloader.dataset)
    writer.add_scalar('val/CrossEntropyLoss', avg_loss, epoch)
    writer.add_scalar('val/Accuracy', avg_acc, epoch)
    print('[VAL] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss


def test(args, model, criterion, device, test_dataloader):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    for i, data in enumerate(test_dataloader, 1):
        # get inputs
        orig_tuple_clips, transformed_tuple_clips, tranformation = data
        orig_tuple_clips = orig_tuple_clips.to(device)
        transformed_tuple_clips = transformed_tuple_clips.to(device)
        target_tranformations = torch.tensor(tranformation).to(device)
        # forward
        outputs = model(orig_tuple_clips, transformed_tuple_clips)
        print("output: {}, target: {}".format(outputs, target_tranformations))
        loss = criterion(outputs, target_tranformations)
        # compute loss and acc
        total_loss += loss.item()
        print("Loss: ", loss.item())
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(target_tranformations == pts).item()
        # print('correct: {}, {}, {}'.format(correct, target_tranformations, pts))
    avg_loss = total_loss / len(test_dataloader)
    avg_acc = correct / len(test_dataloader.dataset)
    print('[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Video Clip Order Prediction')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--model', type=str, default='r3d', help='c3d/r3d/r21d')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--wgtDecay', default=5e-4, type=float)
    parser.add_argument('--lrMul', type=float, default=10.)
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--log', type=str, help='log directory')
    parser.add_argument('--ckpt', type=str, default='', help='checkpoint path')
    parser.add_argument('--desp', type=str, help='additional description')
    parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=2, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')
    parser.add_argument('--seed', type=int, default=100, help='seed for initializing training.')
    parser.add_argument('--train_root_dir', type=str, default='moving_crowd_dataset/in/training',help='training data directory')
    parser.add_argument('--cal_root_dir', type=str, default='moving_crowd_dataset/in/calibration',help='calibration data directory')
    parser.add_argument('--test_root_dir', type=str, default='moving_crowd_dataset/out',help='test data directory')
    parser.add_argument('--img_hgt', type=int, default=224, help='img height')
    parser.add_argument('--img_width', type=int, default=224, help='img width')
    parser.add_argument('--dataset', default='CARLAVCOPDataset', help='dataset - CARLAVCOPDataset/DriftDataset/CrowdDataset')

    args = parser.parse_args()
    return args

dataset_class = {'DriftDataset': DriftDataset, 'CrowdDataset': CrowdDataset}

if __name__ == '__main__':
    args = parse_args()
    print(vars(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu) if use_cuda else "cpu")

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.gpu:
            torch.cuda.manual_seed_all(args.seed)

    ########### model r3d for now ##############
    # if args.model == 'c3d':
    #     base = C3D(with_classifier=False)
    # elif args.model == 'r3d':
    net = r3d_regressor().to(device)
    # elif args.model == 'r21d':   
        # base = R2Plus1DNet(layer_sizes=(1,1,1,1), with_classifier=False)
    # if args.cuda: ASSUMING CUDA is true always
    net = torch.nn.DataParallel(net, device_ids=[args.gpu])

    if args.ckpt != '':
        net.load_state_dict(torch.load(args.ckpt))

    if args.mode == 'train':  ########### Train #############
        if args.ckpt:  # resume training
            net.load_state_dict(torch.load(args.ckpt))
            log_dir = os.path.dirname(args.ckpt)
        else:
            if args.desp:
                exp_name = '{}_cl{}_{}_{}'.format(args.model, args.cl, args.desp, time.strftime('%m%d%H%M'))
            else:
                exp_name = '{}_cl{}_{}'.format(args.model, args.cl, time.strftime('%m%d%H%M'))
            log_dir = os.path.join(args.log, exp_name)
        writer = SummaryWriter(log_dir)

        train_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        train_dataset = dataset_class[args.dataset](root_dir=args.train_root_dir, clip_len=args.cl, train=True, cal=False, transforms_=train_transforms, img_hgt=args.img_hgt, img_width=args.img_width)

        val_dataset = dataset_class[args.dataset](root_dir=args.cal_root_dir, clip_len=args.cl, train=False, cal=True, transforms_=train_transforms, img_hgt=args.img_hgt, img_width=args.img_width)

        print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False,
                                    num_workers=args.workers)
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False,
                                    num_workers=args.workers)

        ### loss funciton, optimizer and scheduler ###
        criterion = nn.CrossEntropyLoss()

        # setup optimizer
        fc2_params = list(map(id, net.module.fc2.parameters()))
        base_params = filter(lambda p: id(p) not in fc2_params, net.parameters())

        # optimizer = optim.SGD([{'params':base_params}, {'params':net.module.fc2.parameters(), 'lr': args.lr*args.lrMul}], lr=args.lr, momentum=0.9, weight_decay=args.wgtDecay, nesterov=True)

        optimizer = optim.AdamW(params= net.parameters(), lr= args.lr, weight_decay=args.wgtDecay)

        prev_best_val_loss = float('inf')
        prev_best_model_path = None
        for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
            time_start = time.time()
            train(args, net, criterion, optimizer, device, train_dataloader, writer, epoch)
            #print('Epoch time: {:.2f} s.'.format(time.time() - time_start))
            val_loss = validate(args, net, criterion, device, val_dataloader, writer, epoch)
            # scheduler.step(val_loss)         
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            # save model every 20 epoches
            if epoch % 20 == 0:
                torch.save(net.state_dict(), os.path.join(log_dir, 'model_{}.pt'.format(epoch)))
            # save model for the best val
            if val_loss < prev_best_val_loss:
                model_path = os.path.join(log_dir, 'best_model_{}.pt'.format(epoch))
                torch.save(net.state_dict(), model_path)
                prev_best_val_loss = val_loss
                if prev_best_model_path:
                    os.remove(prev_best_model_path)
                prev_best_model_path = model_path

    elif args.mode == 'test':  ########### Test #############
        net.load_state_dict(torch.load(args.ckpt))
        net.eval()
        test_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        test_dataset = dataset_class[args.dataset](root_dir=args.test_root_dir, clip_len=args.cl, train=False, cal=False, transforms_=test_transforms, img_hgt=args.img_hgt, img_width=args.img_width, in_dist_test=True)
        print("Test dataset len: ", test_dataset.__len__())
        test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
        print('TEST video number: {}.'.format(len(test_dataset)))
        criterion = nn.CrossEntropyLoss()
        test(args, net, criterion, device, test_dataloader)
