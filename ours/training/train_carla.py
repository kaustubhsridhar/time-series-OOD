"""Applied temporal transformation prediction.

For training with only OF - 
1) 5 transforms - python train_carla.py --cl 16 --log carla_log/final_results/5_classes/ --bs 2 --gpu 0 --use_image False --use_of True --transformation_list speed shuffle reverse periodic identity --epochs 600

For testing - python train_carla.py --bs 2 --mode test --ckpt carla_log/16/r3d_cl16_12091327/model_300.pt --gpu 2

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

from r3d import Regressor as r3d_regressor

from dataset.carla_optical_flow_appended import CARLADataset

import pdb
from distutils.util import strtobool

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
        # print("output: {}, target: {}".format(outputs, target_tranformations))
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
    parser.add_argument('--img_size', type=int, default=224, help='img height and width')
    parser.add_argument("--use_image", type=lambda x:bool(strtobool(x)), default=True, help="Use img info")
    parser.add_argument("--use_of", type=lambda x:bool(strtobool(x)), default=True, help="use optical flow info")
    parser.add_argument('--transformation_list', '--names-list', nargs='+', default=["speed","shuffle","reverse","periodic","identity"])

    args = parser.parse_args()
    return args

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
    # net = r3d_regressor().to(device)
    # net = torch.nn.DataParallel(net, device_ids=[args.gpu])

    in_channels = 3
    if args.use_image and args.use_of:
        in_channels = 6
    net = r3d_regressor(num_classes=len(args.transformation_list), in_channels=in_channels).to(device)

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

        orig_train_dataset = CARLADataset(root_dir='CARLA_dataset/Vanderbilt_data/training', clip_len=16, train=True, transforms_=train_transforms, img_size=args.img_size, use_image=args.use_image, use_of=args.use_of, transformation_list=args.transformation_list)

        train_dataset, val_dataset = random_split(orig_train_dataset, (len(orig_train_dataset)-13, 13), generator=torch.Generator().manual_seed(42)) # split val for 13 videos, we have a total of 33 videos, so training on 20. We will be using the validation dataset as calibration dataset for OOD detection in the CP framework.

        print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False,
                                    num_workers=args.workers)
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False,
                                    num_workers=args.workers)

        ### loss funciton, optimizer and scheduler ###
        criterion = nn.CrossEntropyLoss()

        # setup optimizer
        # fc2_params = list(map(id, net.fc2.parameters()))
        # base_params = filter(lambda p: id(p) not in fc2_params, net.parameters())

        # optimizer = optim.SGD([{'params':base_params}, {'params':net.module.fc2.parameters(), 'lr': args.lr*args.lrMul}], lr=args.lr, momentum=0.9, weight_decay=args.wgtDecay, nesterov=True)

        optimizer = optim.Adam(params= net.parameters(), lr= args.lr, weight_decay=args.wgtDecay)

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
        test_dataset = CARLADataset(root_dir='CARLA_dataset/Vanderbilt_data/testing', clip_len=16, train=False, transforms_=test_transforms, img_size=args.img_size, in_dist_test=True, use_image=args.use_image, use_of=args.use_of, transformation_list=args.transformation_list)
        print("Test dataset len: ", test_dataset.__len__())
        test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
        print('TEST video number: {}.'.format(len(test_dataset)))
        criterion = nn.CrossEntropyLoss()
        test(args, net, criterion, device, test_dataloader)

        
