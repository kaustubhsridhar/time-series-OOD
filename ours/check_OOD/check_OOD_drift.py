'''
command to run 
For CARLA
python check_OOD.py --gpu 0 --cuda --ckpt test/r3d_cl16_11161908/model_300.pt --model r3d --n 5 --save_dir cl16 --dataset CARLAVCOPDataset

For drift on cl = 24
python check_OOD.py --gpu 1 --cuda --ckpt drift_log/r3d_cl24_11302229/model_500.pt --model r3d --n 5 --save_dir drift_log/24 --dataset DriftDataset --cl 24

for crowd on cl = 24
python check_OOD.py --gpu 2 --cuda --ckpt crowd_log/r3d_cl24_11302256/model_500.pt --model r3d --n 5 --save_dir crowd_log/24 --dataset CrowdDataset --cl 24 --cal_root_dir moving_crowd_dataset/in/calibration --in_test_root_dir moving_crowd_dataset/in/testing --out_test_root_dir moving_crowd_dataset/out


*********** FOR GETTING FINAL RESULTS ************
For Drift :
python check_OOD.py --gpu 0 --cuda --ckpt drift_log/4_classes/16/new_in_data/r3d_cl16_12081834/model_3000.pt --model r3d --n 5 --save_dir tmp_results --dataset DriftDataset --cl 16 --out_test_root_dir drift_dataset/out --in_test_root_dir drift_dataset/temp_in/testing/ --cal_root_dir=drift_dataset/temp_in/calibration/ --trials 5

For Crowd : 
python check_OOD.py --gpu 2 --cuda --ckpt crowd_log/16/r3d_cl16_12091250/model_3000.pt --model r3d --n 5 --save_dir tmp_results --dataset CrowdDataset --cl 16 --out_test_root_dir moving_crowd_dataset/out/ --in_test_root_dir moving_crowd_dataset/in/testing/ --cal_root_dir=moving_crowd_dataset/in/calibration/ --trials 1

'''

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, random_split
import itertools

import numpy as np

# from models.c3d import C3D
# from models.r3d import R3DNet
# from models.r21d import R2Plus1DNet
# from models.vcopn import VCOPN

from models.r3d import Regressor as r3d_regressor

from dataset.carla import CARLAVCOPDataset
from dataset.drift import DriftDataset
from dataset.crowd import CrowdDataset

import PIL
import csv

import pdb

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean in_dist_testue expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--bs', type=int, default=2)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ckpt', default='', help="path load the trained network")
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--trials', type=int, default=1, help='no. of trials for taking average for the final results')
parser.add_argument('--model', type=str, default='r3d', help='c3d/r3d/r21d')
parser.add_argument('--cl', type=int, default=16, help='clip length')
parser.add_argument('--img_size', type=int, default=224, help='img height/width')
parser.add_argument('--n', type=int, default=5, help='number of continuous windows with p-value < epsilon to detect OODness in the trace')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--save_dir', type=str, default='win64', help='directory for saving p-vaues')
parser.add_argument('--cal_root_dir', type=str, default='drift_dataset/in/calibration',help='calibration data directory')
parser.add_argument('--in_test_root_dir', type=str, default='drift_dataset/in/testing',help='test data directory')
parser.add_argument('--out_test_root_dir', type=str, default='drift_dataset/out',help='test data directory')
parser.add_argument('--img_hgt', type=int, default=224, help='img height')
parser.add_argument('--img_width', type=int, default=224, help='img width')
parser.add_argument('--dataset', default='CARLAVCOPDataset', help='dataset - CARLAVCOPDataset/DriftDataset/CrowdDataset')

opt = parser.parse_args()
print(opt)

dataset_class = {'CARLAVCOPDataset': CARLAVCOPDataset, 'DriftDataset': DriftDataset, 'CrowdDataset': CrowdDataset}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(opt.gpu) if use_cuda else "cpu")

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

########### model ##############
# if opt.model == 'c3d':
#     base = C3D(with_classifier=False)
# elif opt.model == 'r3d':
#     base = R3DNet(layer_sizes=(1,1,1,1), with_classifier=False)
# elif opt.model == 'r21d':   
#     base = R2Plus1DNet(layer_sizes=(1,1,1,1), with_classifier=False)
# net = VCOPN(base_network=base, feature_size=512, tuple_len=opt.tl).to(device)
net = r3d_regressor().to(device)
net = torch.nn.DataParallel(net, device_ids=[int(opt.gpu)])
net.load_state_dict(torch.load(opt.ckpt))
net.eval()

transforms = transforms.Compose([
            transforms.ToTensor()
        ])

# pdb.set_trace()

criterion = nn.CrossEntropyLoss()

def calc_test_ce_loss(opt, model, criterion, device, test_dataset):
    torch.set_grad_enabled(False)
    model.eval()

    all_traces_ce_loss = []

    for test_data_idx in range(0, test_dataset.__len__()): # loop over all test datapoints
        
        trace_ce_loss = []

        for orig_clip, transformed_clip, transformation in test_dataset.__get_test_item__(test_data_idx): # loop over sliding window in the test trace
            orig_clip = orig_clip.unsqueeze(0)
            transformed_clip = transformed_clip.unsqueeze(0)
            orig_clip = orig_clip.to(device)
            transformed_clip = transformed_clip.to(device)
            transformation = [transformation]
            target_transformation = torch.tensor(transformation).to(device)
            # forward
            output = model(orig_clip, transformed_clip)
            # print("Output: {} and target: {}".format(torch.argmax(output), target_transformation))
            loss = criterion(output, target_transformation)
            # print("Loss: ", float(loss))
            trace_ce_loss.append(float(loss))

        all_traces_ce_loss.append(np.array(trace_ce_loss))
    
    return np.array(all_traces_ce_loss)

def calc_cal_ce_loss(opt, model, criterion, device, cal_dataloader): # for calibration datapoint, we want one randomly sampled window for 1 datapoint
    torch.set_grad_enabled(False)
    model.eval()

    ce_loss_all_iter = []

    # torch.manual_seed(opt.seed)
    # np.random.seed(opt.seed)
    # random.seed(opt.seed)

    for iter in range(0, opt.n): # n iterations with random sampling of windows and transformations on calibration datapoints
        ce_loss = []
        for _, data in enumerate(cal_dataloader, 1): # iteration over all calibration datapoints
            # get inputs
            orig_clips, transformed_clips, transformation = data
            orig_clips = orig_clips.to(device)
            transformed_clips = transformed_clips.to(device)
            target_transformations = torch.tensor(transformation).to(device)
            # forward
            outputs = model(orig_clips, transformed_clips)
            for i in range(len(outputs)):
                loss = criterion(outputs[i].unsqueeze(0), target_transformations[i].unsqueeze(0))
                ce_loss.append(loss.item())
                # print("Loss: {}, transformation: {}, predicted trans: {}".format(loss.item(), transformation[i], outputs[i]))

        print('[Cal] loss: ', ce_loss)
        ce_loss_all_iter.append(np.array(ce_loss))
    return np.array(ce_loss_all_iter)

def calc_p_value(test_ce_loss, cal_set_ce_loss):

    cal_set_ce_loss_reshaped = cal_set_ce_loss
    cal_set_ce_loss_reshaped = cal_set_ce_loss_reshaped.reshape(1,-1) # cal_set_ce_loss reshaped into row vector

    test_ce_loss_reshaped = test_ce_loss
    test_ce_loss_reshaped = test_ce_loss_reshaped.reshape(-1,1) # test_ce_loss reshaped into column vector

    #pdb.set_trace()
    compare = (test_ce_loss_reshaped)<=(cal_set_ce_loss_reshaped)
    p_value = np.sum(compare, axis=1)
    p_value = (p_value+1)/(len(cal_set_ce_loss)+1)
    # print(p_value)

    return p_value

def checkOOD(n = opt.n):  

    # CAL set CE Loss
    # orig_train_dataset = CARLAVCOPDataset(root_dir='CARLA_dataset/Vanderbilt_data/training', clip_len=opt.cl,  train=True, transforms_= transforms, img_size=opt.img_size)

    # train_dataset, cal_dataset = random_split(orig_train_dataset, (len(orig_train_dataset)-13, 13), generator=torch.Generator().manual_seed(42)) # split cal_set for 13 videos, we have a total of 33 videos, so training was done on 20 videos
    
    cal_dataset = dataset_class[opt.dataset](root_dir=opt.cal_root_dir, clip_len=opt.cl, train=False, cal=True, transforms_=transforms, img_hgt=opt.img_hgt, img_width=opt.img_width)

    print("Cal dataset len: ", cal_dataset.__len__())

    cal_dataloader = DataLoader(cal_dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.workers)

    # print("train_dataset_indices: {}, cal_dataset_indices: {}".format(train_dataset.indices, cal_dataset.indices))
    
    cal_set_ce_loss_all_iter = calc_cal_ce_loss(opt, model=net, criterion=criterion, device=device, cal_dataloader=cal_dataloader) # cal_set_ce_loss_all_iter = 2D vector with opt.n verctors, each vector contains loss for all calibration datapoints

    ############################################################################################################
    
    # In-Dist test CE loss
    # in_test_dataset = CARLAVCOPDataset('CARLA_dataset/Vanderbilt_data/testing', clip_len=opt.cl, train=False, transforms_= transforms, img_size=opt.img_size, in_dist_test=True)
    in_test_dataset = dataset_class[opt.dataset](root_dir=opt.in_test_root_dir, clip_len=opt.cl, train=False, cal=False, transforms_=transforms, img_hgt=opt.img_hgt, img_width=opt.img_width, in_dist_test=True)

    print("In test dataset len: ", in_test_dataset.__len__())
    in_test_ce_loss_all_iters = []
    print("Calculating CE loss for in-dist test data n times")
    for iter in range(0, opt.n):
        print('iter: ',iter+1)
        in_test_ce_loss = calc_test_ce_loss(opt, model=net, criterion=criterion, device=device, test_dataset=in_test_dataset) # in_test_ce_loss = 2D vector with number of losses for each datapoint = no of windows in the datapoint
        in_test_ce_loss_all_iters.append(in_test_ce_loss)
    in_test_ce_loss_all_iters = np.array(in_test_ce_loss_all_iters) # 3D array

    #############################################################################################################
    
    # Out-Dist CE loss
    # out_test_dataset = CARLAVCOPDataset('CARLA_dataset/Vanderbilt_data/testing', clip_len=opt.cl, train=False, transforms_= transforms, img_size=opt.img_size, in_dist_test=False)
    out_test_dataset = dataset_class[opt.dataset](root_dir=opt.out_test_root_dir, clip_len=opt.cl, train=False, cal=False, transforms_=transforms, img_hgt=opt.img_hgt, img_width=opt.img_width, in_dist_test=False)
    
    print("Out test dataset len: ", out_test_dataset.__len__())

    out_test_ce_loss_all_iters = []
    print("Calculating CE For OOD test data n times")
    for iter in range(0, opt.n):
        print('iter: ',iter+1)
        out_test_ce_loss = calc_test_ce_loss(opt, model=net, criterion=criterion, device=device, test_dataset=out_test_dataset) # out_test_ce_loss = 2D vector with number of losses for each datapoint = no of windows in the datapoint
        #print("Out loss: ", out_test_ce_loss)
        out_test_ce_loss_all_iters.append(out_test_ce_loss)
    out_test_ce_loss_all_iters = np.array(out_test_ce_loss_all_iters) # 3D array

    ############################################################################################################
    
    # Saving CE losses
    np.savez("{}/in_ce_loss_{}_iters.npz".format(opt.save_dir, opt.n), in_ce_loss=in_test_ce_loss_all_iters)
    np.savez("{}/out_ce_loss_{}_iters.npz".format(opt.save_dir, opt.n), out_ce_loss=out_test_ce_loss_all_iters)
    np.savez("{}/cal_ce_loss_{}_iters.npz".format(opt.save_dir, opt.n), ce_loss=cal_set_ce_loss_all_iter)

    ############################################################################################################
    # in-dist n p-values
    print("Calculating n p-values for in-dist test data")
    # pdb.set_trace()
    for iter in range(0, opt.n): # n iterations
        in_p_values_all_traces = []
        in_test_ce_loss = in_test_ce_loss_all_iters[iter]
        for test_idx in range(0, len(in_test_ce_loss)): # iteration over test datapoints
            in_p_values = []
            for window_idx in range(0, len(in_test_ce_loss[test_idx])): # iteration over windows of a test datapoint
                in_p_values.append(calc_p_value(in_test_ce_loss[test_idx][window_idx], cal_set_ce_loss_all_iter[iter]))
            in_p_values_all_traces.append(np.array(in_p_values))
        np.savez("{}/in_p_values_iter{}.npz".format(opt.save_dir, iter+1), p_values=np.array(in_p_values_all_traces))

    ############################################################################################################

    # out-dist p-values
    print("Calculating n p-values for OOD test data")
    for iter in range(0, opt.n): # n iterations
        out_p_values_all_traces = []
        out_test_ce_loss = out_test_ce_loss_all_iters[iter]
        for test_idx in range(0, len(out_test_ce_loss)): # iter over all test datapoints
            out_p_values = []
            for window_idx in range(0, len(out_test_ce_loss[test_idx])): # iteration over windows of the test datapoint
                out_p_values.append(calc_p_value(out_test_ce_loss[test_idx][window_idx], cal_set_ce_loss_all_iter[iter]))
            out_p_values_all_traces.append(np.array(out_p_values))
        np.savez("{}/out_p_values_iter{}.npz".format(opt.save_dir, iter+1), p_values=np.array(out_p_values_all_traces))
    
# def eval_detection(eval_n):
#     in_p = []
#     out_p = []
#     for iter in range(0, opt.n):
#         in_p.append(np.load("in_p_values_iter{}.npz".format(iter+1), allow_pickle=True)['p_values'])
#         out_p.append(np.load("out_p_values_iter{}.npz".format(iter+1), allow_pickle=True)['p_values'])

#     in_p_values = in_p[0]
#     out_p_values = out_p[0]
    
#     for i in range(1, eval_n):
#         for j in range(0, len(in_p[i])):
#             in_p_values[j] += in_p[i][j]
    
#     for i in range(1, eval_n):
#         for j in range(0, len(out_p[i])):
#             out_p_values[j] += out_p[i][j]

#     # false detection 
#     counter_fd_traces = 0
#     for i in range(0, len(in_p_values)): # iterating over all test iD traces, in_p_values is a 2D list
#         for j in range(0, len(in_p_values[i])): # iterating over all windows in the test trace
#             if in_p_values[i][j] == 0:
#                 counter_fd_traces += 1
#                 break
    
#     print("No. of false detection traces: ", counter_fd_traces)

#     # positive detection 
#     counter_pd_traces = 0
#     for i in range(0, len(out_p_values)): # iterationg over all test OOD traces, out_p_values is a 2D list
#         for j in range(0, len(out_p_values[i])): # iterating over all windows in the test trace
#             if out_p_values[i][j] == 0:
#                 counter_pd_traces += 1
#                 break
    
#     print("No. of positive detection traces: ", counter_pd_traces)

def calc_fisher_value(t_value, eval_n):
    summation = 0
    for i in range(eval_n): # calculating fisher value for the window in the datapoint
        summation += ((-np.log(t_value))**i)/np.math.factorial(i)
    return t_value*summation 

def calc_fisher_batch(p_values, eval_n): # p_values is 3D
    output = [[None]*len(window) for window in p_values[0]] # output is a 2D list for each datapoint, no of datapoints X number of windows in each datapoint
    for i in range(len(p_values[0])): #iterating over test datapoints
        for j in range(len(p_values[0][i])): #iterating over p-values for windows in the test datapoint
            prod = 1
            for k in range(eval_n):
                prod*=p_values[k][i][j][0]

            output[i][j] = calc_fisher_value(prod, eval_n)

    return output  # a 2D fisher value output for each window in each test datapoint

def eval_detection_fisher(eval_n):
    #pdb.set_trace()
    in_p = [] # 3D
    out_p = [] # 3D
    for iter in range(0, eval_n):
        in_p.append(np.load("{}/in_p_values_iter{}.npz".format(opt.save_dir, iter+1), allow_pickle=True)['p_values'])
        out_p.append(np.load("{}/out_p_values_iter{}.npz".format(opt.save_dir, iter+1), allow_pickle=True)['p_values'])

    in_fisher_values = calc_fisher_batch(in_p, eval_n) # a 2D fisher value output for each window in each iD test datapoint
    out_fisher_values = calc_fisher_batch(out_p, eval_n) # a 2D fisher value output for each window in each OOD test datapoint
    # pdb.set_trace()

    
    in_min_fisher_per_trace = [min(d) for d in in_fisher_values]
    # print("min_in_fisher_values", in_min_fisher_per_trace)
    out_min_fisher_per_trace = [min(d) for d in out_fisher_values]
    # print("min_out_fisher_values", out_min_fisher_per_trace)

    np.savez("{}/in_min_fisher_iter{}.npz".format(opt.save_dir, iter+1), in_min_fisher_values=np.array(in_min_fisher_per_trace))
    np.savez("{}/out_min_fisher_iter{}.npz".format(opt.save_dir, iter+1), out_min_fisher_values=np.array(out_min_fisher_per_trace))

    #out_min_fisher_index_per_trace = [d.index(min(d)) for d in out_fisher_values]
    #print("Detection at frames: ", out_min_fisher_index_per_trace)
    # first_ood_frame_per_trace = [77, 46, 61, 50, 79, 64, 60, 57, 40, 57, 58, 46, 99, 86, 82, 83, 53, 54, 55, 46, 72, 57, 61, 42, 41, 56, 44, 36, 67, 70, 71, 50, 73, 85, 70, 53, 84, 79, 49, 78, 48, 81, 58, 43, 104, 72, 65, 65, 45, 87, 46, 39, 77, 50, 80, 38, 62, 59, 71, 61, 52, 49, 63, 52, 68, 82, 92, 66, 47, 53, 54, 55, 41] # the frame no. at which precipitation >= 20
    # print("Detection delay: ", np.array(out_min_fisher_index_per_trace)-np.array(first_ood_frame_per_trace))

    # get_det_delay(in_min_fisher_per_trace, out_fisher_values)
    
    return in_min_fisher_per_trace, out_min_fisher_per_trace

def getAUROC(in_min_fisher_values, out_min_fisher_values):
    fisher_values = np.concatenate((in_min_fisher_values, out_min_fisher_values))

    indist_label = np.ones(len(in_min_fisher_values))
    ood_label = np.zeros(len(out_min_fisher_values))
    label = np.concatenate((indist_label, ood_label))

    from sklearn.metrics import roc_auc_score
    au_roc = roc_auc_score(label, fisher_values)*100
    return au_roc

def get_det_delay(in_min_fisher_per_trace, out_fisher_values):

    # calculating detection delay at 95% TPR (iD is positive here)
    in_min_fisher_per_trace = np.array(in_min_fisher_per_trace)
    in_min_fisher_per_trace_sorted = (np.sort(in_min_fisher_per_trace))[::-1]
    epsilon = in_min_fisher_per_trace_sorted[int(len(in_min_fisher_per_trace_sorted)*0.95)] # epsilon at 95% TPR
    # print("epsilon: ", epsilon) ##### DO -1 here ###############

    det_delay = [1000]*len(out_fisher_values)
    tn = 0

    for i in range(len(out_fisher_values)): # iterating over OOD traces
        # print("out_fisher_values: ", out_fisher_values[i])
        for j in range(len(out_fisher_values[i])): # iterating over windows in the trace
            if out_fisher_values[i][j] < epsilon:
                det_delay[i] = j
                tn += 1
                break

    print("TNR at 95% TPR: ", 100*(tn/len(out_fisher_values)))
    print("Det at win: ", det_delay)

if __name__ == "__main__":
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    auroc_all_trials = []
    for trial in range(opt.trials):
        auroc_one_trial = []
        checkOOD()
        for i in range(opt.n):
            in_min_fisher_values, out_min_fisher_values = eval_detection_fisher(i+1)
            au_roc = getAUROC(in_min_fisher_values, out_min_fisher_values)
            auroc_one_trial.append(au_roc)
            print("For trial: {}, n: {}, AUROC: {}".format(trial+1, i+1, au_roc))
        auroc_all_trials.append(auroc_one_trial)

    auroc_all_trials = np.array(auroc_all_trials)

    print(np.mean(auroc_all_trials,0))
    print(np.std(auroc_all_trials,0))
