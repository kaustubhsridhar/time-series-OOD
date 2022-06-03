'''
command to run 
python check_OOD_GAIT.py --ckpt saved_models/gait_16.pt --save_dir gait_log --bs 3 --cuda --n 100 --transformation_list high_pass high_low low_high identity
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

from models.lenet import Regressor as regressor

from dataset.gait import GAIT

import pdb
from distutils.util import strtobool

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--bs', type=int, default=3)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ckpt', default='', help="path load the trained network")
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--trials', type=int, default=1, help='no. of trials for taking average for the final results')
parser.add_argument('--wl', type=int, default=16, help='window length')
parser.add_argument('--n', type=int, default=5, help='number of continuous windows with p-value < epsilon to detect OODness in the trace')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--save_dir', type=str, default='gait_log', help='directory for saving p-vaues')
parser.add_argument('--cal_root_dir', type=str, default='gait-in-neurodegenerative-disease-database-1.0.0',help='calibration data directory')
parser.add_argument('--in_test_root_dir', type=str, default='gait-in-neurodegenerative-disease-database-1.0.0',help='test data directory')
parser.add_argument('--out_test_root_dir', type=str, default='gait-in-neurodegenerative-disease-database-1.0.0',help='test data directory')
parser.add_argument('--transformation_list', '--names-list', nargs='+', default=["low_pass", "high_pass", "identity"])

opt = parser.parse_args()
print(opt)


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(opt.gpu) if use_cuda else "cpu")

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

########### model ##############
net = regressor(num_classes=len(opt.transformation_list)).to(device)
net.load_state_dict(torch.load(opt.ckpt))
net.eval()

# pdb.set_trace()

criterion = nn.CrossEntropyLoss()

def calc_test_ce_loss(opt, model, criterion, device, test_dataset, in_dist=True):
    torch.set_grad_enabled(False)
    model.eval()

    all_traces_ce_loss = []

    key_list = ["0", "1", "2", "3", "4"]
    trasform_losses_dictionary = dict.fromkeys(key_list)
    for key in key_list:
         trasform_losses_dictionary[key] = []

    for test_data_idx in range(0, test_dataset.__len__()): # loop over all test datapoints
        
        trace_ce_loss = []
        
        for orig_window, transformed_window, transformation in test_dataset.__get_test_item__(test_data_idx): # loop over sliding window in the test trace
            
            orig_window = orig_window.unsqueeze(0)
            transformed_window = transformed_window.unsqueeze(0)
            orig_window = orig_window.to(device)
            transformed_window = transformed_window.to(device)
            transformation = [transformation]
            target_transformation = torch.tensor(transformation).to(device)
            # forward
            output = model(orig_window, transformed_window)
            # print("Output: {} and target: {}".format(torch.argmax(output), target_transformation))
            loss = criterion(output, target_transformation)
            # print("Output: {}, target: {}, loss: {}".format(torch.argmax(output), target_transformation, float(loss)))
            trasform_losses_dictionary['{}'.format(target_transformation.item())].append(float(loss))
            # print("Loss: ", float(loss))
            trace_ce_loss.append(float(loss))

        all_traces_ce_loss.append(np.array(trace_ce_loss))
    
    import pickle 
    if in_dist:
        with open('{}/in_dist_transform_losses.pickle'.format(opt.save_dir), 'wb') as handle:
            pickle.dump(trasform_losses_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open('{}/out_dist_transform_losses.pickle'.format(opt.save_dir), 'wb') as handle:
            pickle.dump(trasform_losses_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return np.array(all_traces_ce_loss)

def calc_cal_ce_loss(opt, model, criterion, device, cal_dataloader): # for calibration datapoint, we want one randomly sampled window for 1 datapoint
    torch.set_grad_enabled(False)
    model.eval()

    ce_loss_all_iter = []

    # torch.manual_seed(opt.seed)
    # np.random.seed(opt.seed)
    # random.seed(opt.seed)

    # definning dictionary for saving losses
    key_list = ["0", "1", "2", "3", "4"]
    trasform_losses_dictionary = dict.fromkeys(key_list)
    for key in key_list:
         trasform_losses_dictionary[key] = []

    for iter in range(0, opt.n): # n iterations with random sampling of windows and transformations on calibration datapoints
        ce_loss = []
        for _, data in enumerate(cal_dataloader, 1): # iteration over all calibration datapoints
            # get inputs
            orig_windows, transformed_windows, transformation = data
            orig_windows = orig_windows.to(device)
            transformed_windows = transformed_windows.to(device)
            target_transformations = torch.tensor(transformation).to(device)
            # forward
            outputs = model(orig_windows, transformed_windows)
            for i in range(len(outputs)):
                loss = criterion(outputs[i].unsqueeze(0), target_transformations[i].unsqueeze(0))
                ce_loss.append(loss.item())
                # print("Loss: {}, transformation: {}, predicted trans: {}".format(loss.item(), transformation[i], outputs[i]))
                trasform_losses_dictionary['{}'.format(target_transformations[i].item())].append(float(loss))

        print('[Cal] loss: ', ce_loss)
        ce_loss_all_iter.append(np.array(ce_loss))
    
    import pickle
    with open('{}/cal_transform_losses.pickle'.format(opt.save_dir), 'wb') as handle:
            pickle.dump(trasform_losses_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    # orig_train_dataset = CARLAVCOPDataset(root_dir='CARLA_dataset/Vanderbilt_data/training', win_len=opt.wl,  train=True, transforms_= transforms, img_size=opt.img_size)

    # train_dataset, cal_dataset = random_split(orig_train_dataset, (len(orig_train_dataset)-13, 13), generator=torch.Generator().manual_seed(42)) # split cal_set for 13 videos, we have a total of 33 videos, so training was done on 20 videos
    
    cal_dataset = GAIT(root_dir=opt.cal_root_dir, win_len=opt.wl, train=False, cal=True, in_dist_test=False, transformation_list=opt.transformation_list)

    print("Cal dataset len: ", cal_dataset.__len__())

    cal_dataloader = DataLoader(cal_dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.workers)

    # print("train_dataset_indices: {}, cal_dataset_indices: {}".format(train_dataset.indices, cal_dataset.indices))
    
    cal_set_ce_loss_all_iter = calc_cal_ce_loss(opt, model=net, criterion=criterion, device=device, cal_dataloader=cal_dataloader) # cal_set_ce_loss_all_iter = 2D vector with opt.n verctors, each vector contains loss for all calibration datapoints

    ############################################################################################################
    
    # In-Dist test CE loss
    # in_test_dataset = CARLAVCOPDataset('CARLA_dataset/Vanderbilt_data/testing', win_len=opt.wl, train=False, transforms_= transforms, img_size=opt.img_size, in_dist_test=True)
    in_test_dataset = GAIT(root_dir=opt.in_test_root_dir, win_len=opt.wl, train=False, cal=False, in_dist_test=True, transformation_list=opt.transformation_list)

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
    # out_test_dataset = CARLAVCOPDataset('CARLA_dataset/Vanderbilt_data/testing', win_len=opt.wl, train=False, transforms_= transforms, img_size=opt.img_size, in_dist_test=False)
    out_test_dataset = GAIT(root_dir=opt.out_test_root_dir, win_len=opt.wl, train=False, cal=False,  in_dist_test=False, transformation_list=opt.transformation_list)
    
    print("Out test dataset len: ", out_test_dataset.__len__())

    out_test_ce_loss_all_iters = []
    print("Calculating CE For OOD test data n times")
    for iter in range(0, opt.n):
        print('iter: ',iter+1)
        out_test_ce_loss = calc_test_ce_loss(opt, model=net, criterion=criterion, device=device, test_dataset=out_test_dataset, in_dist=False) # out_test_ce_loss = 2D vector with number of losses for each datapoint = no of windows in the datapoint
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

    in_fisher_per_win = []
    for trace_idx in range(len(in_fisher_values)): # iterating over each iD trace
        for win_idx in range(len(in_fisher_values[trace_idx])): # iterating over each window in the trace
            in_fisher_per_win.append(in_fisher_values[trace_idx][win_idx])
    in_fisher_per_win = np.array(in_fisher_per_win)

    out_fisher_per_win = []
    for trace_idx in range(len(out_fisher_values)): # iterating over each OOD trace
        for win_idx in range(len(out_fisher_values[trace_idx])): # iterating over each window in the trace
            out_fisher_per_win.append(out_fisher_values[trace_idx][win_idx])
    out_fisher_per_win = np.array(out_fisher_per_win)

    np.savez("{}/in_fisher_iter{}.npz".format(opt.save_dir, iter+1), in_fisher_values_win=in_fisher_per_win)
    np.savez("{}/out_fisher_iter{}.npz".format(opt.save_dir, iter+1), out_fisher_values_win=out_fisher_per_win)
    
    return in_fisher_per_win, out_fisher_per_win

def getAUROC(in_fisher_values, out_fisher_values):
    fisher_values = np.concatenate((in_fisher_values, out_fisher_values))

    indist_label = np.ones(len(in_fisher_values))
    ood_label = np.zeros(len(out_fisher_values))
    label = np.concatenate((indist_label, ood_label))

    from sklearn.metrics import roc_auc_score
    au_roc = roc_auc_score(label, fisher_values)*100
    return au_roc

def getTNR(in_fisher_values, out_fisher_values):

    in_fisher = np.sort(in_fisher_values)[::-1] # sorting in descending order
    tau = in_fisher[int(0.95*len(in_fisher))] # TNR at 95% TPR
    tnr = 100*(len(out_fisher_values[out_fisher_values<tau])/len(out_fisher_values))

    return tnr

if __name__ == "__main__":
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    auroc_all_trials = []
    # tnr_all_trials = []
    for trial in range(opt.trials):
        auroc_one_trial = []
        # tnr_one_trial = []
        checkOOD()
        for i in range(opt.n):
            in_fisher_values_per_win, out_fisher_values_per_win = eval_detection_fisher(i+1)
            au_roc = getAUROC(in_fisher_values_per_win, out_fisher_values_per_win)
            auroc_one_trial.append(au_roc)
            # tnr = getTNR(in_fisher_values_per_win, out_fisher_values_per_win)
            # tnr_one_trial.append(tnr)
            print("For trial: {}, n: {}, AUROC: {}".format(trial+1, i+1, au_roc))
            # print("For trial: {}, n: {}, TNR: {}".format(trial+1, i+1, tnr))
        auroc_all_trials.append(auroc_one_trial)
        # tnr_all_trials.append(tnr_one_trial)

    auroc_all_trials = np.array(auroc_all_trials)
    # tnr_all_trials = np.array(tnr_all_trials)

    print(np.mean(auroc_all_trials,0))
    # print(np.std(auroc_all_trials,0))

    # print("TNR Mean: ", np.mean(tnr_all_trials,0))


    
