'''
command to run 
python check_OOD_baseline.py --disease_type als --wl 16 
'''


import numpy as np
from sklearn.svm import OneClassSVM
from sklearn import preprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--disease_type', type=str, default='als', help='als/hunt/park/all')
parser.add_argument('--root_dir', type=str, default='/home/ramneet/Documents/Time-series OOD detection with CP/gait-in-neurodegenerative-disease-database-1.0.0', help='path to the gait dataset')
parser.add_argument('--wl', type=int, default=16, help='sliding window length')

opt = parser.parse_args()

clip_len = opt.wl
root_dir = opt.root_dir

training_traces_data = [] 
training_traces = [1,2,3,4,5,6] 
for training_trace_id in training_traces: 
    f = open("{}/control{}.ts".format(root_dir,training_trace_id), "r") 
    cur_trace_data = []
    while(True): 
        line = f.readline() 
        if not line: 
            break  
        data = [float(item) for item in line.split()]  
        data = data[1:]  
        cur_trace_data.append(data) 
    f.close()
    training_traces_data.append(cur_trace_data) 

training_win_data = []

for trace_idx in range(len(training_traces_data)):
    for win_start in range(0, len(training_traces_data[trace_idx])-1*clip_len+1):
        training_win_data.append(training_traces_data[trace_idx][win_start:win_start+clip_len])

testing_in_traces_data = []
testing_in_traces = [12,13,14,15,16] 
for testing_in_trace_id in testing_in_traces: 
    f = open("{}/control{}.ts".format(root_dir,testing_in_trace_id), "r") 
    cur_trace_data = []
    while(True): 
        line = f.readline() 
        if not line: 
            break  
        data = [float(item) for item in line.split()]  
        data = data[1:]  
        cur_trace_data.append(data) 
    f.close() 
    testing_in_traces_data.append(cur_trace_data)

testing_in_win_data = [] 

for trace_idx in range(len(testing_in_traces_data)):
    for win_start in range(0, len(testing_in_traces_data[trace_idx])-1*clip_len+1):
        testing_in_win_data.append(testing_in_traces_data[trace_idx][win_start:win_start+clip_len])

 
ood_types = ['park', 'als', 'hunt']
testing_out_trace_ids = {'park': [1,4,7,8,10,11,12,13,14], 
            'als' : [1,12,13,4,5,6,7,3,9],
            'hunt' : [3,4,7,10,13,15,16,18,19]} 

if opt.disease_type != 'all':
    ood_types = [opt.disease_type]

testing_out_traces_data = []

for i,ood_type in enumerate(ood_types):
    print(ood_type)
    for ood_trace_id in testing_out_trace_ids[ood_type]: 
        f = open("{}/{}{}.ts".format(root_dir,ood_type,ood_trace_id), "r") 
        cur_trace_data = []
        while(True): 
            line = f.readline() 
            if not line: 
                break  
            data = [float(item) for item in line.split()]  
            data = data[1:] # excluding time data  
            cur_trace_data.append(data)  
        f.close()
        testing_out_traces_data.append(cur_trace_data)

testing_out_win_data = []

for trace_idx in range(len(testing_out_traces_data)):
    for win_start in range(0, len(testing_out_traces_data[trace_idx])-1*clip_len+1):
        testing_out_win_data.append(testing_out_traces_data[trace_idx][win_start:win_start+clip_len])

training_win_data = np.array(training_win_data)

# auto-correlation
correlated_training_win_data = []
for win_idx, win in enumerate(training_win_data): # iterating over training windows,  win is cl * 12
    cor_win_data = []
    for col_idx in range(win.shape[1]): # col_idx from 0 to 11
        cor_win_data.append(np.correlate(win[:,col_idx], win[:,col_idx]))
    cor_win_data = np.array(cor_win_data)
    cor_win_data = cor_win_data.reshape(-1)
    correlated_training_win_data.append(cor_win_data)

correlated_training_win_data = np.array(correlated_training_win_data)
#normalization
scaler = preprocessing.StandardScaler().fit(correlated_training_win_data)
correlated_training_win_data = scaler.transform(correlated_training_win_data)

testing_in_win_data = np.array(testing_in_win_data)
# auto-correlation
correlated_testing_in_win_data = []
for win_idx, win in enumerate(testing_in_win_data): # iterating over training windows,  win is cl * 12
    cor_win_data = []
    for col_idx in range(win.shape[1]): # col_idx from 0 to 11
        cor_win_data.append(np.correlate(win[:,col_idx], win[:,col_idx]))
    cor_win_data = np.array(cor_win_data)
    cor_win_data = cor_win_data.reshape(-1)
    correlated_testing_in_win_data.append(cor_win_data)

correlated_testing_in_win_data = np.array(correlated_testing_in_win_data)
#normalization
correlated_testing_in_win_data = scaler.transform(correlated_testing_in_win_data)


testing_out_win_data = np.array(testing_out_win_data)
# auto-correlation
correlated_testing_out_win_data = []
for win_idx, win in enumerate(testing_out_win_data): # iterating over training windows,  win is cl * 12
    cor_win_data = []
    for col_idx in range(win.shape[1]): # col_idx from 0 to 11
        cor_win_data.append(np.correlate(win[:,col_idx], win[:,col_idx]))
    cor_win_data = np.array(cor_win_data)
    cor_win_data = cor_win_data.reshape(-1)
    correlated_testing_out_win_data.append(cor_win_data)

correlated_testing_out_win_data = np.array(correlated_testing_out_win_data)
#normalization
correlated_testing_out_win_data = scaler.transform(correlated_testing_out_win_data)

# Training one-class SVM on training_win_data
clf = OneClassSVM().fit(correlated_training_win_data)
in_training_scores = clf.predict(correlated_training_win_data)
testing_in_scores = clf.predict(correlated_testing_in_win_data)
testing_out_scores = clf.predict(correlated_testing_out_win_data)

def getAUROC(in_scores, out_scores):
    all_scores = np.concatenate((in_scores, out_scores))

    indist_label = np.ones(len(in_scores))
    ood_label = np.zeros(len(out_scores))
    label = np.concatenate((indist_label, ood_label))

    from sklearn.metrics import roc_auc_score
    au_roc = roc_auc_score(label, all_scores)*100
    return au_roc

def getTNR(in_scores, out_scores):

    in_scores = np.sort(in_scores)[::-1] # sorting in descending order
    tau = in_scores[int(0.95*len(in_scores))] # TNR at 95% TPR
    tnr = 100*(len(out_scores[out_scores<tau])/len(out_scores))

    return tnr

#get AUROC and TNR
au_roc = getAUROC(testing_in_scores, testing_out_scores)
tnr = getTNR(testing_in_scores, testing_out_scores)
# print("AUROC: {}, TNR: {}".format(au_roc,tnr)) # 75.98592655706742
print("AUROC: {}".format(au_roc))


