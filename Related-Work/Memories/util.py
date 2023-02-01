import json
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import metrics

def plot_one_result(memory_dir,window_size,dist,window_threshold,task):
    prob_list = [0.8,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.0,1.01,1.02,1.03,1.04,1.05]
    fp_prediction = []
    fn_prediction=[]
    if task == "heavy_rain":
        
        for prob in prob_list:
            in_file = "./ood_result"+"_in_"+task+"_"+str(dist)+"_"+str(window_size)+"_"+str(prob)+"_"+str(window_threshold)+".json"
            with open(os.path.join("results",in_file)) as json_file:
                data = json.load(json_file)
                fp_prediction.append(data["detection_rate"])
            json_file.close()
            out_file = "./ood_result"+"_out_"+task+"_"+str(dist)+"_"+str(window_size)+"_"+str(prob)+"_"+str(window_threshold)+".json"
            with open(os.path.join("results",out_file)) as json_file:
                data = json.load(json_file)
                fn_prediction.append(100-data["detection_rate"])
            json_file.close()
            
        plt.figure()

        plt.xticks(fontsize=18,weight="bold")
        plt.yticks(fontsize=18,weight="bold")
        plt.plot(prob_list,fp_prediction, label='d:  '+str(dist)+' T/W: '+str(window_threshold)+'/'+str(window_size))
        plt.legend(fontsize=12)
        plt.ylabel("False Positive Rate",fontsize=18,weight="bold")
        plt.xlabel("Probability Density Threshold",fontsize=18,weight="bold")
        plt.savefig("./results/p_heavy_rain_false_positive_T_"+str(window_threshold)+"_W_"+str(window_size)+"_d_"+str(dist)+".png",bbox_inches='tight')
        plt.close() 

        plt.figure()
        plt.xticks(fontsize=18,weight="bold")
        plt.yticks(fontsize=18,weight="bold")
        plt.plot(prob_list,fn_prediction, label='d:  '+str(dist)+' T/W: '+str(window_threshold)+'/'+str(window_size))
        plt.legend(fontsize=12)
        plt.ylabel("False Negative Rate",fontsize=18,weight="bold")
        plt.xlabel("Probability Density Threshold",fontsize=18,weight="bold")
        plt.savefig("./results/p_heavy_rain_false_negative_T_"+str(window_threshold)+"_W_"+str(window_size)+"_d_"+str(dist)+".png",bbox_inches='tight')
        plt.close() 

def plot_ablation():
    prob_list = [0.8,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.0,1.01,1.02,1.03,1.04,1.05]
    dist=0.2
    window_threshold = 5
    window_size = [5,10,15]
    task = "heavy_rain"
    fp_prediction={window: [] for window in window_size}
    fn_prediction={window: [] for window in window_size}
    for window in window_size:
        for prob in prob_list:
            in_file = "./ood_result"+"_in_"+task+"_"+str(dist)+"_"+str(window)+"_"+str(prob)+"_"+str(window_threshold)+".json"
            with open(os.path.join("results",in_file)) as json_file:
                data = json.load(json_file)
                fp_prediction[window].append(data["detection_rate"])
            json_file.close()
            out_file = "./ood_result"+"_out_"+task+"_"+str(dist)+"_"+str(window)+"_"+str(prob)+"_"+str(window_threshold)+".json"
            with open(os.path.join("results",out_file)) as json_file:
                data = json.load(json_file)
                fn_prediction[window].append(100-data["detection_rate"])
            json_file.close()
            
    plt.figure()
    for window in window_size:
        plt.plot(prob_list,fp_prediction[window], label='d:  '+str(dist)+' T/W: '+str(window_threshold)+'/'+str(window))

    plt.xticks(fontsize=18,weight="bold")
    plt.yticks(fontsize=18,weight="bold")

    plt.legend(fontsize=12)
    plt.ylabel("False Positive Rate",fontsize=18,weight="bold")
    plt.xlabel("Probability Density Threshold",fontsize=18,weight="bold")
    plt.savefig("./results/p_heavy_rain_false_positive.png",bbox_inches='tight')
    plt.close() 

    plt.figure()
    for window in window_size:
        plt.plot(prob_list,fn_prediction[window], label='d:  '+str(dist)+' T/W: '+str(window_threshold)+'/'+str(window))
    plt.xticks(fontsize=18,weight="bold")
    plt.yticks(fontsize=18,weight="bold")

    plt.legend(fontsize=12)
    plt.ylabel("False Negative Rate",fontsize=18,weight="bold")
    plt.xlabel("Probability Density Threshold",fontsize=18,weight="bold")
    plt.savefig("./results/p_heavy_rain_false_negative.png",bbox_inches='tight')
    plt.close() 

def plot_icad(file_name):
    #carla_exp_results_0.4_split.json
    dist_list = [0.2,0.3]
    window_list = [5,10]
    dist_dict={0.2:[0.005,0.01,0.1,0.15,0.2],
    0.3:[0.005,0.01,0.1,0.15,0.2]}

    with open(file_name) as file:
        raw_data = json.loads(file.read())
    file.close()

    data=[]

    fp_prediction={dist:{window: [] for window in window_list} for dist in dist_list}
    fn_prediction={dist:{window: [] for window in window_list} for dist in dist_list}
    for i in raw_data:
        if i['eps'] not in dist_dict[i['dist']]:
            print(i)
        else:
            fp_prediction[i['dist']][i['window']].append(i['in']["detection_rate"])
            fn_prediction[i['dist']][i['window']].append(100-i['out']["detection_rate"])


    keys = [math.log10(i) for i in [0.005,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]]
    print(keys)
    
    fig = plt.figure()

    plt.figure()
    for window in window_list:
        for dist in dist_list:
            plt.plot(keys,fp_prediction[dist][window], label="d: "+str(dist)+" "+"window size "+str(window))

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.legend(fontsize=12)
    plt.ylabel("False Positive Rate",fontsize=18)
    plt.xlabel("OOD Detection Threshold (\u03B5)",fontsize=18)
    plt.savefig("./results/p_heavy_rain_false_positive_icad.png",bbox_inches='tight')
    plt.close() 

    plt.figure()
    for window in window_list:
        for dist in dist_list:
            plt.plot(keys,fn_prediction[dist][window], label="d: "+str(dist)+" "+"window size "+str(window))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.legend(fontsize=12)
    plt.ylabel("False Negative Rate",fontsize=18)
    plt.xlabel("Probability Density Threshold (\u03B5)",fontsize=18)
    plt.savefig("./results/p_heavy_rain_false_negative_icad.png",bbox_inches='tight')
    plt.close()     

def compute_auroc(task):
    in_file = 'test_in.json'
    out_file = 'test_'+task+'.json'
    with open(in_file, "r") as read_file:
        data = json.load(read_file)
    read_file.close()
    y = data['gt']
    y_pred = data['pred']
    with open(out_file, "r") as read_file:
        data = json.load(read_file)
    read_file.close()
    y = y + data['gt']
    y_pred = y_pred + data['pred']
    fpr, tpr, _ = metrics.roc_curve(y,  y_pred)
    auc = metrics.roc_auc_score(y, y_pred)

    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.savefig(task+'.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute_auroc", type = bool, default = False, help = "compute_auroc" )
    parser.add_argument("--task", type = str, default = 'out_foggy', help = "compute_auroc" )   
    args = parser.parse_args()
    
    if args.compute_auroc:
        compute_auroc(args.task)
