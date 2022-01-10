import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from glob import glob
import os

def getTNR(in_scores, out_scores):
    # in_scores, out_scores = np.array(in_scores), np.array(out_scores)

    in_fisher = np.sort(in_scores)[::-1] # sorting in descending order
    tau = in_fisher[int(0.95*len(in_scores))] # TNR at 95% TPR
    tnr = 100*(len(out_scores[out_scores<tau])/len(out_scores))

    return tnr

def get_roc_curve(in_scores, out_scores):
    in_scores = list(in_scores)
    out_scores = list(out_scores)

    GTs = [1 for _ in range(len(in_scores))] + [0 for _ in range(len(out_scores))]
    all_scores = in_scores + out_scores
    assert len(GTs) == len(all_scores)
    fpr, tpr, threshs = roc_curve(GTs, all_scores)
    auroc = roc_auc_score(GTs, all_scores)
    return fpr, tpr, auroc*100

class Plotter():

    def __init__(self, fontsize=18, figsize=(7,5)):
        self.fontsize = fontsize
        self.figsize = figsize

    def setup_plot(self):
        plt.style.use('seaborn')
        plt.rc('font', size=self.fontsize)         # controls default text sizes
        plt.rc('axes', titlesize=self.fontsize)    # fontsize of the axes title
        plt.rc('axes', labelsize=self.fontsize)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=self.fontsize-2)   # fontsize of the tick labels
        plt.rc('ytick', labelsize=self.fontsize-2)   # fontsize of the tick labels
        plt.rc('legend', fontsize=self.fontsize)   # legend fontsize
        plt.rc('figure', titlesize=self.fontsize)  # fontsize of the figure title
        plt.rc('lines', linewidth=3)                # linewidth

    def drift_plot(self, npz_folder_root):
        self.setup_plot()
        plt.figure(0, figsize=self.figsize)
        legend_list = []
        methods = {'ours': 'iDECODeT', 'NTU': 'Feng et al.'}
        colors = {'ours': 'b', 'NTU': 'r'}
        for method in methods.keys():
            if method == 'ours':
                in_scores = np.load(npz_folder_root+f'/drift_in_{method}.npz', allow_pickle=True)['in_fisher_values_win']
                out_scores = np.load(npz_folder_root+f'/drift_out_{method}.npz', allow_pickle=True)['out_fisher_values_win']
            else:
                in_scores = np.load(npz_folder_root+f'/drift_in_{method}.npy')
                out_scores = np.load(npz_folder_root+f'/drift_out_{method}.npy')
            fpr, tpr, auroc = get_roc_curve(in_scores, out_scores)
            tnr = getTNR(in_scores, out_scores)
            plt.plot(fpr, tpr, color=colors[method])
            legend_list.append(f'{methods[method]} (AUROC: {auroc:.2f}, TNR: {tnr:.2f})')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.legend(legend_list)

        try:
            os.mkdir('./plots_saved/')
        except:
            pass
        plt.savefig('./plots_saved/drift.png')

    def replay_plots(self, npz_folder_root):
        self.setup_plot()
        plt.figure(1, figsize=self.figsize)
        all_tnrs = []
        methods = {'ours': 'iDECODeT', 'NTU': 'Feng et al.', 'Vanderbilt': 'Cai et al.'}#, 'Beta-VAE': 'Ramakrishna et al.'}
        colors = {'ours': 'b', 'NTU': 'r', 'Vanderbilt': 'c'}#, 'Beta-VAE': 'k'}
        legend_list = []
        for method in methods.keys():
            if method == 'ours':
                in_scores = np.load(npz_folder_root+f'/replay_win_in_{method}.npz', allow_pickle=True)['in_fisher_values_win']
                out_scores = np.load(npz_folder_root+f'/replay_win_out_{method}.npz', allow_pickle=True)['out_fisher_values_win']
            else:
                in_scores = np.load(npz_folder_root+f'/replay_win_in_{method}.npy')
                out_scores = np.load(npz_folder_root+f'/replay_win_out_{method}.npy')
            fpr, tpr, auroc = get_roc_curve(in_scores, out_scores)
            tnr = getTNR(in_scores, out_scores)
            all_tnrs.append(tnr)
            plt.plot(fpr, tpr, color=colors[method])
            legend_list.append(f'{methods[method]} (AUROC: {auroc:.2f})')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.legend(legend_list)

        try:
            os.mkdir('./plots_saved/')
        except:
            pass
        plt.savefig('./plots_saved/replay_roc.png')

        plt.figure(2, figsize=self.figsize)
        barlist=plt.bar(methods.values(), all_tnrs)
        for i, key in enumerate(methods.keys()):
            barlist[i].set_color(colors[key])
        plt.ylabel('TNR')
        plt.savefig('./plots_saved/replay_tnr.png')
            

        
        
        
        

        
        
        
