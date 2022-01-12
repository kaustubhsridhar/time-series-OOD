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

    def __init__(self, fontsize=22, figsize=(7,5)):
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
        methods = {'ours': 'Ours', 'NTU': 'Feng et al.'}
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
        plt.figure(1, figsize=(8, 6))

        # plt.subplot(1, 2, 1)
        all_tnrs = []
        methods = {'ours': 'Ours', 'NTU': 'Feng et al.', 'Beta-VAE': 'Ramakrishna et al.', 'Vanderbilt': 'Cai et al.', }
        colors = {'ours': 'b', 'NTU': 'r', 'Vanderbilt': 'c', 'Beta-VAE': 'k'}
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
        # plt.legend(legend_list)

        try:
            os.mkdir('./plots_saved/')
        except:
            pass
        plt.savefig('./plots_saved/replay_roc.png')

        plt.figure(2, figsize=(6, 6))
        plt.xticks(color='white')
        # plt.subplot(1, 2, 2)
        barlist=plt.bar(methods.values(), all_tnrs)
        for i, key in enumerate(methods.keys()):
            barlist[i].set_color(colors[key])
        plt.ylabel('TNR (at 95% TPR)')
        plt.tight_layout()
        plt.savefig('./plots_saved/replay_tnr.png')

    def drift_cl_ablation(self, npz_folder_root):
        self.setup_plot()
        plt.figure(3, figsize=self.figsize)
        legend_list = []
        methods = {'ours': 'Ours', 'NTU': 'Feng et al.'}
        colors = {'ours': {'18': 'darkturquoise', '20': 'darkblue'}, 'NTU': {'18': 'crimson', '20': 'tomato'}}
        dash = {'20': '-', '18': '-'} # '16': '-', 
        for method in methods.keys():
            for cl in dash.keys():
                if cl == '16':
                    suffix = ''
                else:
                    suffix = f'_cl_{cl}'

                if method == 'ours':
                    in_scores = np.load(npz_folder_root+f'/drift_in_{method}{suffix}.npz', allow_pickle=True)['in_fisher_values_win']
                    out_scores = np.load(npz_folder_root+f'/drift_out_{method}{suffix}.npz', allow_pickle=True)['out_fisher_values_win']
                else:
                    in_scores = np.load(npz_folder_root+f'/drift_in_{method}{suffix}.npy')
                    out_scores = np.load(npz_folder_root+f'/drift_out_{method}{suffix}.npy')
                fpr, tpr, auroc = get_roc_curve(in_scores, out_scores)
                tnr = getTNR(in_scores, out_scores)
                plt.plot(fpr, tpr, dash[cl], color=colors[method][cl])
                legend_list.append(f'{methods[method]} cl={cl} (AUROC: {auroc:.2f})')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.legend(legend_list)

        try:
            os.mkdir('./plots_saved/')
        except:
            pass
        plt.savefig('./plots_saved/drift_cl_ablation.png')

class DoublePlotter():

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

    def replay_plots(self, npz_folder_root):
        self.setup_plot()
        plt.figure(1, figsize=self.figsize)

        f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})


        all_tnrs = []
        methods = {'ours': 'Ours', 'NTU': 'Feng et al.', 'Vanderbilt': 'Cai et al.', 'Beta-VAE': 'Ramakrishna et al.'}
        colors = {'ours': 'b', 'NTU': 'r', 'Vanderbilt': 'c', 'Beta-VAE': 'k'}
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
            a0.plot(fpr, tpr, color=colors[method])
            legend_list.append(f'{method}')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        # plt.legend(legend_list, loc='lower center')

        # Shrink current axis's height by 10% on the bottom
        box = a0.get_position()
        a0.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        # Put a legend below current axis
        a0.legend(legend_list, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

        plt.xticks(rotation=10)

        barlist=a1.bar(methods.values(), all_tnrs)
        for i, key in enumerate(methods.keys()):
            barlist[i].set_color(colors[key])
        plt.ylabel('TNR (at 95% TPR)')

        # f.tight_layout()
        try:
            os.mkdir('./plots_saved/')
        except:
            pass
        f.savefig('./plots_saved/replay.png')

        
            

        
        
        
        

        
        
        
