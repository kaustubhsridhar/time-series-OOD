import numpy as np
import argparse
import torch
from torch.cuda import current_blas_handle
from network import Bi3DOF, Encoder, Decoder
from datasets import seed, Bi3DOFDataset
from more_utils import make2D, OOD_score_to_iD_score, min_of_each_row, compute_epsilon_on_iD_traces_only, scan_iD_scores_of_windows, scan_iD_scores_of_windows_and_print_list
import os
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

'''
	Inputs are test clips are hdf5 files prodiced byfeature_abstraction.py.
	Place all test clips in folder data/nuscenes-v1.0-mini.test.

	Outputs are scores for OoD detection.
'''
def load_model(test_config): 
	parser = argparse.ArgumentParser() 	   
	args = parser.parse_args()
	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Default values 
	args.training = False
	args.kl_weight = 1
	args.group = 2	
	args.nz = 12	 
	args.nd = 16
	args.mu1, args.mu2 =0 , 0
	# test config
	args.latentprior = test_config["network"]
	args.var1, args.var2 = 1, 1

	# Input dimension as in # [grp,nd,h,w]
	args.input_size = [args.group, args.nd, 120,160]   
	args.transform_size = [113,152]   

	# Assifn test clips
	args.data_file = test_config["test_clips"]
	# Sequentially advance 1 frame per step 
	args.n_seqs = [f - args.nd for f in test_config["frames_per_clip"]] # New: changed to list # Changed to test all sequences(aka window) in video(aka trace). Number of sequences(aka windows) is total frames-frame size(aka nd = 16)

	# Load  weights  
	encoder = Encoder(args)
	decoder = Decoder(args)
	model = Bi3DOF(encoder, decoder, args).to(args.device)   
	model.load_state_dict(torch.load(test_config["model_file"], map_location = args.device))
	model.eval()
	return model, args

def compute_score(model, args):
	args.batch_size = 1
	testset = Bi3DOFDataset(args)    
	test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
	d_horizontal = []
	d_vertical = []
	model.eval()
	with torch.no_grad():
		for b_idx, (batch_data) in enumerate(test_loader): 
			data = batch_data.to(args.device)
			(b1,b2,g,d,h,w) = data.shape  
			data = data.view((b1*b2,g,d,h,w))
			_, (d_grp1, d_grp2) = model.encode(data)
			for i in range(len(d_grp1)):
				d_horizontal.append(d_grp1[i].cpu().numpy()) 
				d_vertical.append(d_grp2[i].cpu().numpy())
	return d_horizontal, d_vertical

seed()
# obtain below from feature_abstraction
frame_lens = {'train': [148, 130, 130, 148, 130, 148, 130, 130, 148, 130, 130, 130, 130, 148, 148, 130, 148, 130, 148, 148], 
				'in': [123, 122, 123, 121, 124, 123, 121, 121, 122, 123, 123, 121, 123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122], 
				'out_foggy': [123, 122, 123, 121, 124, 123, 121, 121, 122, 123, 123, 121, 123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122], 
				'out_night': [123, 122, 123, 121, 124, 123, 121, 121, 122, 123, 123, 121, 123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122], 
				'out_snowy': [123, 122, 123, 121, 124, 123, 121, 121, 122, 123, 123, 121, 123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122], 
				#'out_rainy_old': [111, 141, 142, 112, 114, 141, 140, 141, 141, 140, 111, 112, 111, 113, 114, 111, 114, 141, 114, 111, 116, 141, 112, 122, 112, 141, 111, 112, 141, 141, 112, 112, 111, 116, 142, 140, 111, 116, 111, 116, 116, 114, 113, 111, 142, 115, 114, 111, 141, 116, 122, 114, 114, 141, 112, 141, 114, 141, 111, 111, 111, 113, 111, 114, 111, 141, 116, 111, 122, 117, 111, 111, 111],
				'out_rainy': [141, 122, 112, 114, 141, 140, 141, 140, 111, 112, 111, 113, 111, 114, 111, 114, 141, 114, 111, 116, 111, 114, 141, 111, 111, 141, 142]}

bi3dof_simple_test_in = {
    "model_file" : "bi3dof-simple-600epoch-6seq.pt", # "model/nuscenes-mini/bi3dof-simple-600epoch.pt",
    "network" : "simple",   
    "test_clips": "../NTU_features_all/in.test", # "data/nuscenes-v1.0-mini.test",      
    "frames_per_clip": frame_lens['in']
}

def getOutBi3DOF(type_of_OOD):
	features_folder = "../NTU_features_rainy_only/" # "../NTU_features_all/" # Change to "../NTU_features_rainy_only/" for rainy
	bi3dof_simple_test_out = {
		"model_file" : "bi3dof-simple-600epoch-6seq.pt", # "model/nuscenes-mini/bi3dof-simple-600epoch.pt",
		"network" : "simple",   
		"test_clips": features_folder+"{}.test".format(type_of_OOD), # "data/nuscenes-v1.0-mini.test",      
		"frames_per_clip": frame_lens[type_of_OOD]
	}
	return bi3dof_simple_test_out
##### Plan is below
# OOD_scores_flattened = [OOD_score(w1 of t1), ..., OOD_score(wN of t1), OOD_score(w1 of t2), ..., OOD_score(wN of t2), ..., OOD_score(w1 of tN), ..., OOD_score(wN of tN)]
# OOD_scores = [[OOD_score(w1 of t1), ..., OOD_score(wN of t1)], [OOD_score(w1 of t2), ..., OOD_score(wN of t2)], ..., [OOD_score(w1 of tN), ..., OOD_score(wN of tN)]] # by unflattening
# iD_scores = [[iD_score(w1 of t1), ..., iD_score(wN of t1)], [iD_score(w1 of t2), ..., iD_score(wN of t2)], ..., [iD_score(w1 of tN), ..., iD_score(wN of tN)]] # do negation here
# iD_scores_final = [iD_score(t1), iD_score(t2), ..., iD_score(tN)] # take minimums here
# iD_scores_final, [GT(t1), ..., GT(tN)] --> roc_curve [GT is 0 for OOD and 1 for iD]
# epsilon = 95% smallest position of iD_scores_final's iD traces only
# scan iD_scores for first location in each OOD (not iD) trace when iD_score < epsilon --> gives detection delay and TNR

def run(type_of_OOD):
	print('\n\n\n', type_of_OOD, '\n\n')
	# ROC curve calculation
	iD_scores_all = []; GTs_all = []
	for idx, bi3dof_simple in enumerate([bi3dof_simple_test_in, getOutBi3DOF(type_of_OOD)]): # i.e. for traces in [iD traces, OOD traces]
		model, args = load_model(bi3dof_simple)
		h,v = compute_score(model, args)
		OOD_scores_flattened = [h[i]+v[i] for i in range(len(h))]
		OOD_scores_2D_list = make2D(OOD_scores_flattened, args.n_seqs)
		iD_scores_2D_list = OOD_score_to_iD_score(OOD_scores_2D_list)
		iD_scores_for_each_trace = min_of_each_row(iD_scores_2D_list)
		if ('in' in bi3dof_simple["test_clips"]) and ('rainy' not in bi3dof_simple["test_clips"]):
			GT = 1
		else:
			GT = 0
		GTs_for_each_trace = [GT for _ in range(len(iD_scores_for_each_trace))]

		if GT==0:
			iD_scores_2D_list_of_OOD_traces_only = iD_scores_2D_list

		iD_scores_all.extend(iD_scores_for_each_trace)
		GTs_all.extend(GTs_for_each_trace)

	fpr, tpr, threshs = roc_curve(GTs_all, iD_scores_all)
	auroc = roc_auc_score(GTs_all, iD_scores_all)
	plt.figure()
	plt.plot(fpr, tpr)
	plt.legend(['ROC curve (AUROC: {})'.format(auroc)])
	try:
		os.mkdir('./plots/')
	except:
		pass
	plt.savefig('./plots/plot_{}.png'.format(type_of_OOD))

	# detection delay, TNR = TN/(TN+FP) calculation
	epsilon = compute_epsilon_on_iD_traces_only(iD_scores_all, GTs_all)
	#det_delay, TNR = scan_iD_scores_of_windows(iD_scores_2D_list_of_OOD_traces_only, epsilon)
	det_delay, TNR = scan_iD_scores_of_windows_and_print_list(iD_scores_2D_list_of_OOD_traces_only, epsilon) # only uncomment this for 'out_rainy'

	print('Average detection delay for OOD traces: ', det_delay)
	print('TNR: ', TNR)

if __name__ == "__main__":
	for type_of_OOD in ['out_snowy', 'out_foggy', 'out_night', 'out_rainy']:
		run(type_of_OOD)