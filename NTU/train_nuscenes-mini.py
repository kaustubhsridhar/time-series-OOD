#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from network import Bi3DOF, Encoder, Decoder
from datasets import *
from utils import progress_bar

'''

Train routine for bi3dof and bi3dofopt methods in the paper "Improving Variational Autoencoder 
based Out-of-Distribution Detection for Embedded Real-time Applications" 

A bi3dof detector is trained by the default parameter values. 

To train a bi3dof-optprior detector, first compute the mu and var of optic flow fields of the training data set. 
Then run the python script with arguments --latentprior optimal and mu&var values as described in help of init_param().

More technical details please see in paper link: https://arxiv.org/abs/2107.11750

'''

def init_param():
	parser = argparse.ArgumentParser() 
	parser.add_argument("--episode_size", type=int, default = 5, help="number of videos in one mini-batch")	 
	# parser.add_argument("--n_seq", type=int, default = 6, help="number of sequence/window to sample from one video") 
	# Changed above to use fixed number of random sequences (aka windows) in each video (aka trace) every epoch to train (by changing datasets file to use list of n_seq values ==> helpful to test variable number of sequences at test time)
	parser.add_argument("--nd", type=int, default = 16, help="number of frames in a 3D input cube")    # prev: default=6; changed to 16 based on Ramneet's choice of frames in a window/sequence
	parser.add_argument("--group", type=int, default = 2, 
			help="2 refer to the two latent subspaces, in horizontal and vertical direction respectively.")  
	parser.add_argument("--nz", type=int, default = 12, help="Dimension of one latent subspace")	  
	parser.add_argument("--latentprior", type=str, default='simple', 
			help="If 'optimal' the follows mu and var values will be used to compute distribution descrepancy in latent space.")	
	parser.add_argument("--mu1", type=float, default=0., 
			help="Mean optic flow fields in the horizontal direction. Need to be computed standalone from a specific trainig set.") 
	parser.add_argument("--mu2", type=float, default=0., help="Mean optic flow value in the vertical direction")  
	parser.add_argument("--var1", type=float, default=1.,help="Varaince in the horizontal direction") 
	parser.add_argument("--var2", type=float, default=1., help="Varaince in the horizontal? (vertical) direction")			   
	args = parser.parse_args()

	# min_of_n_seqs = min([f - args.nd for f in frame_lens["train"]]) # 114
	# args.n_seqs = [min_of_n_seqs for _ in range(len(frame_lens["train"]))] # New: changed to list
	args.n_seqs = [6 for _ in range(len(frame_lens["train"]))]

	# input dimension as in # [g,d,h,w]
	args.input_size = [args.group, args.nd, 120,160]   
	args.transform_size = [113,152]   

	# default values 
	args.epochs = 600
	args.lr_base = 0.0001  
	args.kl_weight = 1
	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	return args

def run():
	# load data
	train_id = Bi3DOFDataset(args)
	train_size = len(train_id)	
	train_loader = torch.utils.data.DataLoader(train_id, batch_size=args.episode_size, shuffle=True, num_workers=1)

	# initialize network and optimizor
	encoder = Encoder(args)
	decoder = Decoder(args)			
	vae = Bi3DOF(encoder, decoder, args).to(args.device)
	enc_optimizer = optim.Adam(vae.encoder.parameters(), lr= args.lr_base)
	dec_optimizer = optim.Adam(vae.decoder.parameters(), lr= args.lr_base) 

	for epoch in range(args.epochs):
		loss_reconstruction, loss_latent, num_examples = 0,0,0
		vae.train()
		for batch_idx, batch_data in enumerate(train_loader):   
			batch_data = batch_data.to(args.device)
			(b1,b2,g,d,h,w) = batch_data.shape
			batch_data = batch_data.view((b1*b2,g,d,h,w))
			num_examples += b1*b2
		  
			enc_optimizer.zero_grad()
			dec_optimizer.zero_grad()
			loss, loss_rec, (loss_latent_grp1, loss_latent_grp2) = vae.loss(batch_data, args.kl_weight)
			loss = loss.mean(dim=-1)
			loss.backward()
			enc_optimizer.step()
			dec_optimizer.step()

			loss_reconstruction += loss_rec.view(-1).mean()
			loss_latent += (loss_latent_grp1.view(-1).mean() + loss_latent_grp2.view(-1).mean())

			progress_bar(batch_idx, len(train_loader), 'Epoch%3d  Recontruction/Loss: %.6f  Latent/Loss: %.6f' 
										% (epoch, loss_reconstruction/num_examples, loss_latent/num_examples) )		   
	# save 
	torch.save(vae.state_dict(), "bi3dof-{}-{}epoch-{}seq.pt".format(args.latentprior,  epoch+1, args.n_seqs[0]) )

#
seed()
# obtain below from feature_abstraction
frame_lens = {'train': [148, 130, 130, 148, 130, 148, 130, 130, 148, 130, 130, 130, 130, 148, 148, 130, 148, 130, 148, 148]
				}
args = init_param()
args.training = True
args.data_file = "../NTU_features/train.train" # "data/nuscenes-v1.0-mini.train"
run()
