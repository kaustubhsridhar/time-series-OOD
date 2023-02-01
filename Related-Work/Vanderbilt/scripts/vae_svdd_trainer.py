from scripts.network import VAE, SVDD
import torch
import numpy as np
import torch.optim as optim
import time
from torch.utils.data import TensorDataset, DataLoader
from scripts.carla import CARLADataset
from scripts.drift import driftDataset
from torchvision import transforms



class Trainer():
    def __init__(self, data_type): #X_train, y_train):
        self.data_type = data_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        chosen_transforms = transforms.Compose([transforms.ToTensor()])
        if data_type == "carla":
            root_dir = "/home/yangy96/interpretable_ood_detection/carla_experiments/carla_leave_train/"
            dataset = CARLADataset(root_dir, new_size = (224, 224), transform=chosen_transforms)
        elif data_type == "drift":
            root_dir = "../drift_data/training/"
            dataset = driftDataset(root_dir, new_size = (224, 224), transform=chosen_transforms)
        elif data_type == "lidar":
            root_dir = "../drift_data/training/"
            dataset = lidarDataset(root_dir, new_size = (224, 224), transform=chosen_transforms)


        # tensor_x = torch.Tensor(np.rollaxis(X_train, 3, 1))
        # tensor_y = torch.Tensor(y_train)
        # dataset = TensorDataset(tensor_x, tensor_y)
        self.data_loader = DataLoader(dataset, batch_size=32, shuffle=True) 

        self.nu = 0.1
        self.R = 0.0
        self.c = None
        self.episode_vae = 350 # attempted increase from orig 350 to 500
        self.lr_milestones_vae = [250]
        self.episode_svdd = 350
        self.lr_milestones_svdd = [250]

    def train_vae(self):
        optimizer = optim.Adam(self.vae.parameters(), lr=0.0001, weight_decay=0.5e-6, amsgrad=False) 
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones_vae, gamma=0.1)
        self.vae.train()
        for epoch in range(self.episode_vae):
            loss_epoch = 0.0
            reconstruction_loss_epoch = 0.0
            kl_loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                inputs = inputs.to(self.device)
                optimizer.zero_grad()

                x, mu, logvar = self.vae(inputs)
                reconstruction_loss = torch.sum((x-inputs)**2, dim=tuple(range(1, x.dim())))
                kl_loss = 1 + logvar - (mu).pow(2) - logvar.exp()
                kl_loss = torch.sum(kl_loss, axis=-1) * -0.5
                loss = reconstruction_loss + kl_loss
                reconstruction_loss_mean = torch.mean(reconstruction_loss)
                kl_loss_mean = torch.mean(kl_loss)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                reconstruction_loss_epoch += reconstruction_loss_mean.item()
                kl_loss_epoch += kl_loss_mean.item()
                n_batches += 1

            if (epoch+1) % 10 == 0:
                torch.save(self.vae.state_dict(), f"./{self.data_type}_models/vae_{epoch+1}.pt")
            
            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            print('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(epoch+1, self.episode_vae, epoch_train_time, loss_epoch/n_batches))

    def train_svdd(self):
        c = torch.tensor(self.c, device='cuda') if self.c is not None else None
        optimizer = optim.Adam(self.svdd.parameters(), lr=0.0001, weight_decay=0.5e-6, amsgrad=False)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones_svdd, gamma=0.1)
        self.svdd.train()

        if c == None:
            c = self.init_center_c()

        for epoch in range(self.episode_svdd):
            print('LR is: {}'.format(float(scheduler.get_lr()[0])))
            if epoch in self.lr_milestones_svdd:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for batch_idx, (inputs, target) in enumerate(self.data_loader):
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                outputs = self.svdd(inputs)
                dist = torch.sum((outputs - c)**2, dim=1)
                loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1
        
            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            print('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(epoch+1, self.episode_svdd, epoch_train_time, loss_epoch/n_batches))
        return c

    def init_center_c(self, eps=0.1):
        n_sample = 0
        c = torch.zeros(self.svdd.rep_dim, device=self.device)

        self.svdd.eval()
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.data_loader):
                inputs = inputs.to(self.device)
                outputs = self.svdd(inputs)
                n_sample += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        
        c /= n_sample

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c 
        
    
    def fit(self, train_svdd_also = False):
        # vae 
        if self.data_type == 'carla':
            self.vae = VAE()
            self.vae = self.vae.to(self.device)
        else:
            self.vae = lidarVAE()
            self.vae = self.vae.to(self.device)

        # trying to load pretrained VAE model
        try:
            self.vae.load_state_dict(torch.load(f"./{self.data_type}_models/vae.pt"))
            self.vae.eval()
            print("loaded pretrained VAE model")

        # train VAE model
        except:
            print("cannot find the pretrained VAE model, retrain(ing) it")
            self.train_vae()
            torch.save(self.vae.state_dict(), f"./{self.data_type}_models/vae.pt")
        
        if train_svdd_also:
            self.svdd = SVDD()
            self.svdd = self.svdd.to(self.device)
            self.init_network_weights_from_pretraining()

            c = self.train_svdd()
            torch.save(self.svdd.state_dict(), f"./{self.data_type}_models/svdd.pt")
            np.save(f"./{self.data_type}_models/svdd_c.npy", c.cpu().data.numpy())
    
    def init_network_weights_from_pretraining(self):
        svdd_net_dict = self.svdd.state_dict()
        vae_net_dict = self.vae.state_dict()

        vae_net_dict = {k: v for k, v in vae_net_dict.items() if k in svdd_net_dict}
        svdd_net_dict.update(vae_net_dict)
        self.svdd.load_state_dict(svdd_net_dict)