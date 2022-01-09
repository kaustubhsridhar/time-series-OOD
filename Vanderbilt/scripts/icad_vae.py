import numpy as np
from scipy import stats
import torch
from scripts.network import VAE
from torch.utils.data import DataLoader
import cv2

class ICAD_VAE():
    # offline
    def __init__(self, data_type, epoch=None, calibration_data=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_type = data_type
        # load the pretrained VAE model
        try:
            self.net = VAE()
            self.net = self.net.to(self.device)
            if data_type == "carla":
                model_path = "./carla_models/vae.pt"
                calib_save_path = "./carla_models/nc_calibration_vae.npy"
            elif data_type == "drift":
                if epoch:
                    model_path = f"./drift_models/vae_{epoch}.pt"
                    calib_save_path = f"./drift_models/nc_calibration_vae_{epoch}.npy"
                else:
                    print('mention epoch for drift data!')
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.net.eval()
            print("Loaded the pretrained vae model...")
        except:
            print("Cannot find the pretrained model, please train it first...")

        # Compute or load the precomputed nonconformity scores for calibration data
        if calibration_data is not None: # compute
            # inputs = np.rollaxis(self.calibrationData,3,1)
            calibration_dataloader = DataLoader(calibration_data, batch_size=1, shuffle=False) # New
            errors = []
            # for i in range(len(inputs)):
            for batch_idx, (input_torch, _) in enumerate(calibration_dataloader): # New
                # input_torch = torch.from_numpy(np.expand_dims(inputs[i], axis=0)).float()
                input_torch = input_torch.to(self.device)
                output, _, _ = self.net(input_torch)
                rep = output.cpu().data.numpy()
                input_npy = input_torch.cpu().data.numpy() # New # can add .detach().copy() after input_torch but no backprop so not needed
                # reconstrution_error = (np.square(rep.reshape(1, -1) - inputs[i].reshape(1, -1))).mean(axis=1)
                reconstrution_error = (np.square(rep.reshape(1, -1) - input_npy.reshape(1, -1))).mean(axis=1) # New
                errors.append(reconstrution_error)
                self.calibration_NC = np.array(errors)

                np.save(calib_save_path, self.calibration_NC)
        else: # load
            try:
                self.calibration_NC = np.load(calib_save_path)
            except:
                print("Cannot find precomputed nonconformity scores, please provide the calibration data set")
            
    # online
    # def __call__(self, image):
    def __call__(self, input_torch, idx, debug = False):
        # image = np.expand_dims(image, axis=0)
        # image = np.rollaxis(image, 3, 1)
        # input_torch = torch.from_numpy(image).float()
        input_torch = input_torch.to(self.device)
        output, _, _ = self.net(input_torch)
        rep = output.cpu().data.numpy()
        image = input_torch.cpu().data.numpy() # New 
        if debug: # saving image to debug
            print(idx)
            temp_im1 = rep[0]
            temp_im2 = image[0]
            concat_reconstructed_and_orig_image = np.concatenate((temp_im1, temp_im2), axis=2)
            transpo_image = concat_reconstructed_and_orig_image.transpose((1, 2, 0))
            cv2.imwrite(f'./{self.data_type}_models/data/{idx}.jpg', transpo_image*255)

        reconstrution_error = (np.square(rep.reshape(1, -1) - image.reshape(1, -1))).mean(axis=1)
        p = (100 - stats.percentileofscore(self.calibration_NC, reconstrution_error))/float(100)
        return p