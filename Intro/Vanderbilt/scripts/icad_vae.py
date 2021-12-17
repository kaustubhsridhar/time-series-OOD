import numpy as np
from scipy import stats
import torch
from scripts.network import VAE

class ICAD_VAE():
    # offline
    def __init__(self, calibration_data=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load the pretrained VAE model
        try:
            self.net = VAE()
            self.net = self.net.to(self.device)
            self.net.load_state_dict(torch.load("./models/vae.pt", map_location=self.device))
            self.net.eval()
            print("Loaded the pretrained vae model...")
        except:
            print("Cannot find the pretrained model, please train it first...")

        # Compute or load the precomputed nonconformity scores for calibration data
        if calibration_data: # compute
            inputs = np.rollaxis(self.calibrationData,3,1)
            errors = []
            for i in range(len(inputs)):
                input_torch = torch.from_numpy(np.expand_dims(inputs[i], axis=0)).float()
                input_torch = input_torch.to(self.device)
                output, _, _ = self.net(input_torch)
                rep = output.cpu().data.numpy()
                reconstrution_error = (np.square(rep.reshape(1, -1) - inputs[i].reshape(1, -1))).mean(axis=1)
                errors.append(reconstrution_error)
                self.calibration_NC = np.array(errors)
                np.save("./data/nc_calibration_vae.npy", self.calibration_NC)
        else: # load
            try:
                self.calibration_NC = np.load("./data/nc_calibration_vae.npy")
            except:
                print("Cannot find precomputed nonconformity scores, please provide the calibration data set")
            
    # online
    def __call__(self, image):
        image = np.expand_dims(image, axis=0)
        image = np.rollaxis(image, 3, 1)
        input_torch = torch.from_numpy(image).float()
        input_torch = input_torch.to(self.device)
        output, _, _ = self.net(input_torch)
        rep = output.cpu().data.numpy()
        reconstrution_error = (np.square(rep.reshape(1, -1) - image.reshape(1, -1))).mean(axis=1)
        p = (100 - stats.percentileofscore(self.calibration_NC, reconstrution_error))/float(100)
        return p