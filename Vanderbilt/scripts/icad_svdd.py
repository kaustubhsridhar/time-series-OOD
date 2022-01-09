import numpy as np
from scipy import stats
import torch
from scripts.network import SVDD
from torch.utils.data import DataLoader

class ICAD_SVDD():
    # offline
    def __init__(self, data_type, calibration_data=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load the pretrained SVDD model
        try:
            self.net = SVDD()
            self.net = self.net.to(self.device)
            if data_type == "carla":
                model_path = "./carla_models/svdd.pt"
                npy_path = "./carla_models/svdd_c.npy"
                calib_save_path = "./carla_models/nc_calibration_svdd.npy"
            elif data_type == "drift":
                model_path = "./drift_models/svdd.pt"
                npy_path = "./drift_models/svdd_c.npy"
                calib_save_path = "./drift_models/nc_calibration_svdd.npy"
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.net.eval()
            self.center = np.load(npy_path)
            print("Loaded the pretrained svdd model...")
        except:
            print("Cannot find the pretrained model, please train it first...")

        # Compute or load the precomputed nonconformity scores for calibration data
        if calibration_data is not None: # compute
            # inputs = np.rollaxis(self.calibrationData,3,1)
            calibration_dataloader = DataLoader(calibration_data, batch_size=1, shuffle=False) # New
            dists = []
            # for i in range(len(inputs)):
            for batch_idx, (input_torch, _) in enumerate(calibration_dataloader): # New
                # input_torch = torch.from_numpy(np.expand_dims(inputs[i], axis=0)).float()
                input_torch = input_torch.to(self.device)
                output = self.net(input_torch)
                rep = output.cpu().data.numpy()
                dist = np.sum((rep - self.center)**2, axis=1)
                dists.append(dist)
                self.calibration_NC = np.array(dists)
                np.save(calib_save_path, self.calibration_NC)
        else: # load
            try:
                self.calibration_NC = np.load(calib_save_path)
            except:
                print("Cannot find precomputed nonconformity scores, please provide the calibration data set")
            
    # online
    # def __call__(self, image):
    def __call__(self, input_torch):
        # image = np.expand_dims(image, axis=0)
        # image = np.rollaxis(image, 3, 1)
        # input_torch = torch.from_numpy(image).float()
        input_torch = input_torch.to(self.device)
        output = self.net(input_torch)
        rep = output.cpu().data.numpy()
        dist = np.sum((rep - self.center)**2, axis=1)
        p = (100 - stats.percentileofscore(self.calibration_NC, dist))/float(100)
        return p