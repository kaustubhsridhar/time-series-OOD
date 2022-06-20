from scripts.martingales import SMM
from scripts.detector import StatefulDetector, StatelessDetector
import numpy as np
import argparse
import os
from scripts.carla import CARLA_test_Dataset, CARLA_calib_Dataset
from scripts.drift import drift_test_Dataset, driftDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score
from more_utils import getTNR, get_det_delay_for_detected_traces, make2D, OOD_score_to_iD_score

parser = argparse.ArgumentParser(description='Out-of-distribution detection offline.')
# parser.add_argument("-o", "--out", help="set test images from the out-of-distribution test set", action="store_true", default=False)
parser.add_argument("-v", "--vae", help="set nonconformity measure as VAE-based method", action="store_true", default=False)
parser.add_argument("-N", help="sliding window size for SVDD; # of generated examples for VAE", type=int, default=16)
parser.add_argument("-d", "--data", help="choose from [carla, drift]")
args = parser.parse_args()

carla_frame_lens = {'train': [148, 130, 130, 148, 130, 148, 130, 130, 148, 130, 130, 130, 130, 148, 148, 130, 148, 130, 148, 148], 
				'in': [123, 122, 123, 121, 124, 123, 121, 121, 122, 123, 123, 121, 123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122], 
				'out_foggy': [123, 122, 123, 121, 124, 123, 121, 121, 122, 123, 123, 121, 123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122], 
				'out_night': [123, 122, 123, 121, 124, 123, 121, 121, 122, 123, 123, 121, 123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122], 
				'out_snowy': [123, 122, 123, 121, 124, 123, 121, 121, 122, 123, 123, 121, 123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122], 
				#'out_rainy_old': [111, 141, 142, 112, 114, 141, 140, 141, 141, 140, 111, 112, 111, 113, 114, 111, 114, 141, 114, 111, 116, 141, 112, 122, 112, 141, 111, 112, 141, 141, 112, 112, 111, 116, 142, 140, 111, 116, 111, 116, 116, 114, 113, 111, 142, 115, 114, 111, 141, 116, 122, 114, 114, 141, 112, 141, 114, 141, 111, 111, 111, 113, 111, 114, 111, 141, 116, 111, 122, 117, 111, 111, 111],
				'out_rainy': [141, 122, 112, 114, 141, 140, 141, 140, 111, 112, 111, 113, 111, 114, 111, 114, 141, 114, 111, 116, 111, 114, 141, 111, 111, 141, 142],
				'out_replay': [50, 50, 50, 50, 50, 51, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]}

drift_frame_lens = {'train': [49, 74, 49, 49, 49, 49, 59, 59, 59, 59, 59, 49, 59, 59, 59, 59, 59, 49, 49, 49, 49, 48, 49, 49], 
				'in': [49, 89, 59, 59, 59, 59, 59, 59, 59, 59, 59, 49, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 49, 59, 59, 59, 59, 59, 49, 49, 49, 49, 49, 49], 
				'out': [59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 89, 89, 59, 59, 59, 47, 47, 71, 47, 47, 47, 47, 49, 47, 47, 47, 47, 47, 47, 47, 47, 47, 71, 49, 47, 47, 47, 47, 47, 47, 47, 47, 59, 59, 49, 59, 89, 59, 59, 59, 59, 59, 59, 59, 59, 49, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 49, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 49, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 49, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59]
				}

def run(type_of_OOD, count, saved_model_epoch=None):
    chosen_transforms = transforms.Compose([transforms.ToTensor()])
    # load the test images
    if args.data == "carla":
        OOD_path = f"../carla_data/testing/{type_of_OOD}/out/"
        iD_path = "../carla_data/testing/in/"
        calib_path = "../carla_data/training/"
        test_dataset = CARLA_test_Dataset(OOD_dir=OOD_path, iD_dir=iD_path, new_size=(224,224), transform=chosen_transforms)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)    
        if count == 0:
            calibration_dataset = CARLA_calib_Dataset(root_dir=calib_path, new_size=(224,224), transform=chosen_transforms)
        else:
            calibration_dataset = None

        frame_lens = carla_frame_lens
    elif args.data == "drift":
        OOD_path = f"../drift_data/testing/{type_of_OOD}/"
        iD_path = "../drift_data/testing/in/"
        calib_path = "../drift_data/calibration/"
        test_dataset = drift_test_Dataset(OOD_dir=OOD_path, iD_dir=iD_path, new_size=(224,224), transform=chosen_transforms)
        # test_dataset = driftDataset(root_dir=OOD_path, new_size=(224,224), transform=chosen_transforms)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        if count == 0:
            calibration_dataset = driftDataset(root_dir=calib_path, new_size=(224,224), transform=chosen_transforms)
        else:
            calibration_dataset = None
        
        frame_lens = drift_frame_lens

    if args.vae:
        from scripts.icad_vae import ICAD_VAE
        if args.data == "carla":
            icad = ICAD_VAE(data_type=args.data, calibration_data=calibration_dataset)
        elif args.data == "drift":
            icad = ICAD_VAE(data_type=args.data, epoch = saved_model_epoch, calibration_data=calibration_dataset)
        detector = StatefulDetector(sigma=6, tau=156)
        GTs = []
        S_values = []
        S_values_for_in_points_only = []
        S_values_for_out_points_only = []

        tot = len(test_dataset)
        for idx, (test_image, is_test_image_iD) in enumerate(test_dataloader):
            print(f'{idx+1}/{tot}...')
            smm = SMM(args.N)
            for i in range(args.N):
                p = icad(test_image, idx)
                m = np.log(smm(p))
            S, d = detector(m) # Use S value
            GTs.append(is_test_image_iD)
            S_values.append(S)
            if is_test_image_iD:
                S_values_for_in_points_only.append(S)
            else:
                S_values_for_out_points_only.append(S)
            # print("Time step: {}\t p-value: {}\t logM: {}\t S: {}\t Detector: {}".format(idx, round(p,3), round(m, 3), round(S, 3), d))
        S_values = OOD_score_to_iD_score(S_values)
        S_values_for_in_points_only = OOD_score_to_iD_score(S_values_for_in_points_only)
        S_values_for_out_points_only = OOD_score_to_iD_score(S_values_for_out_points_only)
        S_values_2D_list_for_out_points_only = make2D(S_values_for_out_points_only, frame_lens[type_of_OOD])

        try:
            os.mkdir('./npz_saved/')
        except:
            pass
        second_half_of_type_of_OOD = type_of_OOD.split('_')[-1]
        np.save(f'./npz_saved/{second_half_of_type_of_OOD}_win_in_Vanderbilt', S_values_for_in_points_only)
        np.save(f'./npz_saved/{second_half_of_type_of_OOD}_win_out_Vanderbilt', S_values_for_out_points_only)

        auroc = roc_auc_score(GTs, S_values)
        TNR, tau = getTNR(S_values_for_in_points_only, S_values_for_out_points_only)
        det_delay = get_det_delay_for_detected_traces(S_values_2D_list_for_out_points_only, tau)
        print(f'VAE: (AUROC, TNR, Avg Det Delay): ({auroc}, {TNR}, {det_delay}) \n')
        


    else:
        from scripts.icad_svdd import ICAD_SVDD

        icad = ICAD_SVDD(data_type=args.data, calibration_data=calibration_dataset)
        smm = SMM(args.N)
        detector = StatelessDetector(tau=14)
        GTs = []
        M_values = []
        M_values_for_in_points_only = []
        M_values_for_out_points_only = []

        for idx, (test_image, is_test_image_iD) in enumerate(test_dataloader):
            p = icad(test_image)
            m = np.log(smm(p))
            d = detector(m)
            GTs.append(is_test_image_iD)
            M_values.append(m)
            if is_test_image_iD:
                M_values_for_in_points_only.append(m)
            else:
                M_values_for_out_points_only.append(m)
            # print("Time step: {}\t p-value: {}\t logM: {}\t Detector: {}".format(idx, round(p,3), round(m, 3), d))
        M_values = OOD_score_to_iD_score(M_values)
        M_values_for_in_points_only = OOD_score_to_iD_score(M_values_for_in_points_only)
        M_values_for_out_points_only = OOD_score_to_iD_score(M_values_for_out_points_only)
        M_values_2D_list_for_out_points_only = make2D(M_values_for_out_points_only, frame_lens[type_of_OOD])

        auroc = roc_auc_score(GTs, M_values)
        TNR, tau = getTNR(M_values_for_in_points_only, M_values_for_out_points_only)
        det_delay = get_det_delay_for_detected_traces(M_values_2D_list_for_out_points_only, tau)
        print(f'SVDD: (AUROC, TNR, Avg Det Delay): ({auroc}, {TNR}, {det_delay}) \n')


if __name__ == "__main__":
    count = 0
    if args.data == "carla":
        for type_of_OOD in ['out_replay']:#, 'out_snowy', 'out_foggy', 'out_night', 'out_rainy']:
            print(type_of_OOD)
            run(type_of_OOD, count)
            count += 1
    elif args.data == "drift":
        print('out')
        run("out", count, saved_model_epoch=10)