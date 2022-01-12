import numpy as np
from more_utils import make2D, get_det_delay_for_detected_traces, getTNR
from sklearn.metrics import roc_curve, roc_auc_score

carla_frame_lens = {'train': [148, 130, 130, 148, 130, 148, 130, 130, 148, 130, 130, 130, 130, 148, 148, 130, 148, 130, 148, 148], 
				'in': [123, 122, 123, 121, 124, 123, 121, 121, 122, 123, 123, 121, 123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122], 
				'out_foggy': [123, 122, 123, 121, 124, 123, 121, 121, 122, 123, 123, 121, 123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122], 
				'out_night': [123, 122, 123, 121, 124, 123, 121, 121, 122, 123, 123, 121, 123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122], 
				'out_snowy': [123, 122, 123, 121, 124, 123, 121, 121, 122, 123, 123, 121, 123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122], 
				#'out_rainy_old': [111, 141, 142, 112, 114, 141, 140, 141, 141, 140, 111, 112, 111, 113, 114, 111, 114, 141, 114, 111, 116, 141, 112, 122, 112, 141, 111, 112, 141, 141, 112, 112, 111, 116, 142, 140, 111, 116, 111, 116, 116, 114, 113, 111, 142, 115, 114, 111, 141, 116, 122, 114, 114, 141, 112, 141, 114, 141, 111, 111, 111, 113, 111, 114, 111, 141, 116, 111, 122, 117, 111, 111, 111],
				'out_rainy': [141, 122, 112, 114, 141, 140, 141, 140, 111, 112, 111, 113, 111, 114, 111, 114, 141, 114, 111, 116, 111, 114, 141, 111, 111, 141, 142],
				'out_replay': [50, 50, 50, 50, 50, 51, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]}

def transform_vals(np_array):
    return -1*np.exp(-1*np_array)

sliding_window_len = 6

for type_of_OOD in ['out_replay', 'out_snowy', 'out_foggy', 'out_night', 'out_rainy']:
    second_half = type_of_OOD.split('_')[-1]
    S_values_for_out_points_only = np.load(f'../npz_saved_sliding_{sliding_window_len}/{second_half}_win_out_Beta-VAE.npy')
    S_values_for_in_points_only = np.load(f'../npz_saved_sliding_{sliding_window_len}/{second_half}_win_in_Beta-VAE.npy')
    
    S_values_for_out_points_only = list(S_values_for_out_points_only)
    S_values_for_in_points_only = list(S_values_for_in_points_only)

    # print(len(S_values_for_in_points_only), sum(carla_frame_lens["in"]), len(S_values_for_out_points_only), sum(carla_frame_lens[type_of_OOD]))

    S_values_2D_list_for_out_points_only = make2D(S_values_for_out_points_only, carla_frame_lens[type_of_OOD])

    all_S_values = S_values_for_in_points_only + S_values_for_out_points_only
    GTs = [1 for _ in range(len(S_values_for_in_points_only))] + [0 for _ in range(len(S_values_for_out_points_only))]

    auroc = roc_auc_score(GTs, all_S_values)
    TNR, tau = getTNR(S_values_for_in_points_only, S_values_for_out_points_only)
    det_delay = get_det_delay_for_detected_traces(S_values_2D_list_for_out_points_only, tau)
    print(type_of_OOD, '\n')
    print(f'VAE: (AUROC, TNR, Avg Det Delay): ({auroc}, {TNR}, {det_delay}) \n\n')
