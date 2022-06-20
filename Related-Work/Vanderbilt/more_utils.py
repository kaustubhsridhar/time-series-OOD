import numpy as np

def getTNR(in_scores, out_scores):
    in_scores, out_scores = np.array(in_scores), np.array(out_scores)

    in_fisher = np.sort(in_scores)[::-1] # sorting in descending order
    tau = in_fisher[int(0.95*len(in_scores))] # TNR at 95% TPR
    tnr = 100*(len(out_scores[out_scores<tau])/len(out_scores))

    return tnr, tau

def get_det_delay_for_detected_traces(scores_2D_list, tau):
   det_delays = []
   for trace_idx, row in enumerate(scores_2D_list):
       for window_idx, val in enumerate(row):
           if val<tau:
               det_delays.append(window_idx)
               break
   if len(det_delays) == 0:
       return -1
   avg_det_delay = sum(det_delays)/len(det_delays)
   return avg_det_delay

def OOD_score_to_iD_score(list_1D):
	return [-1*i for i in list_1D]

def make2D(scores_flattened, list_of_no_of_windows_in_traces):
	assert len(scores_flattened) == sum(list_of_no_of_windows_in_traces)
	curr = 0
	scores_2D = []
	for no_windows_in_trace in list_of_no_of_windows_in_traces:
		current_trace_scores = scores_flattened[curr:curr+no_windows_in_trace]
		scores_2D.append(current_trace_scores)
		curr += no_windows_in_trace
	return scores_2D