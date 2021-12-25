import numpy as np

def make2D(scores_flattened, list_of_no_of_windows_in_traces):
	assert len(scores_flattened) == sum(list_of_no_of_windows_in_traces)
	curr = 0
	scores_2D = []
	for no_windows_in_trace in list_of_no_of_windows_in_traces:
		current_trace_scores = scores_flattened[curr:curr+no_windows_in_trace]
		scores_2D.append(current_trace_scores)
		curr += no_windows_in_trace
	return scores_2D

def OOD_score_to_iD_score(list_2D):
	return [[-1*i for i in row] for row in list_2D]

def min_of_each_row(list_2D):
	return [min(row) for row in list_2D]

def collapse_to_1D(list_2D):
    list_1D = []
    for row in list_2D:
        list_1D.extend(row)
    return list_1D

def compute_epsilon_on_iD_traces_only(iD_scores_all, GTs_all, TPR=0.95):
    only_iD_traces_scores = []
    for trace_idx, trace_GT in enumerate(GTs_all):
        if trace_GT==1: # i.e if iD
            only_iD_traces_scores.append(iD_scores_all[trace_idx])
    # sort in descending order
    only_iD_traces_scores = np.array(sorted(only_iD_traces_scores, reverse=True))
    n_traces = len(only_iD_traces_scores)
    epsilon = only_iD_traces_scores[int(TPR*n_traces)-1]
    print('No of iD traces: {} | TPR {} location: {}'.format(n_traces, TPR, int(TPR*n_traces)-1))
    print("iD traces' scores: ", only_iD_traces_scores)
    print('epsilon: ', epsilon)
    return epsilon

def scan_iD_scores_of_windows(iD_scores_2D_list, epsilon):
    det_delays = []
    not_detected_as_OOD = 0
    for row in iD_scores_2D_list:
        for window_idx, val in enumerate(row):
            if val<epsilon:
                det_delays.append(window_idx)
                break
            if window_idx == len(row)-1:
                not_detected_as_OOD += 1
    if len(det_delays) !=0:
        avg_det_delay = sum(det_delays)/len(det_delays)
    else:
        avg_det_delay = -1
    TN = len(det_delays)
    FP = not_detected_as_OOD
    TNR = TN / (TN + FP)
    assert (TN + FP)==len(iD_scores_2D_list)
    return avg_det_delay, TNR

def scan_iD_scores_of_windows_and_print_list(iD_scores_2D_list, epsilon):
    det_delays = []
    not_detected_as_OOD = 0
    detected_as_OOD = 0
    for row in iD_scores_2D_list:
        for window_idx, val in enumerate(row):
            if val<epsilon:
                detected_as_OOD += 1
                det_delays.append(window_idx)
                break
            if window_idx == len(row)-1:
                not_detected_as_OOD += 1
                det_delays.append(-1)
    TN = detected_as_OOD
    FP = not_detected_as_OOD
    TNR = TN / (TN + FP)
    print('Window detected idx: ', det_delays)
    return -1, TNR